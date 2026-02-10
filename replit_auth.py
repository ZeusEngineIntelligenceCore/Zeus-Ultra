import jwt
import os
import time
import uuid
from functools import wraps
from urllib.parse import urlencode

import requests
from jwt import PyJWKClient

from flask import g, session, redirect, request, render_template, url_for
from flask_dance.consumer import (
    OAuth2ConsumerBlueprint,
    oauth_authorized,
    oauth_error,
)
from flask_dance.consumer.storage import BaseStorage
from flask_login import LoginManager, login_user, logout_user, current_user
from oauthlib.oauth2.rfc6749.errors import InvalidGrantError
from sqlalchemy.exc import NoResultFound
from werkzeug.local import LocalProxy

from app import app, db
from models import OAuth, User

login_manager = LoginManager(app)

ISSUER_URL = os.environ.get('ISSUER_URL', "https://replit.com/oidc")
REPL_ID = os.environ.get('REPL_ID', '')


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, user_id)


class UserSessionStorage(BaseStorage):

    def get(self, blueprint):
        if not current_user.is_authenticated:
            return None
        try:
            token = db.session.query(OAuth).filter_by(
                user_id=current_user.get_id(),
                browser_session_key=g.browser_session_key,
                provider=blueprint.name,
            ).one().token
        except NoResultFound:
            token = None
        return token

    def set(self, blueprint, token):
        if not current_user.is_authenticated:
            return
        db.session.query(OAuth).filter_by(
            user_id=current_user.get_id(),
            browser_session_key=g.browser_session_key,
            provider=blueprint.name,
        ).delete()
        new_model = OAuth()
        new_model.user_id = current_user.get_id()
        new_model.browser_session_key = g.browser_session_key
        new_model.provider = blueprint.name
        new_model.token = token
        db.session.add(new_model)
        db.session.commit()

    def delete(self, blueprint):
        if not current_user.is_authenticated:
            return
        db.session.query(OAuth).filter_by(
            user_id=current_user.get_id(),
            browser_session_key=g.browser_session_key,
            provider=blueprint.name).delete()
        db.session.commit()


def make_replit_blueprint():
    if not REPL_ID:
        raise SystemExit("the REPL_ID environment variable must be set")

    replit_bp = OAuth2ConsumerBlueprint(
        "replit_auth",
        __name__,
        client_id=REPL_ID,
        client_secret=None,
        base_url=ISSUER_URL,
        authorization_url_params={
            "prompt": "login consent",
        },
        token_url=ISSUER_URL + "/token",
        token_url_params={
            "auth": (),
            "include_client_id": True,
        },
        auto_refresh_url=ISSUER_URL + "/token",
        auto_refresh_kwargs={
            "client_id": REPL_ID,
        },
        authorization_url=ISSUER_URL + "/auth",
        use_pkce=True,
        code_challenge_method="S256",
        scope=["openid", "profile", "email", "offline_access"],
        storage=UserSessionStorage(),
    )

    @replit_bp.before_app_request
    def set_applocal_session():
        if '_browser_session_key' not in session:
            session['_browser_session_key'] = uuid.uuid4().hex
        session.modified = True
        g.browser_session_key = session['_browser_session_key']
        g.flask_dance_replit = replit_bp.session

    @replit_bp.route("/logout")
    def logout():
        try:
            del replit_bp.token
        except Exception:
            pass
        logout_user()

        end_session_endpoint = ISSUER_URL + "/session/end"
        encoded_params = urlencode({
            "client_id": REPL_ID,
            "post_logout_redirect_uri": request.url_root,
        })
        logout_url = f"{end_session_endpoint}?{encoded_params}"

        return redirect(logout_url)

    @replit_bp.route("/error")
    def error():
        return render_template("403.html"), 403

    return replit_bp


def save_user(user_claims):
    user = User()
    user.id = user_claims['sub']
    user.email = user_claims.get('email')
    user.first_name = user_claims.get('first_name')
    user.last_name = user_claims.get('last_name')
    user.profile_image_url = user_claims.get('profile_image_url')
    merged_user = db.session.merge(user)
    db.session.commit()
    return merged_user


def get_jwks_client():
    """Fetch the JWKS URI from the OIDC provider's discovery document."""
    discovery_url = f"{ISSUER_URL}/.well-known/openid-configuration"
    discovery_doc = requests.get(discovery_url, timeout=10).json()
    jwks_uri = discovery_doc["jwks_uri"]
    return PyJWKClient(jwks_uri)


@oauth_authorized.connect
def logged_in(blueprint, token):
    id_token = token['id_token']
    jwks_client = get_jwks_client()
    signing_key = jwks_client.get_signing_key_from_jwt(id_token)
    user_claims = jwt.decode(
        id_token,
        signing_key.key,
        algorithms=["RS256"],
        options={"verify_aud": False}
    )
    user = save_user(user_claims)
    login_user(user)
    blueprint.token = token
    next_url = session.pop("next_url", None)
    if next_url is not None:
        return redirect(next_url)


@oauth_error.connect
def handle_error(blueprint, error, error_description=None, error_uri=None):
    return redirect(url_for('replit_auth.error'))


def _is_api_request():
    from flask import jsonify as _jfy
    return request.path.startswith('/api/') or request.headers.get('Accept', '').startswith('application/json') or request.headers.get('X-Requested-With') == 'XMLHttpRequest'


def require_login(f):

    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            if _is_api_request():
                from flask import jsonify as _jfy
                return _jfy({"error": "Authentication required", "auth_url": "/auth/replit_auth/login"}), 401
            session["next_url"] = get_next_navigation_url(request)
            return redirect(url_for('replit_auth.login'))

        token = replit.token if replit else None
        if token:
            expires_at = token.get('expires_at', 0)
            if expires_at and time.time() > expires_at:
                refresh_token_url = ISSUER_URL + "/token"
                try:
                    new_token = replit.refresh_token(
                        token_url=refresh_token_url,
                        client_id=REPL_ID
                    )
                    replit.token = new_token
                except InvalidGrantError:
                    if _is_api_request():
                        from flask import jsonify as _jfy
                        return _jfy({"error": "Session expired", "auth_url": "/auth/replit_auth/login"}), 401
                    session["next_url"] = get_next_navigation_url(request)
                    return redirect(url_for('replit_auth.login'))

        return f(*args, **kwargs)

    return decorated_function


def get_next_navigation_url(request):
    is_navigation_url = request.headers.get(
        'Sec-Fetch-Mode') == 'navigate' and request.headers.get(
            'Sec-Fetch-Dest') == 'document'
    if is_navigation_url:
        return request.url
    return request.referrer or request.url


replit = LocalProxy(lambda: g.flask_dance_replit)
