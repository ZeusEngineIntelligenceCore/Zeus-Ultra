#!/bin/bash
# ============================================
# ZEUS TRADING BOT - Raspberry Pi 5 Setup
# Run from project root directory
# ============================================

set -e

echo "============================================"
echo "  ZEUS TRADING BOT - RPi5 Setup"
echo "============================================"
echo ""

# Check we're in the right directory
if [ ! -f "main_rpi5.py" ]; then
    echo "[ERROR] main_rpi5.py not found!"
    echo "Please run this script from the Zeus project root directory."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "[CHECK] Python version: $PYTHON_VERSION"

# Create directories
echo "[SETUP] Creating directories..."
mkdir -p data logs

# Create virtual environment if it doesn't exist
if [ ! -d "zeus_env" ]; then
    echo "[SETUP] Creating virtual environment..."
    python3 -m venv zeus_env
else
    echo "[SETUP] Virtual environment already exists"
fi

# Activate virtual environment
echo "[SETUP] Activating virtual environment..."
source zeus_env/bin/activate

# Upgrade pip
echo "[SETUP] Upgrading pip..."
pip install --upgrade pip

# Install dependencies from requirements file
echo "[SETUP] Installing Python dependencies..."
if [ -f "requirements_rpi5.txt" ]; then
    pip install -r requirements_rpi5.txt
else
    pip install flask flask-login gunicorn pytz aiohttp numpy python-telegram-bot werkzeug
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "[SETUP] Creating .env template..."
    cat > .env << 'EOF'
# ZEUS Trading Bot Environment Variables
# Edit these with your actual credentials

export KRAKEN_API_KEY="your_kraken_api_key_here"
export KRAKEN_API_SECRET="your_kraken_api_secret_here"
export TELEGRAM_BOT_TOKEN="your_telegram_bot_token_here"
export TELEGRAM_CHAT_ID="your_telegram_chat_id_here"
export TRADING_MODE="PAPER"
export PORT=5000
export SECRET_KEY="$(openssl rand -hex 32)"
EOF
    echo ""
    echo "[IMPORTANT] Edit .env file with your actual credentials!"
    echo ""
else
    echo "[SETUP] .env file already exists"
fi

# Create run script
echo "[SETUP] Creating run script..."
cat > run_zeus.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source zeus_env/bin/activate
source .env
echo "Starting Zeus Trading Bot..."
echo "Mode: $TRADING_MODE"
echo "Dashboard: http://$(hostname -I | awk '{print $1}'):${PORT:-5000}"
python main_rpi5.py
EOF
chmod +x run_zeus.sh

# Create systemd service file
ZEUS_DIR=$(pwd)
echo "[SETUP] Creating systemd service file..."
cat > zeus.service << EOF
[Unit]
Description=Zeus Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$ZEUS_DIR
EnvironmentFile=$ZEUS_DIR/.env
ExecStart=$ZEUS_DIR/zeus_env/bin/python $ZEUS_DIR/main_rpi5.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo ""
echo "============================================"
echo "  SETUP COMPLETE!"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Edit your credentials:"
echo "   nano .env"
echo ""
echo "2. Test the bot manually:"
echo "   ./run_zeus.sh"
echo ""
echo "3. (Optional) Install as a system service:"
echo "   sudo cp zeus.service /etc/systemd/system/"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl enable zeus"
echo "   sudo systemctl start zeus"
echo ""
echo "4. View service logs:"
echo "   sudo journalctl -u zeus -f"
echo ""
IP_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "YOUR_PI_IP")
echo "Dashboard will be available at: http://$IP_ADDR:5000"
echo ""
