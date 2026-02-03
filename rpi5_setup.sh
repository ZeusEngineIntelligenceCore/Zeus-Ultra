#!/bin/bash
# ============================================
# ZEUS TRADING BOT - Raspberry Pi 5 Setup
# ============================================

set -e

echo "============================================"
echo "  ZEUS TRADING BOT - RPi5 Setup"
echo "============================================"
echo ""

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

# Install dependencies
echo "[SETUP] Installing Python dependencies..."
pip install flask flask-login gunicorn pytz aiohttp numpy python-telegram-bot

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
export SECRET_KEY="generate_a_random_secret_key_here"
EOF
    echo ""
    echo "[IMPORTANT] Edit .env file with your actual credentials!"
    echo ""
else
    echo "[SETUP] .env file already exists"
fi

# Create systemd service file
echo "[SETUP] Creating systemd service file..."
cat > zeus.service << EOF
[Unit]
Description=Zeus Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
EnvironmentFile=$(pwd)/.env
ExecStart=$(pwd)/zeus_env/bin/python $(pwd)/main_rpi5.py
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
echo "   source zeus_env/bin/activate"
echo "   source .env"
echo "   python main_rpi5.py"
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
echo "Dashboard will be available at: http://YOUR_PI_IP:5000"
echo ""
