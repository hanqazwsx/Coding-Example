#!/usr/bin/env bash
# ==========================================
# Stage 1: Environment Setup Script (Linux/Mac)
# ==========================================

set -e

echo "Creating Python virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete! To activate the environment, run:"
echo "    source venv/bin/activate"
echo ""
echo "Then copy .env.example to .env and fill in your DEEPSEEK_API_KEY:"
echo "    cp .env.example .env"
echo ""
