@echo off
REM ==========================================
REM Stage 1: Environment Setup Script (Windows)
REM ==========================================

echo Creating Python virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Setup complete! To activate the environment, run:
echo     call venv\Scripts\activate.bat
echo.
echo Then copy .env.example to .env and fill in your DEEPSEEK_API_KEY:
echo     copy .env.example .env
echo.
