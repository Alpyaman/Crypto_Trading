@echo off
echo Starting Crypto Trading AI Backend...
echo.

cd /d "%~dp0backend"

REM Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat

REM Install/update requirements
echo Installing requirements...
pip install -r requirements.txt

REM Check if .env file exists
if not exist ".env" (
    echo.
    echo WARNING: .env file not found!
    echo Please copy .env.example to .env and configure your API credentials.
    echo.
    pause
    exit /b 1
)

REM Start the application
echo.
echo Starting FastAPI server...
python -m app.main

pause