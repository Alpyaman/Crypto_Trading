@echo off
echo 🚀 Starting Crypto Trading AI with WebSocket Support
echo.
echo Features:
echo - Real-time training progress updates
echo - Live market data streaming  
echo - Enhanced error handling
echo - Professional UI with loading states
echo.
echo Starting backend server...
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause