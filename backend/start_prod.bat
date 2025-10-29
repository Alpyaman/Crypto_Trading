@echo off
echo ⚠️  PRODUCTION MODE - LIVE TRADING
echo This uses real money and live API keys
echo.
echo Make sure you have:
echo - Live Binance API keys configured
echo - Sufficient account balance
echo - Risk management settings reviewed
echo.
set /p confirm="Are you sure you want to start LIVE TRADING? (yes/no): "
if /i "%confirm%"=="yes" (
    python start.py --env production
) else (
    echo Cancelled
)
pause