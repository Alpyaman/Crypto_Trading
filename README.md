# Crypto Trading AI Application

Advanced cryptocurrency trading bot with machine learning capabilities using PPO reinforcement learning and comprehensive real-time analytics.

## Features

- 🤖 **Enhanced ML System**: 86-feature PPO algorithm with VecNormalize for stable training
- 📊 **Real-Time Analytics**: Live market data, futures trading, and comprehensive dashboard
- 🎯 **Multiple Environments**: Development (testnet), staging, production, and testing modes
- 📈 **Advanced Charts**: Real-time price visualization with Chart.js integration
- � **Transparent Trading**: Detailed 5-minute analysis cycles with decision logging
- 📱 **Modern Web Interface**: Dark-themed dashboard with live progress tracking
- �️ **Enhanced Security**: Environment-specific configurations and risk management

## Quick Start

### Development Mode (Testnet - Recommended for testing)
```bash
cd backend
start_dev.bat
```

### Production Mode (Live Trading) ⚠️ **USE WITH CAUTION**
```bash
cd backend
start_prod.bat
```

### Access Dashboard
Open your browser to the URL shown in the console (typically `http://localhost:8000`)

## Environment Configuration

The system supports multiple deployment environments:

- **Development** (`.env.development`): Safe testnet trading with debug logging
- **Production** (`.env.production`): Live trading with enhanced security measures  
- **Staging** (`.env.staging`): Production-like features on testnet
- **Test** (`.env.test`): Mock APIs for automated testing

## Project Structure

```
backend/
├── app/
│   ├── main.py                    # FastAPI application entry point
│   ├── config.py                  # Application configuration
│   ├── models/
│   │   ├── enhanced_futures_env.py   # Enhanced 86-feature trading environment
│   │   └── trading_env.py            # Original trading environment
│   ├── services/
│   │   ├── binance_service.py        # Binance API integration
│   │   ├── enhanced_ml_service.py    # Advanced ML with VecNormalize
│   │   ├── enhanced_trading_service.py # Comprehensive trading with logging
│   │   ├── ml_service.py             # Original ML service
│   │   ├── trading_service.py        # Original trading service
│   │   ├── training_callbacks.py     # Training progress tracking
│   │   └── training_state.py         # Training state management
│   ├── api/
│   │   ├── routes.py                 # Original API routes
│   │   └── enhanced_routes.py        # Enhanced API with progress tracking
│   └── utils/
│       └── env_loader.py             # Environment configuration loader
├── models/checkpoints/               # Saved ML models and training data
├── .env.*                           # Environment-specific configurations
├── start*.bat                       # Environment startup scripts
└── requirements.txt
frontend/
├── index.html                       # Main dashboard interface
├── script.js                       # Dashboard JavaScript logic
└── styles.css                      # Modern dark theme styling
```

## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Crypto_Trading
   ```

2. **Set up Python environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   # source .venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Configure your environment**
   - Choose your environment (development recommended for first-time users)
   - Configure API credentials in the appropriate `.env.*` file
   - For testnet (safe): Use `.env.development`
   - For live trading: Use `.env.production` with real API keys

5. **Start the application**
   ```bash
   # For safe testnet trading (recommended)
   start_dev.bat
   
   # For live trading (advanced users only)
   start_prod.bat
   ```

6. **Access the dashboard**
   Open your browser to the URL displayed in the console

## API Credentials Configuration

### Development (Testnet) - Recommended for testing
Edit `.env.development`:
```env
BINANCE_API_KEY=your_testnet_api_key
BINANCE_SECRET_KEY=your_testnet_secret_key
USE_TESTNET=true
```

### Production (Live Trading) ⚠️ **Real Money**
Edit `.env.production`:
```env
BINANCE_API_KEY=your_live_api_key
BINANCE_SECRET_KEY=your_live_secret_key
USE_TESTNET=false
```

## Dashboard Features

### Real-Time Market Analysis
- Live BTCUSDT price charts with technical indicators
- Real-time futures account balance display
- Position tracking and profit/loss monitoring

### ML Training Progress
- Training step progression with live updates
- Loss metrics and convergence monitoring  
- Model performance tracking over time

### Trading Transparency
- Detailed 5-minute analysis cycles
- Decision reasoning and risk assessments
- Comprehensive trading logs with timestamps

### Risk Management
- Real-time risk calculations
- Position sizing based on account balance
- Stop-loss and take-profit automation

## Security & Best Practices

1. **Always test on testnet first** using development mode
2. **Never commit API credentials** - they're automatically excluded via `.gitignore`
3. **Start with small amounts** when moving to production
4. **Monitor trading logs** regularly for unexpected behavior
5. **Use appropriate risk settings** for your risk tolerance

## Troubleshooting

### Common Issues

**API Connection Errors**
- Verify API credentials in your `.env` file
- Check if using correct testnet/mainnet keys
- Ensure IP address is whitelisted on Binance

**Training Not Starting**
- Check ML service logs in the dashboard
- Verify sufficient disk space for model checkpoints
- Restart the application if training appears stuck

**Dashboard Not Loading**
- Ensure backend is running (check console for errors)
- Try refreshing browser or clearing cache
- Check firewall settings for localhost access
   ```

3. **Install dependencies**
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp backend/.env.example backend/.env
   # Edit .env with your Binance API credentials
   ```

5. **Get Binance API credentials**
   - Create account on [Binance](https://binance.com) or [Binance Testnet](https://testnet.binance.vision/)
   - Generate API Key and Secret
   - Add them to your `.env` file

## Usage

### Starting the Server

```bash
cd backend
python -m app.main
# Or using uvicorn directly:
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The server will start on `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key API Endpoints

#### Market Data
- `GET /api/v1/market/price/{symbol}` - Get current price
- `GET /api/v1/market/ticker/{symbol}` - Get 24h ticker stats
- `GET /api/v1/market/klines/{symbol}` - Get historical data

#### Account Management
- `GET /api/v1/account/balance` - Get account balances
- `POST /api/v1/account/order` - Place manual order

#### Machine Learning
- `POST /api/v1/ml/train` - Train ML model
- `GET /api/v1/ml/status` - Get model status
- `POST /api/v1/ml/predict/{symbol}` - Get trading prediction

#### Automated Trading
- `POST /api/v1/trading/start` - Start automated trading
- `POST /api/v1/trading/stop` - Stop trading
- `GET /api/v1/trading/status` - Get trading status
- `GET /api/v1/trading/history` - Get trading history

### Training the ML Model

Before starting automated trading, you need to train the ML model:

```bash
curl -X POST "http://localhost:8000/api/v1/ml/train" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "BTCUSDT", "timesteps": 100000}'
```

### Starting Automated Trading

```bash
curl -X POST "http://localhost:8000/api/v1/trading/start" \
     -H "Content-Type: application/json" \
     -d '{"symbol": "BTCUSDT", "mode": "balanced"}'
```

## Trading Modes

1. **Conservative**: Low risk, high confidence threshold (80%), 10% position size
2. **Balanced**: Medium risk, moderate confidence (65%), 25% position size  
3. **Aggressive**: High risk, lower confidence (55%), 50% position size

## Technical Indicators Used

- **SMA** (10, 30): Simple Moving Averages
- **EMA** (12): Exponential Moving Average
- **MACD**: Moving Average Convergence Divergence
- **RSI** (14): Relative Strength Index
- **Bollinger Bands**: Volatility indicator
- **Volume**: Trading volume analysis

## Risk Management

- Position sizing based on account balance and risk parameters
- Stop-loss and take-profit levels
- Confidence-based trade filtering
- Maximum position limits per trading mode

## Environment Variables

```env
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_TESTNET=true  # Use testnet for development
DEBUG=true
LOG_LEVEL=INFO
```

## Development

### Running in Development Mode

```bash
cd backend
uvicorn app.main:app --reload
```

### Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

## Security Notes

- **Never commit your `.env` file** with real API credentials
- Use Binance testnet for development and testing
- Implement proper authentication for production use
- Consider using environment-specific configurations
- Monitor your trading bot closely, especially in live trading

## Future Enhancements

- [ ] Web-based frontend dashboard
- [ ] Multiple exchange support
- [ ] Portfolio management features
- [ ] Advanced risk management tools
- [ ] Backtesting capabilities
- [ ] Real-time notifications
- [ ] Database persistence
- [ ] User authentication and authorization

## Disclaimer

This software is for educational and research purposes. Cryptocurrency trading involves substantial risk of loss. The developers are not responsible for any financial losses incurred through the use of this software. Always test thoroughly with testnet before using real funds.

## License

MIT License - see LICENSE file for details