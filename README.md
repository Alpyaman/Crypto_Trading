# Crypto Trading AI Application

A sophisticated cryptocurrency trading application powered by machine learning and real-time market data from Binance.

## Features

- 🤖 **AI-Powered Trading**: Uses PPO (Proximal Policy Optimization) reinforcement learning for intelligent trading decisions
- 📊 **Real-Time Market Data**: Integration with Binance API for live market data and order execution
- 🎯 **Multiple Trading Modes**: Conservative, Balanced, and Aggressive trading strategies
- 📈 **Technical Indicators**: RSI, MACD, Bollinger Bands, and moving averages
- 🔄 **Automated Trading**: Fully automated trading with customizable risk parameters
- 📱 **REST API**: Complete RESTful API for integration with frontends or external systems
- 🛡️ **Risk Management**: Built-in position sizing, stop-loss, and take-profit mechanisms

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── models/
│   │   ├── __init__.py
│   │   └── trading_env.py   # RL trading environment
│   ├── services/
│   │   ├── __init__.py
│   │   ├── binance_service.py   # Binance API integration
│   │   ├── ml_service.py        # ML model management
│   │   └── trading_service.py   # Trading orchestration
│   └── api/
│       ├── __init__.py
│       └── routes.py        # API routes
├── requirements.txt
├── .env.example
└── .env
```

## Installation

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