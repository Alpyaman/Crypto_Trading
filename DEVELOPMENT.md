# Development Guide - Crypto Trading AI

## Quick Start

1. **Test your setup**:
   ```powershell
   cd backend
   python test_setup.py
   ```

2. **Configure environment**:
   ```powershell
   copy .env.example .env
   # Edit .env with your Binance API credentials
   ```

3. **Run the application**:
   ```powershell
   # Option 1: Use the batch script
   start.bat
   
   # Option 2: Manual start
   cd backend
   .venv\Scripts\activate
   python -m app.main
   ```

4. **Access the API**:
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/

## Key Components

### 1. Services Layer (`app/services/`)

#### `binance_service.py`
- **Purpose**: Handles all Binance API interactions
- **Key methods**:
  - `get_account_balance()` - Get account balances
  - `get_current_price(symbol)` - Get current price
  - `place_market_order()` - Execute orders
  - `get_historical_klines()` - Get market data

#### `ml_service.py` 
- **Purpose**: Manages machine learning model operations
- **Key methods**:
  - `train_model()` - Train PPO reinforcement learning model
  - `load_model()` - Load existing model
  - `predict()` - Get trading predictions

#### `trading_service.py`
- **Purpose**: Orchestrates automated trading
- **Key methods**:
  - `start_trading()` - Begin automated trading
  - `stop_trading()` - Stop trading
  - `get_trading_status()` - Get current status

### 2. Models Layer (`app/models/`)

#### `trading_env.py`
- **Purpose**: Custom Gym environment for RL training
- **Features**: 
  - OHLCV data processing
  - Technical indicators (RSI, MACD, Bollinger Bands)
  - Action space: Hold, Buy, Sell
  - Portfolio state tracking

### 3. API Layer (`app/api/`)

#### `routes.py`
- **Purpose**: RESTful API endpoints
- **Route groups**:
  - `/api/v1/market/*` - Market data endpoints
  - `/api/v1/account/*` - Account management
  - `/api/v1/ml/*` - Machine learning operations
  - `/api/v1/trading/*` - Trading control

## Development Workflow

### 1. Setting Up for Development

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
copy .env.example .env
# Configure your API keys in .env
```

### 2. Testing Individual Components

```python
# Test Binance connection
from app.services.binance_service import BinanceService
service = BinanceService("api_key", "api_secret", testnet=True)
balance = service.get_account_balance()
print(balance)

# Test ML service
from app.services.ml_service import MLService
ml = MLService()
# Training requires API credentials and takes time
```

### 3. Running in Development Mode

```powershell
# Start with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. API Testing

Use the built-in Swagger UI at http://localhost:8000/docs or test with curl:

```powershell
# Get account balance
curl "http://localhost:8000/api/v1/account/balance"

# Get BTC price
curl "http://localhost:8000/api/v1/market/price/BTCUSDT"

# Start training (background task)
curl -X POST "http://localhost:8000/api/v1/ml/train" -H "Content-Type: application/json" -d "{\"symbol\": \"BTCUSDT\", \"timesteps\": 10000}"

# Get model status
curl "http://localhost:8000/api/v1/ml/status"
```

## Configuration

### Environment Variables (.env)

```env
# Required
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Development settings
BINANCE_TESTNET=true  # Use testnet for development
DEBUG=true
LOG_LEVEL=INFO
```

### Trading Modes

1. **Conservative**: 
   - Confidence threshold: 80%
   - Max position: 10% of balance
   - Stop loss: 2%, Take profit: 4%

2. **Balanced**: 
   - Confidence threshold: 65%
   - Max position: 25% of balance
   - Stop loss: 3%, Take profit: 6%

3. **Aggressive**: 
   - Confidence threshold: 55%
   - Max position: 50% of balance
   - Stop loss: 5%, Take profit: 10%

## Common Development Tasks

### Adding a New Trading Indicator

1. **Modify `trading_env.py`**:
```python
def _add_indicators(self, df):
    # Add your indicator
    df['new_indicator'] = your_calculation(df['close'])
    return df
```

2. **Update observation space** in `__init__()` if needed

3. **Test the indicator** before training

### Adding a New API Endpoint

1. **Add to `routes.py`**:
```python
@router.get("/your-endpoint")
async def your_function():
    return {"result": "data"}
```

2. **Test the endpoint** in Swagger UI

3. **Add error handling** and validation

### Customizing Trading Strategy

Modify `trading_service.py`:
- Adjust `_get_risk_parameters()` for different risk profiles
- Modify `_execute_trading_decision()` for custom logic
- Update `_calculate_position_size()` for different sizing algorithms

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Run `python test_setup.py` to check setup
   - Ensure virtual environment is activated
   - Check Python path configuration

2. **API Connection Issues**:
   - Verify API credentials in `.env`
   - Check if using testnet vs mainnet
   - Confirm Binance API permissions

3. **Model Training Failures**:
   - Check available memory (RL training is memory-intensive)
   - Verify market data is available
   - Start with smaller timesteps for testing

4. **Trading Issues**:
   - Ensure model is trained and loaded
   - Check account balance and permissions
   - Verify symbol format (e.g., "BTCUSDT")

### Debugging Tips

1. **Enable detailed logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

2. **Use testnet first**:
```env
BINANCE_TESTNET=true
```

3. **Monitor logs** in real-time during trading

4. **Test components individually** before integration

## Performance Considerations

1. **Model Training**: 
   - CPU-intensive, consider using powerful machine
   - Start with smaller timesteps (10,000) for testing
   - Training can take 30+ minutes for 100,000 timesteps

2. **Real-time Trading**:
   - API rate limits (1200 requests/minute for Binance)
   - Network latency affects order execution
   - Consider using WebSocket for real-time data

3. **Memory Usage**:
   - RL models can use significant RAM
   - Historical data storage grows over time
   - Consider data cleanup strategies

## Security Best Practices

1. **API Key Security**:
   - Never commit `.env` file to version control
   - Use testnet for development
   - Restrict API key permissions on Binance
   - Enable IP restrictions if possible

2. **Production Deployment**:
   - Use environment variables instead of `.env` file
   - Implement proper authentication
   - Use HTTPS in production
   - Monitor for suspicious activity

3. **Risk Management**:
   - Start with small amounts
   - Set maximum daily/weekly loss limits
   - Implement circuit breakers
   - Monitor positions closely

## Next Steps

1. **Enhance the ML Model**:
   - Experiment with different RL algorithms
   - Add more technical indicators
   - Implement ensemble methods

2. **Add More Features**:
   - Portfolio management across multiple assets
   - Backtesting capabilities
   - Real-time notifications
   - Web dashboard

3. **Production Readiness**:
   - Add comprehensive error handling
   - Implement proper logging and monitoring
   - Add database for persistence
   - Create deployment scripts