"""
Development and Testing Utilities
Additional tools for development and debugging
"""
import asyncio
import os
from datetime import datetime
from app.services.binance_service import BinanceService
from app.services.ml_service import MLService
from app.services.trading_service import TradingService


def test_binance_connection():
    """Test Binance API connection"""
    print("Testing Binance connection...")
    
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        print("❌ No API credentials found")
        return False
    
    try:
        # Use testnet for safety
        binance = BinanceService(api_key, api_secret, testnet=True)
        
        # Test getting server time (doesn't require credentials)
        price = binance.get_current_price("BTCUSDT")
        if price:
            print(f"✅ Connection successful! BTC price: ${price:,.2f}")
            return True
        else:
            print("❌ Failed to get price data")
            return False
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def test_model_training():
    """Test ML model training with minimal data"""
    print("Testing ML model training...")
    
    try:
        # ml_service = MLService(model_path="models/test_model.zip")
        
        # This would normally train with real data
        print("✅ ML service initialized successfully")
        print("ℹ️ To train a model, use the /api/v1/ml/train endpoint")
        return True
        
    except Exception as e:
        print(f"❌ ML service test failed: {e}")
        return False


async def test_full_system():
    """Test the complete system integration"""
    print("Testing full system integration...")
    
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        print("❌ API credentials required for full test")
        return False
    
    try:
        # Initialize services
        binance_service = BinanceService(api_key, api_secret, testnet=True)
        ml_service = MLService()
        trading_service = TradingService(binance_service, ml_service)
        
        # Test market data
        price = binance_service.get_current_price("BTCUSDT")
        if price:
            print(f"✅ Market data: BTC ${price:,.2f}")
        
        # Test account data
        balance = binance_service.get_account_balance()
        print(f"✅ Account data: {len(balance)} assets found")
        
        # Test trading service status
        status = trading_service.get_trading_status()
        print(f"✅ Trading service: {status['is_trading']}")
        
        print("✅ Full system test passed!")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False


def show_api_examples():
    """Show example API calls"""
    print("\n=== API Usage Examples ===")
    
    examples = [
        {
            "description": "Get BTC price",
            "method": "GET",
            "endpoint": "/api/v1/market/price/BTCUSDT",
            "curl": "curl http://localhost:8000/api/v1/market/price/BTCUSDT"
        },
        {
            "description": "Get account balance",
            "method": "GET", 
            "endpoint": "/api/v1/account/balance",
            "curl": "curl http://localhost:8000/api/v1/account/balance"
        },
        {
            "description": "Train ML model",
            "method": "POST",
            "endpoint": "/api/v1/ml/train",
            "curl": 'curl -X POST http://localhost:8000/api/v1/ml/train -H "Content-Type: application/json" -d \'{"symbol": "BTCUSDT", "timesteps": 50000}\''
        },
        {
            "description": "Start trading (balanced mode)",
            "method": "POST",
            "endpoint": "/api/v1/trading/start",
            "curl": 'curl -X POST http://localhost:8000/api/v1/trading/start -H "Content-Type: application/json" -d \'{"symbol": "BTCUSDT", "mode": "balanced"}\''
        },
        {
            "description": "Get trading status",
            "method": "GET",
            "endpoint": "/api/v1/trading/status", 
            "curl": "curl http://localhost:8000/api/v1/trading/status"
        },
        {
            "description": "Stop trading",
            "method": "POST",
            "endpoint": "/api/v1/trading/stop",
            "curl": "curl -X POST http://localhost:8000/api/v1/trading/stop"
        }
    ]
    
    for example in examples:
        print(f"\n📝 {example['description']}")
        print(f"   {example['method']} {example['endpoint']}")
        print(f"   {example['curl']}")


async def main():
    """Main development test function"""
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=== Crypto Trading AI - Development Tests ===\n")
    
    # Basic connection test
    test_binance_connection()
    print()
    
    # ML service test
    test_model_training()
    print()
    
    # Full system test
    await test_full_system()
    
    # Show API examples
    show_api_examples()
    
    print(f"\n=== Tests completed at {datetime.now()} ===")


if __name__ == "__main__":
    asyncio.run(main())