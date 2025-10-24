"""
Test Binance API credentials
"""
from app.services.binance_service import BinanceService
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def test_credentials():
    """Test if Binance API credentials work"""
    print("🔍 Testing Binance API Credentials")
    print("=" * 40)
    
    # Get credentials from environment
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    testnet = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
    
    print(f"API Key: {api_key[:10]}...{api_key[-10:] if api_key else 'None'}")
    print(f"Testnet Mode: {testnet}")
    
    if not api_key or not api_secret:
        print("❌ API credentials not found in .env file")
        return False
    
    try:
        # Initialize Binance service
        print("\n📡 Connecting to Binance API...")
        binance_service = BinanceService(api_key, api_secret, testnet=testnet)
        
        # Test 1: Get account info
        print("🔍 Testing account access...")
        balance = binance_service.get_account_balance()
        if balance is not None:
            print("✅ Account access successful!")
            print(f"Found {len(balance)} assets in account")
            
            # Show some balances
            for asset, info in list(balance.items())[:5]:
                if info['total'] > 0:
                    print(f"  {asset}: {info['total']}")
        else:
            print("❌ Failed to get account balance")
            return False
        
        # Test 2: Get market data
        print("\n📈 Testing market data access...")
        btc_price = binance_service.get_current_price("BTCUSDT")
        if btc_price:
            print("✅ Market data access successful!")
            print(f"  BTC/USDT Price: ${btc_price:,.2f}")
        else:
            print("❌ Failed to get market data")
            return False
        
        # Test 3: Get 24h ticker
        print("\n📊 Testing ticker data...")
        ticker = binance_service.get_24h_ticker("BTCUSDT")
        if ticker:
            print("✅ Ticker data access successful!")
            print(f"  24h Change: {ticker['change_percent']:.2f}%")
            print(f"  24h Volume: {ticker['volume']:,.0f}")
        else:
            print("❌ Failed to get ticker data")
            return False
        
        print("\n🎉 All tests passed! Your API credentials are working correctly.")
        
        if testnet:
            print("\n⚠️  Note: You're using TESTNET mode.")
            print("   This is perfect for development and testing!")
            print("   No real money will be used in trades.")
        else:
            print("\n⚠️  WARNING: You're using LIVE trading mode!")
            print("   Real money will be used in trades.")
            print("   Make sure you understand the risks!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ API test failed: {e}")
        print("\nPossible solutions:")
        print("1. Check if your API keys are correct")
        print("2. Ensure API permissions include 'Read Info' and 'Enable Spot & Margin Trading'")
        print("3. Check if your IP is whitelisted (if IP restrictions are enabled)")
        print("4. Verify if you're using testnet keys with testnet=true")
        return False


if __name__ == "__main__":
    test_credentials()