"""
Live API Setup Verification
Check if your Binance API is properly configured for futures trading
"""
import os
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Load environment variables
load_dotenv()

def verify_api_setup():
    """Verify Binance API setup for futures trading"""
    print("🔍 Verifying Binance API Setup for Futures Trading")
    print("=" * 60)
    
    # Check environment variables
    print("\n1. Checking Environment Variables...")
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    testnet = os.getenv("BINANCE_TESTNET", "false").lower() == "true"
    
    if not api_key:
        print("❌ BINANCE_API_KEY not found in environment")
        return False
    
    if not api_secret:
        print("❌ BINANCE_API_SECRET not found in environment")
        return False
    
    print(f"✅ API Key found: {api_key[:10]}...")
    print(f"✅ API Secret found: {'*' * 20}")
    print(f"📍 Mode: {'TESTNET' if testnet else 'LIVE'}")
    
    # Test API connection
    print("\n2. Testing API Connection...")
    try:
        client = Client(api_key, api_secret, testnet=testnet)
        
        # Test basic connection
        status = client.get_system_status()
        print(f"✅ System Status: {status['msg']}")
        
        # Test account access
        client.get_account()
        print("✅ Account Access: Connected")
        
    except BinanceAPIException as e:
        print(f"❌ API Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        return False
    
    # Test futures access
    print("\n3. Testing Futures Access...")
    try:
        # Test futures account access
        futures_account = client.futures_account()
        print("✅ Futures Account: Accessible")
        
        # Check if account has balance
        total_balance = 0
        for asset in futures_account['assets']:
            balance = float(asset['walletBalance'])
            if balance > 0:
                total_balance += balance
                print(f"   {asset['asset']}: {balance} (Available: {asset['availableBalance']})")
        
        if total_balance == 0:
            print("⚠️  Warning: No balance in futures account")
        else:
            print(f"✅ Total Futures Balance: ${total_balance:.2f}")
        
        # Test futures position access
        positions = client.futures_position_information()
        print(f"✅ Positions Access: {len(positions)} symbols available")
        
    except BinanceAPIException as e:
        if "Invalid API-key" in str(e) or "permission" in str(e).lower():
            print(f"❌ Futures Permissions Error: {e}")
            print("🔧 Solution: Enable futures trading in your API key settings")
            return False
        else:
            print(f"❌ Futures API Error: {e}")
            return False
    except Exception as e:
        print(f"❌ Futures Test Error: {e}")
        return False
    
    # Test futures trading permissions
    print("\n4. Testing Trading Permissions...")
    try:
        # Test getting symbol info (read permission)
        exchange_info = client.futures_exchange_info()
        symbols = [s for s in exchange_info['symbols'] if s['symbol'] == 'BTCUSDT']
        if symbols:
            print("✅ Market Data Access: Working")
        
        # Note: We don't test actual trading to avoid placing real orders
        print("✅ Trading Permissions: Ready (not tested to avoid real trades)")
        
    except Exception as e:
        print(f"❌ Trading Permissions Error: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 API SETUP VERIFICATION COMPLETE")
    print("=" * 60)
    
    if not testnet:
        print("🚀 READY FOR LIVE FUTURES TRADING!")
        print("⚠️  Remember:")
        print("   • Start with small positions")
        print("   • Use conservative mode initially")
        print("   • Monitor trades closely")
        print("   • Only risk what you can afford to lose")
    else:
        print("🧪 Connected to testnet (spot trading only)")
        print("💡 To enable futures, switch to live API in .env file")
    
    return True

if __name__ == "__main__":
    success = verify_api_setup()
    if not success:
        print("\n❌ Setup verification failed. Please check the LIVE_API_SETUP.md guide.")
    else:
        print("\n✅ Setup verification successful! Ready to trade.")