"""
Quick AI Trader Starter
"""
import requests

def start_trading():
    """Start AI trading"""
    print("🚀 Starting AI Crypto Trader...")
    
    try:
        # Start trading
        url = "http://localhost:8000/api/v1/trading/start"
        data = {
            "symbol": "BTCUSDT", 
            "mode": "conservative"
        }
        
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ AI Trading Started Successfully!")
            print(f"   Symbol: {result.get('message', 'BTCUSDT')}")
            print(f"   Mode: {result.get('mode', 'conservative')}")
            return True
        else:
            print(f"❌ Failed to start trading: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_status():
    """Check trading status"""
    try:
        response = requests.get("http://localhost:8000/api/v1/trading/status")
        if response.status_code == 200:
            status = response.json()
            is_trading = status.get('is_trading', False)
            total_trades = status.get('total_trades', 0)
            
            print(f"\n📊 Trading Status: {'🟢 ACTIVE' if is_trading else '🔴 STOPPED'}")
            print(f"📋 Total Trades: {total_trades}")
            
            current_position = status.get('current_position')
            if current_position:
                print("💼 Current Position:")
                print(f"   Symbol: {current_position.get('symbol')}")
                print(f"   Side: {current_position.get('side')}")
                print(f"   Quantity: {current_position.get('quantity')}")
                print(f"   Entry Price: ${current_position.get('entry_price', 0):,.2f}")
            
            return status
        else:
            print(f"❌ Failed to get status: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error getting status: {e}")
        return None

def main():
    print("🤖 AI Crypto Trader Quick Start")
    print("=" * 40)
    
    # Check current status
    print("🔍 Checking current status...")
    status = check_status()
    
    if status and status.get('is_trading'):
        print("\n✅ AI Trader is already running!")
        print("💡 Use the dashboard (python dashboard.py) to monitor")
    else:
        print("\n🚀 Starting AI Trader...")
        success = start_trading()
        
        if success:
            print("\n🎉 SUCCESS! Your AI Crypto Trader is now LIVE!")
            print("\n🎯 What's happening now:")
            print("   • AI analyzes Bitcoin market every 5 minutes")
            print("   • Makes BUY/SELL/HOLD decisions based on ML model")
            print("   • Uses conservative risk management (safe for testing)")
            print("   • Trades with your testnet funds (10,000 USDT)")
            print("   • You can monitor progress in real-time")
            
            print("\n📊 Monitor Commands:")
            print("   • python dashboard.py (comprehensive dashboard)")
            print("   • curl http://localhost:8000/api/v1/trading/status")
            print("   • curl http://localhost:8000/api/v1/trading/history")
            
            print("\n🛑 To Stop Trading:")
            print("   • curl -X POST http://localhost:8000/api/v1/trading/stop")
            
            print("\n💡 Tips:")
            print("   • First trades may take 5-10 minutes as AI analyzes market")
            print("   • Conservative mode only trades with high confidence")
            print("   • This is testnet - no real money at risk!")
            
        else:
            print("\n❌ Failed to start AI trader")
            print("💡 Check if the server is running: python -m app.main")

if __name__ == "__main__":
    main()