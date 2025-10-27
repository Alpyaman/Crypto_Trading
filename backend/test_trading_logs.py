"""
Test Enhanced Trading Logging
Quick test to verify the enhanced trading service logging is working
"""
import requests
import json
import time

def test_trading_status():
    """Test the enhanced trading status endpoint"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Enhanced Trading Status...")
    
    try:
        # Check enhanced trading status (correct path)
        response = requests.get(f"{base_url}/enhanced/trading/status")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Enhanced Trading Status:")
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ Status check failed: {response.status_code}")
            print(response.text)
        
        # Also check the regular trading endpoints
        print("\n💼 Checking Regular Trading Status...")
        regular_response = requests.get(f"{base_url}/api/trading/status")
        
        if regular_response.status_code == 200:
            regular_data = regular_response.json()
            print("📊 Regular Trading Status:")
            print(json.dumps(regular_data, indent=2))
        
        # Check account information
        print("\n💰 Checking Futures Account...")
        account_response = requests.get(f"{base_url}/api/account/futures")
        
        if account_response.status_code == 200:
            account_data = account_response.json()
            account_info = account_data.get('account_info', {})
            print("💳 Account Summary:")
            print(f"   Balance: ${account_info.get('total_wallet_balance', 0):.2f}")
            print(f"   Available: ${account_info.get('available_balance', 0):.2f}")
            print(f"   Positions: {account_data.get('position_count', 0)}")
        
        # Check if training is running
        print("\n🤖 Checking ML Training Status...")
        training_response = requests.get(f"{base_url}/api/ml/training-progress")
        
        if training_response.status_code == 200:
            training_data = training_response.json()
            print("📊 Training Status:")
            print(f"   Is Training: {training_data.get('isTraining', False)}")
            print(f"   Progress: {training_data.get('progress', 0)}%")
            print(f"   Status: {training_data.get('status', 'unknown')}")
            
            if training_data.get('isTraining'):
                print(f"   Timesteps: {training_data.get('currentTimestep', 0)}/{training_data.get('totalTimesteps', 0)}")
                print(f"   Time Elapsed: {training_data.get('timeElapsed', '00:00:00')}")
                print(f"   Time Remaining: {training_data.get('timeRemaining', '00:00:00')}")
        
    except Exception as e:
        print(f"💥 Error: {e}")

def monitor_logs():
    """Monitor trading logs by checking status repeatedly"""
    print("📊 Starting Trading Log Monitor...")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            test_trading_status()
            print("\n" + "="*50)
            time.sleep(30)  # Check every 30 seconds
    except KeyboardInterrupt:
        print("\n🛑 Log monitoring stopped")

if __name__ == "__main__":
    print("🔍 Enhanced Trading Log Test")
    print("1. Single status check")
    print("2. Monitor logs every 30 seconds")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        monitor_logs()
    else:
        test_trading_status()