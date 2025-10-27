"""
Quick test script to verify the futures account API is working correctly
"""
import requests
import json

def test_futures_account():
    """Test the futures account endpoint"""
    try:
        print("🧪 Testing futures account API...")
        response = requests.get('http://localhost:8000/api/account/futures')
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response:")
            print(json.dumps(data, indent=2))
            
            # Check specific values
            account_info = data.get('account_info', {})
            print("\n📊 Account Summary:")
            print(f"   Total Wallet Balance: ${account_info.get('total_wallet_balance', 0):.2f}")
            print(f"   Available Balance: ${account_info.get('available_balance', 0):.2f}")
            print(f"   Used Margin: ${account_info.get('used_margin', 0):.2f}")
            print(f"   Unrealized PnL: ${account_info.get('total_unrealized_pnl', 0):.2f}")
            print(f"   Positions: {data.get('position_count', 0)}")
            
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure it's running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_futures_account()