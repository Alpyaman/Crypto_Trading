"""
Test script to verify timeout fixes
"""
import requests

def test_api_endpoints():
    """Test the new health check endpoints"""
    base_url = "http://localhost:8000/api/v1"
    
    print("🧪 Testing API Health Endpoints")
    print("=" * 50)
    
    # Test basic health
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Basic Health: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Basic Health: Failed - {e}")
    
    # Test Binance health
    try:
        response = requests.get(f"{base_url}/health/binance", timeout=10)
        print(f"✅ Binance Health: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Binance Health: Failed - {e}")
    
    # Test detailed status
    try:
        response = requests.get(f"{base_url}/health/binance/status", timeout=10)
        print(f"✅ Binance Status: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Binance Status: Failed - {e}")
    
    # Test price endpoint with better error handling
    print("\n📊 Testing Price Endpoint")
    print("-" * 30)
    try:
        response = requests.get(f"{base_url}/market/price/BTCUSDT", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Price: ${data['price']:,.2f}")
        else:
            print(f"⚠️ Price Error: {response.status_code} - {response.text}")
    except requests.exceptions.Timeout:
        print("⏰ Price request timed out (expected behavior during network issues)")
    except Exception as e:
        print(f"❌ Price Error: {e}")

if __name__ == "__main__":
    test_api_endpoints()