import requests

def test_endpoints():
    base_url = "http://localhost:8000"
    
    endpoints = [
        "/api/status",
        "/api/market-data/BTCUSDT", 
        "/api/indicators/BTCUSDT",
        "/api/ml/status",
        "/api/account/balance"
    ]
    
    for endpoint in endpoints:
        try:
            print(f"\n🔍 Testing {endpoint}")
            response = requests.get(f"{base_url}{endpoint}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: {endpoint}")
                print(f"📊 Data keys: {list(data.keys()) if isinstance(data, dict) else 'Non-dict response'}")
            else:
                print(f"❌ Error {response.status_code}: {endpoint}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"💥 Exception testing {endpoint}: {e}")

if __name__ == "__main__":
    print("🚀 Testing GUI API endpoints...")
    test_endpoints()
    print("\n✨ Testing complete!")