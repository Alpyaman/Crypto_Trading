"""
Complete test suite for the Crypto Trading AI
"""
import requests
import time

class CompleteTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def wait_for_server(self, max_attempts: int = 10) -> bool:
        """Wait for server to be ready"""
        print("⏳ Waiting for server to be ready...")
        
        for attempt in range(max_attempts):
            try:
                response = self.session.get(f"{self.base_url}/", timeout=5)
                if response.status_code == 200:
                    print("✅ Server is ready!")
                    return True
            except Exception:
                pass
            
            print(f"   Attempt {attempt + 1}/{max_attempts}...")
            time.sleep(2)
        
        print("❌ Server not responding")
        return False
    
    def test_full_functionality(self):
        """Run complete functionality test"""
        print("\n🧪 Running Complete Functionality Test")
        print("=" * 50)
        
        # Test 1: Health check
        print("1️⃣ Testing health endpoint...")
        try:
            health = self.session.get(f"{self.base_url}/").json()
            print(f"   Status: {health.get('status')}")
            print(f"   Environment: {health.get('environment')}")
            print(f"   Testnet: {health.get('testnet_mode')}")
            
            services = health.get('services', {})
            for service, status in services.items():
                icon = "✅" if status else "❌"
                print(f"   {service}: {icon}")
            
            model_status = health.get('model_status', {})
            model_loaded = model_status.get('loaded', False)
            print(f"   ML Model: {'✅' if model_loaded else '❌'}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
        
        # Test 2: Configuration
        print("\n2️⃣ Testing configuration...")
        try:
            config = self.session.get(f"{self.base_url}/config").json()
            creds_configured = config.get('api_credentials_configured', False)
            print(f"   API Credentials: {'✅' if creds_configured else '❌'}")
            print(f"   Environment: {config.get('environment')}")
            print(f"   Testnet Mode: {config.get('testnet_mode')}")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            return False
        
        # Test 3: Market data
        print("\n3️⃣ Testing market data...")
        try:
            # Test price endpoint
            price_resp = self.session.get(f"{self.base_url}/api/v1/market/price/BTCUSDT")
            if price_resp.status_code == 200:
                price_data = price_resp.json()
                print(f"   ✅ BTC Price: ${price_data.get('price', 0):,.2f}")
            else:
                print(f"   ❌ Price endpoint failed: {price_resp.status_code}")
            
            # Test ticker endpoint
            ticker_resp = self.session.get(f"{self.base_url}/api/v1/market/ticker/BTCUSDT")
            if ticker_resp.status_code == 200:
                ticker_data = ticker_resp.json()
                print(f"   ✅ 24h Change: {ticker_data.get('change_percent', 0):.2f}%")
            else:
                print(f"   ❌ Ticker endpoint failed: {ticker_resp.status_code}")
                
        except Exception as e:
            print(f"   ❌ Market data failed: {e}")
        
        # Test 4: ML endpoints
        print("\n4️⃣ Testing ML endpoints...")
        try:
            ml_status = self.session.get(f"{self.base_url}/api/v1/ml/status").json()
            model_loaded = ml_status.get('model_loaded', False)
            print(f"   Model Status: {'✅ Loaded' if model_loaded else '❌ Not loaded'}")
            
            if not model_loaded:
                print("   💡 You can train a model with: POST /api/v1/ml/train")
            else:
                # Test prediction
                pred_resp = self.session.post(f"{self.base_url}/api/v1/ml/predict/BTCUSDT")
                if pred_resp.status_code == 200:
                    pred_data = pred_resp.json()
                    print(f"   ✅ Prediction: {pred_data.get('prediction')} (confidence: {pred_data.get('confidence', 0):.2f})")
                else:
                    print(f"   ⚠️ Prediction not available: {pred_resp.status_code}")
                    
        except Exception as e:
            print(f"   ❌ ML endpoints failed: {e}")
        
        # Test 5: Trading endpoints
        print("\n5️⃣ Testing trading endpoints...")
        try:
            trading_status = self.session.get(f"{self.base_url}/api/v1/trading/status").json()
            is_trading = trading_status.get('is_trading', False)
            print(f"   Trading Status: {'🟢 Active' if is_trading else '🔴 Stopped'}")
            
            total_trades = trading_status.get('total_trades', 0)
            print(f"   Total Trades: {total_trades}")
            
        except Exception as e:
            print(f"   ❌ Trading endpoints failed: {e}")
        
        # Summary
        print("\n📋 Test Complete!")
        print(f"🔗 API Docs: {self.base_url}/docs")
        print(f"🔗 Alternative Docs: {self.base_url}/redoc")
        
        return True

def main():
    tester = CompleteTester()
    
    # Wait for server
    if not tester.wait_for_server():
        print("\n💡 Make sure the server is running:")
        print("   python -m app.main")
        return
    
    # Run tests
    tester.test_full_functionality()
    
    print("\n🎯 Next Steps:")
    print("1. Train ML Model: curl -X POST 'http://localhost:8000/api/v1/ml/train' -H 'Content-Type: application/json' -d '{\"symbol\": \"BTCUSDT\", \"timesteps\": 10000}'")
    print("2. Start Trading: curl -X POST 'http://localhost:8000/api/v1/trading/start' -H 'Content-Type: application/json' -d '{\"symbol\": \"BTCUSDT\", \"mode\": \"conservative\"}'")
    print("3. Monitor Trading: curl 'http://localhost:8000/api/v1/trading/status'")

if __name__ == "__main__":
    main()