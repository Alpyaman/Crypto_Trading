"""
Training Progress Monitor
Monitor ML model training progress and guide next steps
"""
import requests
import time

class TrainingMonitor:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_model_status(self):
        """Check if model training is complete"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/ml/status")
            if response.status_code == 200:
                status = response.json()
                return status.get('model_loaded', False)
            return False
        except Exception:
            return False
    
    def get_health_status(self):
        """Get overall system health"""
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def test_prediction(self, symbol: str = "BTCUSDT"):
        """Test model prediction"""
        try:
            response = self.session.post(f"{self.base_url}/api/v1/ml/predict/{symbol}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def start_conservative_trading(self, symbol: str = "BTCUSDT"):
        """Start conservative mode trading"""
        try:
            data = {"symbol": symbol, "mode": "conservative"}
            response = self.session.post(
                f"{self.base_url}/api/v1/trading/start",
                json=data
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def get_trading_status(self):
        """Get current trading status"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/trading/status")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def monitor_training(self, check_interval: int = 30, max_wait: int = 1800):
        """Monitor training progress"""
        print("🤖 ML Model Training Progress Monitor")
        print("=" * 50)
        
        print(f"⏰ Training started - checking every {check_interval} seconds")
        print("📊 Training 10,000 timesteps typically takes 5-15 minutes")
        print("💡 You can check server logs for detailed training progress")
        
        start_time = time.time()
        checks = 0
        
        while time.time() - start_time < max_wait:
            checks += 1
            elapsed = int(time.time() - start_time)
            
            print(f"\n🔍 Check #{checks} (elapsed: {elapsed//60}m {elapsed%60}s)")
            
            # Check model status
            model_ready = self.check_model_status()
            
            if model_ready:
                print("🎉 MODEL TRAINING COMPLETE!")
                return True
            else:
                print("⏳ Still training...")
                
                # Show system health
                health = self.get_health_status()
                if health:
                    services = health.get('services', {})
                    all_good = all(services.values())
                    if all_good:
                        print("   ✅ All services running normally")
                    else:
                        print("   ⚠️ Some services may have issues")
            
            # Wait before next check
            print(f"   💤 Waiting {check_interval} seconds...")
            time.sleep(check_interval)
        
        print("⏰ Training is taking longer than expected")
        print("💡 Check the server logs for any error messages")
        return False
    
    def post_training_setup(self):
        """Setup and test after training completion"""
        print("\n🚀 Post-Training Setup & Testing")
        print("=" * 40)
        
        # Test 1: Verify model is loaded
        print("1️⃣ Verifying model status...")
        model_ready = self.check_model_status()
        if model_ready:
            print("   ✅ Model successfully loaded!")
        else:
            print("   ❌ Model not loaded - something went wrong")
            return False
        
        # Test 2: Test prediction
        print("\n2️⃣ Testing AI predictions...")
        prediction = self.test_prediction()
        if prediction:
            action = prediction.get('prediction', 'UNKNOWN')
            confidence = prediction.get('confidence', 0)
            print(f"   ✅ AI Prediction: {action} (confidence: {confidence:.2f})")
            
            if confidence > 0.5:
                print("   🎯 Model seems confident in predictions!")
            else:
                print("   📊 Model needs more training for higher confidence")
        else:
            print("   ❌ Prediction test failed")
            return False
        
        # Test 3: Start paper trading
        print("\n3️⃣ Starting paper trading (conservative mode)...")
        trading_result = self.start_conservative_trading()
        if trading_result:
            print("   ✅ Paper trading started successfully!")
            print("   🛡️ Using conservative mode (safe for testing)")
        else:
            print("   ❌ Failed to start trading")
            return False
        
        # Test 4: Monitor initial trading
        print("\n4️⃣ Checking initial trading status...")
        time.sleep(2)  # Wait a moment
        
        status = self.get_trading_status()
        if status:
            is_trading = status.get('is_trading', False)
            total_trades = status.get('total_trades', 0)
            
            print(f"   Trading Active: {'✅' if is_trading else '❌'}")
            print(f"   Total Trades: {total_trades}")
            
            if is_trading:
                print("   🎉 Your AI trader is now live!")
                print("   📊 It will analyze market conditions and make decisions")
                print("   ⏰ Check back in 5-10 minutes for first trades")
            
        return True
    
    def show_next_steps(self):
        """Show user what to do next"""
        print("\n🎯 What's Next?")
        print("=" * 30)
        print("📊 Monitor trading: curl 'http://localhost:8000/api/v1/trading/status'")
        print("📈 View API docs: http://localhost:8000/docs")
        print("⏹️ Stop trading: curl -X POST 'http://localhost:8000/api/v1/trading/stop'")
        print("📋 Trading history: curl 'http://localhost:8000/api/v1/trading/history'")
        
        print("\n💡 Pro Tips:")
        print("- This is testnet mode - no real money involved")
        print("- Conservative mode has built-in safety limits")
        print("- Monitor your bot regularly, especially initially")
        print("- You can train longer models (50k+ timesteps) for better performance")
        
        print("\n🔄 Advanced Commands:")
        print("- Train longer: curl -X POST 'http://localhost:8000/api/v1/ml/train' -H 'Content-Type: application/json' -d '{\"timesteps\": 50000}'")
        print("- Switch to balanced mode: curl -X POST 'http://localhost:8000/api/v1/trading/start' -H 'Content-Type: application/json' -d '{\"mode\": \"balanced\"}'")

def main():
    monitor = TrainingMonitor()
    
    print("🤖 Crypto Trading AI - Training Monitor")
    print("=" * 50)
    
    # Monitor training
    success = monitor.monitor_training()
    
    if success:
        # Setup after training
        setup_success = monitor.post_training_setup()
        
        if setup_success:
            print("\n🎉 CONGRATULATIONS!")
            print("Your AI-powered crypto trader is now live and running!")
            
            # Show next steps
            monitor.show_next_steps()
        else:
            print("\n⚠️ Training completed but setup failed")
            print("Check the server logs for errors")
    else:
        print("\n⏰ Training is still in progress")
        print("You can:")
        print("1. Wait longer (training can take 15-30 minutes)")
        print("2. Check server logs for progress")
        print("3. Run this monitor again later: python training_monitor.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure the server is running on http://localhost:8000")