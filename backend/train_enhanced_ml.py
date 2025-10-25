"""
Enhanced ML Training Script
Comprehensive script to train, test, and validate the enhanced ML system for futures trading
"""
import requests
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedMLTrainer:
    """Enhanced ML training and testing orchestrator"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
        self.enhanced_base = f"{self.api_base}/enhanced"
        
    def check_server_status(self) -> bool:
        """Check if the server is running"""
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ Server is online - Version: {data.get('version', 'unknown')}")
                logger.info(f"   Environment: {data.get('environment', 'unknown')}")
                logger.info(f"   Testnet mode: {data.get('testnet_mode', 'unknown')}")
                
                services = data.get('services', {})
                logger.info(f"   Services - Binance: {services.get('binance', False)}, "
                           f"ML: {services.get('ml', False)}, Trading: {services.get('trading', False)}")
                
                model_status = data.get('model_status', {})
                logger.info(f"   Model loaded: {model_status.get('loaded', False)}")
                
                return True
            else:
                logger.error(f"❌ Server returned status code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to server. Make sure it's running on http://localhost:8000")
            return False
        except Exception as e:
            logger.error(f"❌ Error checking server status: {e}")
            return False
    
    def train_enhanced_model(self, symbol: str = "BTCUSDT", timesteps: int = 200000, algorithm: str = "PPO") -> bool:
        """Train the enhanced ML model"""
        try:
            logger.info("🚀 Starting enhanced model training...")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Algorithm: {algorithm}")
            logger.info(f"   Timesteps: {timesteps:,}")
            
            payload = {
                "symbol": symbol,
                "total_timesteps": timesteps,
                "algorithm": algorithm
            }
            
            response = requests.post(
                f"{self.enhanced_base}/train",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Training started successfully!")
                logger.info(f"   Message: {data.get('message', 'No message')}")
                
                # Show training parameters
                params = data.get('training_params', {})
                logger.info(f"   Training parameters: {params}")
                
                return True
            else:
                logger.error(f"❌ Training failed with status {response.status_code}")
                logger.error(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting training: {e}")
            return False
    
    def load_enhanced_model(self) -> bool:
        """Load the trained enhanced model"""
        try:
            logger.info("📥 Loading enhanced model...")
            
            response = requests.post(f"{self.enhanced_base}/load-model")
            
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Enhanced model loaded successfully!")
                
                model_info = data.get('model_info', {})
                if model_info:
                    logger.info(f"   Model type: {model_info.get('model_type', 'Unknown')}")
                    logger.info(f"   Features: {model_info.get('feature_count', 0)}")
                    logger.info(f"   Market regime: {model_info.get('market_regime', 'Unknown')}")
                
                return True
            else:
                logger.error(f"❌ Failed to load model: {response.status_code}")
                logger.error(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get enhanced model information"""
        try:
            response = requests.get(f"{self.enhanced_base}/model-info")
            
            if response.status_code == 200:
                data = response.json()
                model_info = data.get('model_info', {})
                
                logger.info("📊 Enhanced Model Information:")
                logger.info(f"   Loaded: {model_info.get('loaded', False)}")
                if model_info.get('loaded', False):
                    logger.info(f"   Type: {model_info.get('model_type', 'Unknown')}")
                    logger.info(f"   Features: {model_info.get('feature_count', 0)}")
                    logger.info(f"   Market regime: {model_info.get('market_regime', 'Unknown')}")
                    logger.info(f"   Path: {model_info.get('model_path', 'Unknown')}")
                
                return model_info
            else:
                logger.error(f"❌ Failed to get model info: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error getting model info: {e}")
            return {}
    
    def test_enhanced_prediction(self, symbol: str = "BTCUSDT", account_balance: float = 10000.0) -> Dict[str, Any]:
        """Test enhanced prediction capabilities"""
        try:
            logger.info(f"🔮 Testing enhanced prediction for {symbol}...")
            
            payload = {
                "symbol": symbol,
                "account_balance": account_balance
            }
            
            response = requests.post(
                f"{self.enhanced_base}/predict",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                prediction = data.get('prediction', {})
                analysis = data.get('analysis', {})
                
                logger.info("✅ Enhanced prediction completed!")
                logger.info(f"   Action: {prediction.get('action_name', 'Unknown')} (confidence: {prediction.get('confidence', 0):.3f})")
                logger.info(f"   Position size: {prediction.get('position_size', 0):.6f}")
                logger.info(f"   Current price: ${prediction.get('current_price', 0):.2f}")
                
                logger.info("📈 Market Analysis:")
                logger.info(f"   Market regime: {analysis.get('market_regime', 'Unknown')}")
                logger.info(f"   Volatility: {analysis.get('volatility', 0):.4f}")
                logger.info(f"   Risk score: {analysis.get('risk_score', 0):.3f}")
                
                support_resistance = analysis.get('support_resistance', {})
                if support_resistance:
                    logger.info(f"   Support: ${support_resistance.get('support', 0):.2f}")
                    logger.info(f"   Resistance: ${support_resistance.get('resistance', 0):.2f}")
                
                return data
            else:
                logger.error(f"❌ Prediction failed: {response.status_code}")
                logger.error(f"   Response: {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error testing prediction: {e}")
            return {}
    
    def get_market_analysis(self, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Get comprehensive market analysis"""
        try:
            logger.info(f"📊 Getting market analysis for {symbol}...")
            
            response = requests.get(f"{self.enhanced_base}/market-analysis/{symbol}")
            
            if response.status_code == 200:
                data = response.json()
                analysis = data.get('analysis', {})
                
                logger.info("✅ Market analysis completed!")
                logger.info(f"   Current price: ${analysis.get('current_price', 0):.2f}")
                logger.info(f"   Market regime: {analysis.get('market_regime', 'Unknown')}")
                
                # Technical indicators
                indicators = analysis.get('technical_indicators', {})
                logger.info("📈 Technical Indicators:")
                logger.info(f"   RSI(14): {indicators.get('rsi_14', 0):.1f}")
                logger.info(f"   MACD: {indicators.get('macd', 0):.4f}")
                logger.info(f"   BB Position: {indicators.get('bb_position', 0):.3f}")
                logger.info(f"   Volume Ratio: {indicators.get('volume_ratio', 0):.2f}")
                logger.info(f"   Volatility: {indicators.get('volatility', 0):.4f}")
                
                # Trend analysis
                trend = analysis.get('trend_analysis', {})
                logger.info("📊 Trend Analysis:")
                logger.info(f"   SMA(20): ${trend.get('sma_20', 0):.2f}")
                logger.info(f"   EMA(20): ${trend.get('ema_20', 0):.2f}")
                logger.info(f"   Price change: {trend.get('price_change', 0):.4f}")
                logger.info(f"   Trend strength: {trend.get('trend_strength', 0):.3f}")
                
                return data
            else:
                logger.error(f"❌ Market analysis failed: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error getting market analysis: {e}")
            return {}
    
    def extract_enhanced_features(self, symbol: str = "BTCUSDT", limit: int = 100) -> Dict[str, Any]:
        """Extract and analyze enhanced features"""
        try:
            logger.info(f"🔧 Extracting enhanced features for {symbol}...")
            
            response = requests.get(f"{self.enhanced_base}/features/extract/{symbol}?limit={limit}")
            
            if response.status_code == 200:
                data = response.json()
                features = data.get('features', {})
                
                logger.info("✅ Feature extraction completed!")
                logger.info(f"   Data points: {features.get('data_points', 0)}")
                logger.info(f"   Total features: {features.get('feature_count', 0)}")
                
                # Show some statistics
                stats = features.get('statistics', {})
                logger.info("📊 Price Statistics:")
                logger.info(f"   Mean price: ${stats.get('mean_price', 0):.2f}")
                logger.info(f"   Price range: ${stats.get('min_price', 0):.2f} - ${stats.get('max_price', 0):.2f}")
                logger.info(f"   Std deviation: ${stats.get('std_price', 0):.2f}")
                
                # Show feature categories
                feature_names = features.get('feature_names', [])
                if feature_names:
                    logger.info(f"📋 Sample features: {feature_names[:10]}")
                    logger.info(f"   ... and {len(feature_names) - 10} more features")
                
                return data
            else:
                logger.error(f"❌ Feature extraction failed: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error extracting features: {e}")
            return {}
    
    def start_enhanced_trading(self, symbol: str = "BTCUSDT", mode: str = "balanced", leverage: int = 10) -> bool:
        """Start enhanced trading (for testing purposes)"""
        try:
            logger.info("🚀 Starting enhanced trading...")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Mode: {mode}")
            logger.info(f"   Leverage: {leverage}x")
            
            payload = {
                "symbol": symbol,
                "mode": mode,
                "leverage": leverage
            }
            
            response = requests.post(
                f"{self.enhanced_base}/trading/start",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Enhanced trading started successfully!")
                logger.info(f"   Message: {data.get('message', 'No message')}")
                
                config = data.get('config', {})
                logger.info(f"   Configuration: {config}")
                
                return True
            else:
                logger.error(f"❌ Failed to start trading: {response.status_code}")
                logger.error(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting trading: {e}")
            return False
    
    def stop_enhanced_trading(self) -> bool:
        """Stop enhanced trading"""
        try:
            logger.info("🛑 Stopping enhanced trading...")
            
            response = requests.post(f"{self.enhanced_base}/trading/stop")
            
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Enhanced trading stopped successfully!")
                logger.info(f"   Message: {data.get('message', 'No message')}")
                return True
            else:
                logger.error(f"❌ Failed to stop trading: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error stopping trading: {e}")
            return False
    
    def get_trading_status(self) -> Dict[str, Any]:
        """Get enhanced trading status"""
        try:
            response = requests.get(f"{self.enhanced_base}/trading/status")
            
            if response.status_code == 200:
                data = response.json()
                status = data.get('trading_status', {})
                
                logger.info("📊 Enhanced Trading Status:")
                logger.info(f"   Active: {status.get('is_trading', False)}")
                
                if 'current_position' in status:
                    position = status['current_position']
                    if position.get('side'):
                        logger.info(f"   Position: {position.get('side')} {position.get('size', 0):.6f}")
                        logger.info(f"   Entry price: ${position.get('entry_price', 0):.2f}")
                        logger.info(f"   Unrealized PnL: ${position.get('unrealized_pnl', 0):.2f}")
                    else:
                        logger.info("   Position: None")
                
                if 'performance' in status:
                    perf = status['performance']
                    logger.info(f"   Total trades: {perf.get('total_trades', 0)}")
                    logger.info(f"   Win rate: {perf.get('win_rate', 0):.1f}%")
                    logger.info(f"   Total PnL: ${perf.get('total_pnl', 0):.2f}")
                    logger.info(f"   Max drawdown: {perf.get('max_drawdown', 0):.3f}")
                
                return status
            else:
                logger.error(f"❌ Failed to get trading status: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error getting trading status: {e}")
            return {}
    
    def run_comprehensive_test(self, symbol: str = "BTCUSDT"):
        """Run comprehensive test of all enhanced ML features"""
        logger.info("🧪 Starting comprehensive enhanced ML test...")
        logger.info("=" * 60)
        
        # Test sequence
        tests = [
            ("Server Status", lambda: self.check_server_status()),
            ("Model Info", lambda: bool(self.get_model_info())),
            ("Load Model", lambda: self.load_enhanced_model()),
            ("Extract Features", lambda: bool(self.extract_enhanced_features(symbol))),
            ("Market Analysis", lambda: bool(self.get_market_analysis(symbol))),
            ("Enhanced Prediction", lambda: bool(self.test_enhanced_prediction(symbol))),
        ]
        
        results = []
        for test_name, test_func in tests:
            logger.info(f"\n🔍 Running test: {test_name}")
            try:
                result = test_func()
                results.append((test_name, result))
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                results.append((test_name, False))
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("🎯 COMPREHENSIVE TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"   {test_name}: {status}")
        
        logger.info(f"\n📊 Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            logger.info("🎉 All enhanced ML features are working perfectly!")
        else:
            logger.warning(f"⚠️  {total - passed} test(s) failed. Please check the logs above.")
        
        return passed == total


def main():
    """Main training script"""
    print("🤖 ENHANCED ML TRAINING & TESTING SCRIPT")
    print("=" * 60)
    print("This script will train and test the enhanced ML system for futures trading.")
    print("Make sure the server is running on http://localhost:8000")
    print("=" * 60)
    
    trainer = EnhancedMLTrainer()
    
    # Interactive menu
    while True:
        print("\n📋 Available Actions:")
        print("1. Check server status")
        print("2. Train enhanced model")
        print("3. Load enhanced model")
        print("4. Test enhanced prediction")
        print("5. Get market analysis")
        print("6. Extract enhanced features")
        print("7. Start enhanced trading (testing)")
        print("8. Stop enhanced trading")
        print("9. Get trading status")
        print("10. Run comprehensive test")
        print("0. Exit")
        
        choice = input("\n🎯 Enter your choice (0-10): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            trainer.check_server_status()
        elif choice == "2":
            symbol = input("Enter symbol (default: BTCUSDT): ").strip() or "BTCUSDT"
            timesteps = input("Enter timesteps (default: 200000): ").strip()
            timesteps = int(timesteps) if timesteps.isdigit() else 200000
            algorithm = input("Enter algorithm (PPO/A2C, default: PPO): ").strip() or "PPO"
            trainer.train_enhanced_model(symbol, timesteps, algorithm)
        elif choice == "3":
            trainer.load_enhanced_model()
        elif choice == "4":
            symbol = input("Enter symbol (default: BTCUSDT): ").strip() or "BTCUSDT"
            trainer.test_enhanced_prediction(symbol)
        elif choice == "5":
            symbol = input("Enter symbol (default: BTCUSDT): ").strip() or "BTCUSDT"
            trainer.get_market_analysis(symbol)
        elif choice == "6":
            symbol = input("Enter symbol (default: BTCUSDT): ").strip() or "BTCUSDT"
            trainer.extract_enhanced_features(symbol)
        elif choice == "7":
            symbol = input("Enter symbol (default: BTCUSDT): ").strip() or "BTCUSDT"
            mode = input("Enter mode (conservative/balanced/aggressive, default: balanced): ").strip() or "balanced"
            leverage = input("Enter leverage (default: 10): ").strip()
            leverage = int(leverage) if leverage.isdigit() else 10
            trainer.start_enhanced_trading(symbol, mode, leverage)
        elif choice == "8":
            trainer.stop_enhanced_trading()
        elif choice == "9":
            trainer.get_trading_status()
        elif choice == "10":
            symbol = input("Enter symbol (default: BTCUSDT): ").strip() or "BTCUSDT"
            trainer.run_comprehensive_test(symbol)
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    main()