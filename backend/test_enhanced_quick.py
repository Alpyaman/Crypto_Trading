"""
Quick test for Enhanced ML Service
Test the enhanced ML service to ensure all features are working correctly
"""
import pandas as pd
import numpy as np
import logging
import sys
import os

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data(length=200):
    """Create test market data"""
    dates = pd.date_range(start='2024-01-01', periods=length, freq='1H')
    
    # Generate realistic OHLCV data
    base_price = 50000
    price_changes = np.random.normal(0, 0.02, length)
    cumulative_changes = np.cumsum(price_changes)
    
    close_prices = base_price * (1 + cumulative_changes)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close_prices * (1 + np.random.normal(0, 0.001, length)),
        'high': close_prices * (1 + np.abs(np.random.normal(0, 0.01, length))),
        'low': close_prices * (1 - np.abs(np.random.normal(0, 0.01, length))),
        'close': close_prices,
        'volume': np.random.normal(1000000, 200000, length)
    })
    
    # Ensure high >= close >= low and high >= open >= low
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    
    return df

def test_enhanced_ml_service():
    """Test the enhanced ML service"""
    try:
        logger.info("🧪 Testing Enhanced ML Service")
        
        # Import the enhanced ML service
        from app.services.enhanced_ml_service import EnhancedMLService
        
        # Create service instance
        ml_service = EnhancedMLService()
        logger.info("✅ Enhanced ML Service created successfully")
        
        # Create test data
        test_data = create_test_data(200)
        logger.info(f"✅ Created test data with {len(test_data)} rows")
        
        # Test feature extraction
        logger.info("🔧 Testing feature extraction...")
        enhanced_features = ml_service.extract_enhanced_features(test_data, 'BTCUSDT')
        logger.info(f"✅ Extracted {len(enhanced_features.columns)} features")
        
        # Print first few feature names
        logger.info(f"📋 Sample features: {list(enhanced_features.columns[:10])}")
        
        # Test market regime detection
        logger.info("🔍 Testing market regime detection...")
        regime = ml_service.detect_market_regime(enhanced_features)
        logger.info(f"✅ Detected market regime: {regime}")
        
        # Test position size calculation
        logger.info("💰 Testing position size calculation...")
        position_size = ml_service.calculate_position_size(
            confidence=0.8,
            account_balance=10000.0,
            current_price=50000.0,
            volatility=0.02
        )
        logger.info(f"✅ Calculated position size: {position_size:.6f}")
        
        # Test model info (without loading actual model)
        logger.info("📊 Testing model info...")
        model_info = ml_service.get_enhanced_model_info()
        logger.info(f"✅ Model info retrieved: {model_info}")
        
        logger.info("\n🎉 All Enhanced ML Service tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced ML Service test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_enhanced_environment():
    """Test the enhanced futures environment"""
    try:
        logger.info("\n🏢 Testing Enhanced Futures Environment")
        
        # Import the enhanced environment
        from app.models.enhanced_futures_env import EnhancedFuturesEnv
        
        # Create environment
        env = EnhancedFuturesEnv(
            symbol='BTCUSDT',
            initial_balance=10000.0,
            window_size=20
        )
        logger.info("✅ Enhanced Futures Environment created successfully")
        
        # Create and set test data
        test_data = create_test_data(200)
        env.set_data(test_data)
        logger.info("✅ Test data loaded into environment")
        
        # Test environment reset
        obs, info = env.reset()
        logger.info(f"✅ Environment reset - Observation shape: {obs.shape}")
        logger.info(f"📊 Initial info: Balance=${info['balance']:.2f}, Total=${info['total_value']:.2f}")
        
        # Test a few steps
        logger.info("🔄 Testing environment steps...")
        for step in range(3):
            action = np.random.randint(0, 4)  # Random action
            obs, reward, done, truncated, info = env.step(action)
            logger.info(f"   Step {step+1}: Action={action}, Reward={reward:.4f}, Balance=${info['balance']:.2f}")
        
        logger.info("✅ Enhanced Futures Environment tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Enhanced Environment test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("🚀 ENHANCED ML SYSTEM QUICK TEST")
    print("=" * 50)
    
    # Run tests
    ml_success = test_enhanced_ml_service()
    env_success = test_enhanced_environment()
    
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS:")
    print(f"Enhanced ML Service: {'✅ PASSED' if ml_success else '❌ FAILED'}")
    print(f"Enhanced Environment: {'✅ PASSED' if env_success else '❌ FAILED'}")
    
    if ml_success and env_success:
        print("\n🎉 All tests passed! Enhanced ML system is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the logs above for details.")