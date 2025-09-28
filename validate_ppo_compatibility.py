#!/usr/bin/env python3
"""
PPO Live System Compatibility Validator
Checks if live trading system is compatible with trained PPO model.
"""

import sys
import os
import logging
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_ppo_compatibility():
    """Validate PPO model compatibility with live system."""
    try:
        logger.info("🔧 Starting PPO compatibility validation...")
        
        # Import live trading system
        from live_trading_system import LiveTradingManager
        
        # Initialize system
        system = LiveTradingManager()
        
        # Check PPO model loading
        if not system.ppo_enabled:
            logger.error("❌ PPO model failed to load properly")
            return False
            
        logger.info("✅ PPO model loaded successfully")
        
        # Check trading pairs consistency
        expected_pairs = [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT',
            'BNBUSDT', 'SOLUSDT', 'LINKUSDT', 'UNIUSDT',
            'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT', 'SANDUSDT', 'MANAUSDT'
        ]
        
        if system.trading_pairs != expected_pairs:
            logger.error(f"❌ Trading pairs mismatch!")
            logger.error(f"Expected: {expected_pairs}")
            logger.error(f"Got: {system.trading_pairs}")
            return False
            
        logger.info("✅ Trading pairs match training configuration")
        
        # Check environment configuration
        env_config = {
            'symbols': len(system.trading_pairs),
            'window_size': 30,
            'action_space': 3 * len(system.trading_pairs),
            'trading_fee': 0.001
        }
        
        if system.ppo_env.action_space.n != env_config['action_space']:
            logger.error(f"❌ Action space mismatch: got {system.ppo_env.action_space.n}, expected {env_config['action_space']}")
            return False
            
        logger.info(f"✅ Action space correct: {system.ppo_env.action_space.n} actions")
        
        # Test observation generation
        logger.info("🧪 Testing observation generation...")
        obs = system._get_ppo_observation()
        
        if obs is None:
            logger.error("❌ Failed to generate PPO observation")
            return False
            
        logger.info(f"✅ Observation generated: shape {obs.shape}")
        
        # Test prediction
        logger.info("🧪 Testing PPO prediction...")
        action, confidence = system._get_simple_ppo_prediction(obs)
        
        if action is None:
            logger.error("❌ PPO prediction failed")
            return False
            
        logger.info(f"✅ PPO prediction successful: action={action}, confidence={confidence:.2f}")
        
        # Validate action decoding
        symbol_idx = action // 3
        action_type = action % 3
        
        if symbol_idx >= len(system.trading_pairs):
            logger.error(f"❌ Invalid symbol index: {symbol_idx}")
            return False
            
        symbol = system.trading_pairs[symbol_idx]
        action_name = ['HOLD', 'BUY', 'SELL'][action_type]
        
        logger.info(f"✅ Action decoded: {action_name} {symbol}")
        
        # Test confidence threshold
        min_confidence = system.ppo_config.get('min_confidence', 0.65)
        logger.info(f"📊 Confidence check: {confidence:.1%} vs {min_confidence:.1%} threshold")
        
        if confidence >= min_confidence:
            logger.info("✅ Confidence meets trading threshold")
        else:
            logger.warning("⚠️ Confidence below trading threshold")
        
        # Check configuration consistency
        logger.info("🔧 Checking configuration consistency...")
        
        # Conservative mode settings
        if system.ppo_config['conservative_mode']:
            logger.info("🛡️ Conservative mode enabled")
            if system.ppo_config['min_confidence'] != 0.75:
                logger.warning(f"⚠️ Conservative mode confidence should be 0.75, got {system.ppo_config['min_confidence']}")
        
        # Test multiple predictions for consistency
        logger.info("🔄 Testing prediction consistency...")
        predictions = []
        for i in range(5):
            action, conf = system._get_simple_ppo_prediction(obs)
            predictions.append((action, conf))
        
        # Check if all predictions use the same confidence (should be 0.8 with fixed approach)
        confidences = [p[1] for p in predictions]
        if all(c == confidences[0] for c in confidences):
            logger.info(f"✅ Prediction consistency: all {len(predictions)} predictions have confidence {confidences[0]:.1%}")
        else:
            logger.warning(f"⚠️ Prediction inconsistency: confidences vary {[f'{c:.1%}' for c in confidences]}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Compatibility validation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run the compatibility validation."""
    logger.info("=" * 60)
    logger.info("🧪 PPO LIVE SYSTEM COMPATIBILITY VALIDATOR")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    success = validate_ppo_compatibility()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("=" * 60)
    if success:
        logger.info("🎉 COMPATIBILITY VALIDATION PASSED!")
        logger.info("✅ Live trading system is fully compatible with trained PPO model")
        logger.info("🚀 System ready for live trading deployment")
    else:
        logger.error("💥 COMPATIBILITY VALIDATION FAILED!")
        logger.error("❌ Issues found that need to be resolved before live trading")
    
    logger.info(f"⏱️  Validation completed in {duration:.2f} seconds")
    logger.info("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)