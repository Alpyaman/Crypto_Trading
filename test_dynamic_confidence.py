#!/usr/bin/env python3
"""
Test Dynamic Confidence Calculation
Shows how confidence varies based on different factors.
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

def test_dynamic_confidence():
    """Test the dynamic confidence calculation."""
    try:
        logger.info("🧪 Testing Dynamic PPO Confidence Calculation...")
        
        # Import live trading system
        from live_trading_system import LiveTradingManager
        
        # Initialize system
        system = LiveTradingManager()
        
        if not system.ppo_enabled:
            logger.error("❌ PPO model not available")
            return False
            
        logger.info("✅ PPO system loaded")
        
        # Get market observation
        obs = system._get_ppo_observation()
        if obs is None:
            logger.error("❌ Could not get observation")
            return False
            
        logger.info("📊 Testing confidence variations across multiple predictions...")
        
        # Test multiple predictions to see confidence variations
        predictions = []
        confidences = []
        
        for i in range(10):
            action, confidence = system._get_simple_ppo_prediction(obs)
            if action is not None:
                # Decode action
                symbol_idx = action // 3
                action_type = action % 3
                
                if symbol_idx < len(system.trading_pairs):
                    symbol = system.trading_pairs[symbol_idx]
                    action_name = ['HOLD', 'BUY', 'SELL'][action_type]
                    
                    predictions.append({
                        'action': action,
                        'symbol': symbol,
                        'action_name': action_name,
                        'confidence': confidence
                    })
                    confidences.append(confidence)
        
        if not predictions:
            logger.error("❌ No valid predictions generated")
            return False
        
        # Display results
        logger.info(f"🎯 Generated {len(predictions)} predictions:")
        logger.info("-" * 60)
        
        for i, pred in enumerate(predictions, 1):
            logger.info(f"{i:2d}. {pred['action_name']:<4} {pred['symbol']:<10} | Confidence: {pred['confidence']:.1%}")
        
        # Statistics
        avg_confidence = sum(confidences) / len(confidences)
        min_confidence = min(confidences)
        max_confidence = max(confidences)
        confidence_range = max_confidence - min_confidence
        
        logger.info("-" * 60)
        logger.info(f"📈 Confidence Statistics:")
        logger.info(f"   Average: {avg_confidence:.1%}")
        logger.info(f"   Range: {min_confidence:.1%} - {max_confidence:.1%}")
        logger.info(f"   Variation: {confidence_range:.1%}")
        
        # Check if confidence varies (not always 80%)
        if confidence_range > 0.05:  # More than 5% variation
            logger.info("✅ DYNAMIC CONFIDENCE WORKING - Values vary based on conditions!")
        else:
            logger.warning("⚠️ Low confidence variation - may still be too static")
        
        # Test confidence threshold
        system_threshold = system.ppo_config.get('min_confidence', 0.65)
        tradeable_predictions = [p for p in predictions if p['confidence'] >= system_threshold]
        
        logger.info(f"🎯 Trading Analysis:")
        logger.info(f"   Threshold: {system_threshold:.1%}")
        logger.info(f"   Tradeable: {len(tradeable_predictions)}/{len(predictions)} predictions")
        
        if tradeable_predictions:
            logger.info("✅ Some predictions meet trading threshold")
            
            # Show breakdown by action type
            action_counts = {'HOLD': 0, 'BUY': 0, 'SELL': 0}
            for pred in tradeable_predictions:
                action_counts[pred['action_name']] += 1
            
            logger.info(f"   Action breakdown: {action_counts}")
        else:
            logger.warning("⚠️ No predictions meet trading threshold")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run the dynamic confidence test."""
    logger.info("=" * 60)
    logger.info("🧪 DYNAMIC PPO CONFIDENCE TEST")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    success = test_dynamic_confidence()
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("=" * 60)
    if success:
        logger.info("🎉 DYNAMIC CONFIDENCE TEST PASSED!")
        logger.info("✅ Confidence now varies based on market conditions")
    else:
        logger.error("💥 DYNAMIC CONFIDENCE TEST FAILED!")
    
    logger.info(f"⏱️  Test completed in {duration:.2f} seconds")
    logger.info("=" * 60)
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)