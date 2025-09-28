#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PPO Environment Debug Test
Test the PPO environment to see what's causing None observations
"""

import sys
import os
import traceback
from datetime import datetime

# Add project path
sys.path.append('.')

def test_ppo_environment():
    """Test PPO environment initialization and observation generation."""
    print("🧪 Testing PPO Environment")
    print("=" * 50)
    
    try:
        # Import required modules
        print("📦 Importing modules...")
        from crypto_trading_env import CryptoTradingEnv
        from stable_baselines3 import PPO
        
        # Trading pairs (same as in live system)
        trading_pairs = [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT', 
            'BNBUSDT', 'SOLUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT', 
            'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT', 'SANDUSDT', 'MANAUSDT'
        ]
        
        print(f"✅ Modules imported successfully")
        print(f"🎯 Testing with {len(trading_pairs)} trading pairs")
        
        # Create environment
        print("\n🏗️ Creating PPO environment...")
        ppo_env = CryptoTradingEnv(
            symbols=trading_pairs,
            initial_balance=10000,
            trading_fee=0.001,
            window_size=30,
            period="6mo",
            interval="1d"
        )
        print("✅ PPO environment created")
        
        # Test environment reset
        print("\n🔄 Testing environment reset...")
        obs, info = ppo_env.reset()
        
        if obs is None:
            print("❌ Environment returned None observation")
            return False
        
        print(f"✅ Environment reset successful")
        print(f"🔍 Observation shape: {obs.shape}")
        print(f"🔍 Observation type: {type(obs)}")
        print(f"🔍 Observation range: [{obs.min():.4f}, {obs.max():.4f}]")
        
        # Check for problematic values
        if hasattr(obs, 'isnan') and obs.isnan().any():
            print("⚠️ Observation contains NaN values")
            nan_count = obs.isnan().sum()
            print(f"   NaN count: {nan_count}")
        
        if hasattr(obs, 'isinf') and obs.isinf().any():
            print("⚠️ Observation contains infinite values")
            inf_count = obs.isinf().sum()
            print(f"   Inf count: {inf_count}")
        
        # Test model loading if available
        model_path = "models/ppo_crypto_trader_v3_enhanced_final.zip"
        if os.path.exists(model_path):
            print(f"\n🤖 Testing model loading from: {model_path}")
            try:
                ppo_model = PPO.load(model_path)
                print("✅ PPO model loaded successfully")
                
                # Test prediction
                print("🎯 Testing model prediction...")
                action, action_prob = ppo_model.predict(obs, deterministic=False)
                
                print(f"✅ Prediction successful")
                print(f"🔍 Action: {action} (type: {type(action)})")
                print(f"🔍 Action probability: {action_prob} (type: {type(action_prob)})")
                
                # Test action decoding
                if action is not None:
                    try:
                        action_int = int(action)
                        symbol_idx = action_int // 3
                        action_type = action_int % 3
                        
                        if symbol_idx < len(trading_pairs):
                            symbol = trading_pairs[symbol_idx]
                            action_names = ['HOLD', 'BUY', 'SELL']
                            print(f"🎯 Decoded action: {action_names[action_type]} {symbol}")
                        else:
                            print(f"⚠️ Invalid symbol index: {symbol_idx} >= {len(trading_pairs)}")
                            
                    except Exception as e:
                        print(f"❌ Action decoding failed: {e}")
                
            except Exception as e:
                print(f"❌ Model loading/prediction failed: {e}")
                print(f"🔍 Traceback: {traceback.format_exc()}")
        else:
            print(f"⚠️ Model not found at: {model_path}")
        
        print("\n🎉 PPO environment test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ PPO environment test failed: {e}")
        print(f"🔍 Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = test_ppo_environment()
    if success:
        print("\n✅ PPO system appears to be working correctly")
        print("💡 If live trading still shows None values, check network/API issues")
    else:
        print("\n❌ PPO system has issues that need to be resolved")
