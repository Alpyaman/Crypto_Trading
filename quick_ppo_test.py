#!/usr/bin/env python3
"""
Quick PPO Model Test
Simple test to verify PPO model works before live trading
"""

import os
import numpy as np
from datetime import datetime

# PPO Model imports
try:
    from stable_baselines3 import PPO
    from crypto_trading_env import CryptoTradingEnv
    print("✅ PPO dependencies loaded successfully")
except ImportError as e:
    print(f"❌ PPO dependencies failed: {e}")
    exit(1)

def quick_ppo_test():
    """Quick test of PPO model functionality."""
    model_path = "models/ppo_crypto_trader_v3_enhanced_final.zip"
    
    print("🧠 Quick PPO Model Test")
    print("=" * 30)
    
    # Check model file
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    try:
        # Load model
        print("📂 Loading PPO model...")
        model = PPO.load(model_path)
        print("✅ Model loaded successfully")
        
        # Create environment
        print("🏗️ Creating environment...")
        trading_pairs = [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT',
            'BNBUSDT', 'SOLUSDT', 'LINKUSDT', 'UNIUSDT',
            'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT', 'SANDUSDT', 'MANAUSDT'
        ]
        
        env = CryptoTradingEnv(
            symbols=trading_pairs,
            initial_balance=10000,
            trading_fee=0.001,
            window_size=30,
            period="1y",
            interval="1d"
        )
        print("✅ Environment created successfully")
        
        # Test predictions
        print("🎯 Testing predictions...")
        obs, info = env.reset()
        print(f"📊 Observation shape: {obs.shape}")
        
        # Test 5 predictions
        for i in range(5):
            action, _states = model.predict(obs, deterministic=True)
            
            # Convert action
            if hasattr(action, 'item'):
                action = action.item()
            action = int(action)
            
            # Decode action
            symbol_idx = action // 3
            action_type = action % 3
            
            if symbol_idx < len(trading_pairs):
                symbol = trading_pairs[symbol_idx]
                action_name = ['HOLD', 'BUY', 'SELL'][action_type]
                print(f"  Prediction {i+1}: {action_name} {symbol}")
                
                # Take step
                obs, reward, done, truncated, info = env.step(action)
                print(f"    Reward: {reward:.4f}, Done: {done}")
                
                if done or truncated:
                    obs, info = env.reset()
            else:
                print(f"  Prediction {i+1}: Invalid action {action}")
                break
        
        print("✅ Quick test completed successfully!")
        print("🎉 PPO model is working and ready for live trading")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = quick_ppo_test()
    if not success:
        print("⚠️ Fix issues before using in live trading")
    else:
        print("🚀 Ready to proceed with live trading integration!")