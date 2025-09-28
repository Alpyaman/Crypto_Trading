#!/usr/bin/env python3
"""
PPO Paper Trading Test
Run PPO model in paper trading mode to test performance over multiple days
"""

import os
import numpy as np
from datetime import datetime, timedelta
import time

# PPO Model imports
try:
    from stable_baselines3 import PPO
    from crypto_trading_env import CryptoTradingEnv
    print("✅ PPO dependencies loaded successfully")
except ImportError as e:
    print(f"❌ PPO dependencies failed: {e}")
    exit(1)

class PaperTradingTest:
    """Paper trading test for PPO model."""
    
    def __init__(self):
        self.model_path = "models/ppo_crypto_trader_v3_enhanced_final.zip"
        self.trading_pairs = [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT',
            'BNBUSDT', 'SOLUSDT', 'LINKUSDT', 'UNIUSDT',
            'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT', 'SANDUSDT', 'MANAUSDT'
        ]
        self.initial_balance = 100
        self.model = None
        self.env = None
        
    def setup(self):
        """Setup model and environment."""
        print("🧠 Setting up PPO paper trading test...")
        
        # Load model
        self.model = PPO.load(self.model_path)
        print("✅ Model loaded")
        
        # Create environment
        self.env = CryptoTradingEnv(
            symbols=self.trading_pairs,
            initial_balance=self.initial_balance,
            trading_fee=0.001,
            window_size=30,
            period="1y",
            interval="1d"
        )
        print("✅ Environment created")
        
    def run_paper_trading(self, num_steps=100):
        """Run paper trading simulation."""
        print(f"📊 Running {num_steps} step paper trading simulation...")
        print("=" * 60)
        
        obs, info = self.env.reset()
        total_reward = 0
        portfolio_values = []
        trades_executed = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        for step in range(num_steps):
            # Get PPO prediction
            action, _states = self.model.predict(obs, deterministic=False)
            
            # Convert action
            if hasattr(action, 'item'):
                action = action.item()
            action = int(action)
            
            # Decode action
            symbol_idx = action // 3
            action_type = action % 3
            
            if symbol_idx < len(self.trading_pairs):
                symbol = self.trading_pairs[symbol_idx]
                action_name = ['HOLD', 'BUY', 'SELL'][action_type]
                trades_executed[action_name] += 1
                
                # Execute step
                obs, reward, done, truncated, info = self.env.step(action)
                total_reward += reward
                
                # Get portfolio value
                portfolio_value = info.get('portfolio_value', self.initial_balance)
                portfolio_values.append(portfolio_value)
                
                # Print progress every 10 steps
                if (step + 1) % 10 == 0:
                    profit_pct = ((portfolio_value - self.initial_balance) / self.initial_balance) * 100
                    print(f"Step {step+1:3d}: {action_name} {symbol} | Portfolio: ${portfolio_value:.2f} ({profit_pct:+.2f}%)")
                
                if done or truncated:
                    obs, info = self.env.reset()
                    
        # Final results
        final_value = portfolio_values[-1] if portfolio_values else self.initial_balance
        total_return = ((final_value - self.initial_balance) / self.initial_balance) * 100
        
        print("=" * 60)
        print("📈 PAPER TRADING RESULTS:")
        print(f"💰 Initial Balance: ${self.initial_balance:.2f}")
        print(f"💰 Final Portfolio: ${final_value:.2f}")
        print(f"📊 Total Return: {total_return:+.2f}%")
        print(f"🎯 Total Reward: {total_reward:.2f}")
        print(f"📈 Average Portfolio Value: ${np.mean(portfolio_values):.2f}")
        print(f"📉 Portfolio Volatility: {np.std(portfolio_values):.2f}")
        print("\n🔄 Trading Activity:")
        for action, count in trades_executed.items():
            pct = (count / num_steps) * 100
            print(f"  {action}: {count} times ({pct:.1f}%)")
        
        print("=" * 60)
        
        # Performance evaluation
        if total_return > 0:
            print("🎉 POSITIVE RETURNS - Model shows promise!")
        elif total_return > -5:
            print("⚖️ NEUTRAL PERFORMANCE - Model is stable")
        else:
            print("⚠️ NEGATIVE RETURNS - Consider retraining or parameter adjustment")
            
        return {
            'final_value': final_value,
            'total_return': total_return,
            'trades': trades_executed,
            'portfolio_values': portfolio_values
        }

def main():
    """Run paper trading test."""
    print("📊 PPO PAPER TRADING TEST")
    print("=" * 40)
    
    tester = PaperTradingTest()
    tester.setup()
    results = tester.run_paper_trading(50)  # 50 steps test
    
    print("\n✅ Paper trading test completed!")
    print("🚀 If results look good, proceed to live trading!")

if __name__ == "__main__":
    main()