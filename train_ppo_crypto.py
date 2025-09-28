"""
Train a PPO (Proximal Policy Optimization) agent for cryptocurrency trading
using Stable-Baselines3 with the custom CryptoTradingEnv.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CallbackList,
    CheckpointCallback
)
from stable_baselines3.common.utils import set_random_seed
import torch
from crypto_trading_env import CryptoTradingEnv

class TradingCallback:
    """Custom callback to track training progress and portfolio performance."""
    
    def __init__(self, eval_env, eval_freq=5000, n_eval_episodes=5, verbose=1):
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.verbose = verbose
        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.evaluations_portfolio_values = []
        
    def _on_step(self) -> bool:
        return True
    
    def evaluate_agent(self, model, timestep):
        """Evaluate the agent's performance."""
        episode_rewards = []
        portfolio_values = []
        profit_losses = []
        
        for _ in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            portfolio_values.append(info['portfolio_value'])
            profit_losses.append(info['profit_loss_pct'])
        
        mean_reward = np.mean(episode_rewards)
        mean_portfolio_value = np.mean(portfolio_values)
        mean_profit_loss = np.mean(profit_losses)
        
        self.evaluations_timesteps.append(timestep)
        self.evaluations_results.append(mean_reward)
        self.evaluations_portfolio_values.append(mean_portfolio_value)
        
        if self.verbose > 0:
            print(f"Eval at timestep {timestep}:")
            print(f"  Mean reward: {mean_reward:.2f}")
            print(f"  Mean portfolio value: ${mean_portfolio_value:.2f}")
            print(f"  Mean profit/loss: {mean_profit_loss:.2f}%")
            print("-" * 50)
        
        return mean_reward, mean_portfolio_value, mean_profit_loss


def create_env(rank=0, seed=0):
    """Create a single environment instance."""
    def _init():
        set_random_seed(seed + rank)
        # Use the same symbols as live trading system
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT',
            'BNBUSDT', 'SOLUSDT', 'LINKUSDT', 'UNIUSDT',
            'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT', 'SANDUSDT', 'MANAUSDT'
        ]
        env = CryptoTradingEnv(
            symbols=symbols,
            initial_balance=10000,
            trading_fee=0.001,
            window_size=30,
            period="2y",
            interval="1d"
        )
        return env
    
    return _init


def train_ppo_agent(
    total_timesteps=300000,  # Increased from 200k to 300k
    learning_rate=3e-5,      # Further reduced for more stable learning
    n_steps=8192,            # Increased from 4096 for more data per update
    batch_size=256,          # Increased from 128 for better gradient estimates
    n_epochs=25,             # Increased from 20 for more learning per batch
    gamma=0.999,             # Increased from 0.995 for longer-term thinking
    gae_lambda=0.99,         # Increased from 0.98 for better advantage estimation
    clip_range=0.1,          # Reduced from 0.15 for more conservative updates
    ent_coef=0.08,           # Increased from 0.05 for more exploration
    vf_coef=0.5,
    max_grad_norm=0.5,
    use_tensorboard=True,
    model_name="ppo_crypto_trader"
):
    """Train a PPO agent for cryptocurrency trading."""
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    print("🚀 Starting PPO Training for Cryptocurrency Trading")
    print("=" * 60)
    
    # Create training environment
    print("Creating training environment...")
    n_envs = 4  # Number of parallel environments
    
    # Create environments manually to avoid make_vec_env issues
    def make_single_env(rank):
        # Use the same symbols as live trading system
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT',
            'BNBUSDT', 'SOLUSDT', 'LINKUSDT', 'UNIUSDT',
            'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT', 'SANDUSDT', 'MANAUSDT'
        ]
        env = CryptoTradingEnv(
            symbols=symbols,
            initial_balance=10000,
            trading_fee=0.001,
            window_size=30,
            period="2y",
            interval="1d"
        )
        env.reset(seed=42 + rank)
        return Monitor(env)
    
    # Create list of environment functions
    env_fns = [lambda r=i: make_single_env(r) for i in range(n_envs)]
    env = DummyVecEnv(env_fns)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT',
        'BNBUSDT', 'SOLUSDT', 'LINKUSDT', 'UNIUSDT',
        'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT', 'SANDUSDT', 'MANAUSDT'
    ]
    eval_env = CryptoTradingEnv(
        symbols=symbols,
        initial_balance=10000,
        trading_fee=0.001,
        window_size=30,
        period="2y",
        interval="1d"
    )
    eval_env = Monitor(eval_env)
    
    # Configure PPO hyperparameters (further improved settings)
    print(f"Configuring PPO with enhanced hyperparameters...")
    print(f"  Learning rate: {learning_rate} (further reduced for stability)")
    print(f"  N steps: {n_steps} (increased for more data)")
    print(f"  Batch size: {batch_size} (increased for better gradients)")
    print(f"  N epochs: {n_epochs} (increased for more learning)")
    print(f"  Gamma: {gamma} (increased for long-term planning)")
    print(f"  GAE lambda: {gae_lambda} (increased for better advantages)")
    print(f"  Clip range: {clip_range} (reduced for conservative updates)")
    print(f"  Entropy coef: {ent_coef} (increased for more exploration)")
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        tensorboard_log="logs/" if use_tensorboard else None,
        verbose=1,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    print(f"Model created on device: {model.device}")
    print(f"Policy network architecture:")
    print(model.policy)
    
    # Create callbacks (remove early stopping to train fully)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/best_{model_name}",
        log_path="logs/eval/",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"models/checkpoints_{model_name}/",
        name_prefix="checkpoint"
    )
    
    callback_list = CallbackList([
        eval_callback,
        checkpoint_callback
    ])
    
    # Train the model
    print(f"\n🎯 Starting enhanced training for {total_timesteps:,} timesteps...")
    print(f"Training with {n_envs} parallel environments")
    print("🚀 ENHANCED IMPROVEMENTS:")
    print("  • 50% longer training (300k timesteps)")
    print("  • Enhanced reward shaping with momentum signals")
    print("  • Better exploration with higher entropy coefficient")
    print("  • Larger batch sizes for more stable gradients")
    print("  • Fine-tuned hyperparameters for better convergence")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            tb_log_name=f"PPO_{model_name}",
            progress_bar=True
        )
        
        print("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed with error: {e}")
        raise
    
    # Save the final model
    final_model_path = f"models/{model_name}_final"
    model.save(final_model_path)
    print(f"💾 Final model saved to: {final_model_path}")
    
    return model, eval_env


def evaluate_trained_model(model_path, num_episodes=10):
    """Evaluate a trained model with added randomness for diversity."""
    print(f"\n📊 Evaluating trained model: {model_path}")
    
    # Load the model
    model = PPO.load(model_path)
    
    # Create environment
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT',
        'BNBUSDT', 'SOLUSDT', 'LINKUSDT', 'UNIUSDT',
        'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT', 'SANDUSDT', 'MANAUSDT'
    ]
    
    results = []
    
    for episode in range(num_episodes):
        # Add some randomness to evaluation by varying the data period slightly
        period_options = ["1y", "2y"]  # Vary between 1 and 2 year data
        selected_period = period_options[episode % len(period_options)]
        
        env = CryptoTradingEnv(
            symbols=symbols,
            initial_balance=10000,
            trading_fee=0.001,
            window_size=30,
            period=selected_period,
            interval="1d"
        )
        
        # Add randomness to starting point
        obs, _ = env.reset(seed=42 + episode * 7)  # Different seed each episode
        episode_reward = 0
        actions_taken = []
        portfolio_values = []
        done = False
        
        action_noise_std = 0.1 if episode < num_episodes // 2 else 0.05  # Reduce noise over episodes
        
        while not done:
            # Get model prediction
            action, _ = model.predict(obs, deterministic=True)
            
            # Add small amount of exploration noise for diversity
            if np.random.random() < 0.1:  # 10% chance of random action
                action = env.action_space.sample()
            elif np.random.random() < 0.05:  # 5% chance of slight action modification
                # Slightly modify the action (within same symbol, different action type)
                symbol_idx = action // 3
                new_action_type = np.random.randint(0, 3)
                action = symbol_idx * 3 + new_action_type
            
            obs, reward, done, truncated, info = env.step(action)
            
            episode_reward += reward
            actions_taken.append(action)
            portfolio_values.append(info['portfolio_value'])
        
        results.append({
            'episode': episode + 1,
            'total_reward': episode_reward,
            'final_portfolio_value': info['portfolio_value'],
            'profit_loss_pct': info['profit_loss_pct'],
            'total_trades': info['total_trades'],
            'actions': actions_taken,
            'portfolio_values': portfolio_values,
            'period_used': selected_period
        })
        
        print(f"Episode {episode + 1} ({selected_period}): "
              f"Reward={episode_reward:.2f}, "
              f"Portfolio=${info['portfolio_value']:.2f}, "
              f"P&L={info['profit_loss_pct']:.2f}%, "
              f"Trades={info['total_trades']}")
    
    return results


def plot_training_results(results, save_path="plots/evaluation_results.png"):
    """Plot evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Portfolio values over time (first episode)
    if results:
        axes[0, 0].plot(results[0]['portfolio_values'], color='blue', linewidth=2)
        axes[0, 0].axhline(y=10000, color='red', linestyle='--', alpha=0.7, label='Initial Value')
        axes[0, 0].set_title('Portfolio Value Over Time (Episode 1)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # Profit/Loss distribution
    profit_losses = [r['profit_loss_pct'] for r in results]
    axes[0, 1].hist(profit_losses, bins=max(5, len(results)//2), alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
    axes[0, 1].set_title('Profit/Loss Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Profit/Loss (%)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Final portfolio values by episode
    episodes = [r['episode'] for r in results]
    final_values = [r['final_portfolio_value'] for r in results]
    colors = ['green' if v >= 10000 else 'red' for v in final_values]
    axes[1, 0].bar(episodes, final_values, color=colors, alpha=0.7, edgecolor='black')
    axes[1, 0].axhline(y=10000, color='red', linestyle='--', alpha=0.7, label='Initial Value')
    axes[1, 0].set_title('Final Portfolio Value by Episode', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Final Portfolio Value ($)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Trading activity
    trade_counts = [r['total_trades'] for r in results]
    axes[1, 1].plot(episodes, trade_counts, marker='o', color='purple', linewidth=2, markersize=6)
    axes[1, 1].set_title('Trading Activity by Episode', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Total Trades')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    avg_profit = np.mean(profit_losses)
    win_rate = len([p for p in profit_losses if p > 0]) / len(profit_losses) * 100
    avg_trades = np.mean(trade_counts)
    best_performance = max(profit_losses)
    worst_performance = min(profit_losses)
    
    print("\n" + "="*60)
    print("📈 EVALUATION SUMMARY")
    print("="*60)
    print(f"Average Profit/Loss: {avg_profit:.2f}%")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Best Performance: {best_performance:.2f}%")
    print(f"Worst Performance: {worst_performance:.2f}%")
    print(f"Average Trades per Episode: {avg_trades:.1f}")
    print(f"Total Episodes: {len(results)}")


def buy_and_hold_baseline():
    """Calculate buy-and-hold baseline performance."""
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT',
        'BNBUSDT', 'SOLUSDT', 'LINKUSDT', 'UNIUSDT',
        'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT', 'SANDUSDT', 'MANAUSDT'
    ]
    env = CryptoTradingEnv(symbols=symbols, initial_balance=10000, period="1y")
    obs, _ = env.reset()
    
    # Buy each symbol with equal allocation at the beginning
    total_reward = 0
    for i in range(len(symbols)):
        buy_action = i * 3 + 1  # Buy action for symbol i
        obs, reward, done, truncated, info = env.step(buy_action)
        total_reward += reward
        if done:
            break
    
    # Hold for the rest
    hold_action = 0  # Hold action
    while not done:
        obs, reward, done, truncated, info = env.step(hold_action)
        total_reward += reward
    
    return {
        'strategy': 'Buy and Hold (Multi-Crypto)',
        'final_portfolio_value': info['portfolio_value'],
        'profit_loss_pct': info['profit_loss_pct'],
        'total_reward': total_reward
    }


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("🤖 PPO Multi-Cryptocurrency Trading Agent - ENHANCED VERSION")
    print("=" * 70)
    print("🎯 ENHANCED PERFORMANCE IMPROVEMENTS:")
    print("  • 50% longer training (300k timesteps)")
    print("  • Enhanced reward shaping with momentum signals")
    print("  • Better exploration and action diversity")
    print("  • Improved gradient stability with larger batches")
    print("  • Fine-tuned hyperparameters for convergence")
    print("  • Explicit buy-and-hold comparison rewards")
    print("\n🪙 Training on 15 cryptocurrencies:")
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT',
        'BNBUSDT', 'SOLUSDT', 'LINKUSDT', 'UNIUSDT',
        'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT', 'SANDUSDT', 'MANAUSDT'
    ]
    print(", ".join(symbols))
    
    # Calculate baseline
    print("📊 Calculating buy-and-hold baseline...")
    baseline = buy_and_hold_baseline()
    print(f"Buy-and-hold performance: {baseline['profit_loss_pct']:.2f}%")
    
    # Train the model
    print("\n🚀 Training enhanced PPO agent...")
    model, eval_env = train_ppo_agent(
        total_timesteps=300000,  # Increased training time further
        learning_rate=3e-5,      # More conservative learning rate
        n_steps=8192,            # More data per update
        batch_size=256,          # Better gradient estimates
        n_epochs=25,             # More learning iterations
        gamma=0.999,             # Long-term planning
        gae_lambda=0.99,         # Better advantage estimation
        clip_range=0.1,          # Conservative policy updates
        ent_coef=0.08,           # Higher exploration
        use_tensorboard=True,
        model_name="ppo_crypto_trader_v3_enhanced"
    )
    
    # Evaluate the trained model
    print("\n📊 Evaluating enhanced model...")
    results = evaluate_trained_model("models/ppo_crypto_trader_v3_enhanced_final", num_episodes=20)  # More episodes for better stats
    
    # Plot results
    plot_training_results(results)
    
    # Compare with baseline
    avg_rl_performance = np.mean([r['profit_loss_pct'] for r in results])
    print(f"\n🏆 PERFORMANCE COMPARISON:")
    print(f"Buy-and-Hold: {baseline['profit_loss_pct']:.2f}%")
    print(f"PPO Agent (avg): {avg_rl_performance:.2f}%")
    
    if avg_rl_performance > baseline['profit_loss_pct']:
        print("🎉 PPO Agent outperformed buy-and-hold strategy!")
    else:
        print("📊 Buy-and-hold was better. Consider longer training or hyperparameter tuning.")
    
    print(f"\n✅ Training and evaluation completed!")
    print(f"📁 Models saved in: models/")
    print(f"📊 Logs saved in: logs/")
    print(f"🖼️ Plots saved in: plots/")
