"""
Machine Learning Service
Handles model training and predictions
"""
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Optional, Tuple
import logging

from app.models.trading_env import CryptoTradingEnv

logger = logging.getLogger(__name__)


class MLService:
    """Service for ML model operations"""
    
    def __init__(self, model_path: str = "models/ppo_crypto_trader.zip"):
        self.model_path = model_path
        self.model = None
        self.env = None
        
    def train_model(
        self,
        api_key: str,
        api_secret: str,
        symbol: str = 'BTCUSDT',
        total_timesteps: int = 100000,
        learning_rate: float = 3e-4
    ) -> bool:
        """Train a new PPO model"""
        try:
            logger.info(f"Starting model training for {symbol}")
            
            # Create environment
            env = CryptoTradingEnv(
                symbol=symbol,
                initial_balance=1000.0,
                trading_fee=0.001,
                window_size=30
            )
            
            # Load data
            env.load_data(api_key, api_secret, interval='1h', limit=1000)
            
            # Wrap environment
            vec_env = DummyVecEnv([lambda: env])
            
            # Create model
            self.model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=1
            )
            
            # Train
            logger.info("Training model...")
            self.model.learn(
                total_timesteps=total_timesteps,
                progress_bar=True
            )
            
            # Save model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def load_model(self) -> bool:
        """Load a trained model"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Model not found at {self.model_path}")
                return False
            
            self.model = PPO.load(self.model_path)
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> Tuple[int, float]:
        """
        Make a prediction using the loaded model
        
        Returns:
            action (int): 0=Hold, 1=Buy, 2=Sell
            confidence (float): Confidence score 0-1
        """
        if self.model is None:
            logger.warning("Model not loaded")
            return 0, 0.0  # Default to Hold
        
        try:
            action, _states = self.model.predict(
                observation,
                deterministic=deterministic
            )
            
            # Convert to int
            if hasattr(action, 'item'):
                action = action.item()
            action = int(action)
            
            # Calculate confidence (simplified)
            # In production, you'd want to use the policy's action probabilities
            confidence = 0.7  # Placeholder
            
            return action, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0, 0.0
    
    def evaluate_model(
        self,
        api_key: str,
        api_secret: str,
        symbol: str = 'BTCUSDT',
        num_episodes: int = 10
    ) -> dict:
        """Evaluate model performance"""
        if self.model is None:
            logger.warning("Model not loaded")
            return {}
        
        try:
            # Create test environment
            env = CryptoTradingEnv(
                symbol=symbol,
                initial_balance=1000.0,
                trading_fee=0.001,
                window_size=30
            )
            env.load_data(api_key, api_secret, interval='1h', limit=500)
            
            results = []
            
            for episode in range(num_episodes):
                obs, _ = env.reset()
                episode_reward = 0
                done = False
                
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = env.step(action)
                    episode_reward += reward
                
                results.append({
                    'episode': episode + 1,
                    'reward': episode_reward,
                    'final_portfolio': info['portfolio_value'],
                    'num_trades': info['trades']
                })
            
            # Calculate statistics
            avg_reward = np.mean([r['reward'] for r in results])
            avg_portfolio = np.mean([r['final_portfolio'] for r in results])
            win_rate = len([r for r in results if r['final_portfolio'] > 1000]) / num_episodes
            
            return {
                'episodes': results,
                'avg_reward': avg_reward,
                'avg_final_portfolio': avg_portfolio,
                'win_rate': win_rate * 100
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if self.model is None:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'model_type': 'PPO',
            'policy_type': str(type(self.model.policy)),
            'model_path': self.model_path
        }
    
    def create_prediction_env(
        self,
        api_key: str,
        api_secret: str,
        symbol: str = 'BTCUSDT'
    ) -> Optional[CryptoTradingEnv]:
        """Create an environment for live predictions"""
        try:
            env = CryptoTradingEnv(
                symbol=symbol,
                initial_balance=1000.0,
                trading_fee=0.001,
                window_size=30
            )
            
            # Load recent data
            env.load_data(api_key, api_secret, interval='1h', limit=500)
            
            self.env = env
            return env
            
        except Exception as e:
            logger.error(f"Error creating prediction environment: {e}")
            return None
    
    def get_action_name(self, action: int) -> str:
        """Convert action code to human-readable name"""
        actions = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        return actions.get(action, 'UNKNOWN')