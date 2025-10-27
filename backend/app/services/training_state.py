"""
Training State Manager
Shared state for tracking training progress across background tasks and API endpoints
"""
import threading
from datetime import datetime
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class TrainingStateManager:
    """Thread-safe singleton for managing training state"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self._state_lock = threading.Lock()
            self._training_state = {
                'is_training': False,
                'progress': 0.0,
                'current_timestep': 0,
                'total_timesteps': 0,
                'current_episode': 0,
                'loss': 0.0,
                'reward': 0.0,
                'mean_reward': 0.0,
                'episode_length': 0,
                'learning_rate': 0.0,
                'start_time': None,
                'estimated_time_remaining': '00:00:00',
                'status': 'idle',
                'error_message': None,
                'algorithm': 'PPO',
                'symbol': 'BTCUSDT'
            }
            self.initialized = True
            logger.info("Training state manager initialized")
    
    def start_training(self, total_timesteps: int, algorithm: str = 'PPO', symbol: str = 'BTCUSDT'):
        """Mark training as started"""
        with self._state_lock:
            self._training_state.update({
                'is_training': True,
                'progress': 0.0,
                'current_timestep': 0,
                'total_timesteps': total_timesteps,
                'current_episode': 0,
                'start_time': datetime.now(),
                'status': 'starting',
                'error_message': None,
                'algorithm': algorithm,
                'symbol': symbol
            })
            logger.info(f"Training started: {algorithm} on {symbol} for {total_timesteps} timesteps")
    
    def update_progress(self, 
                       current_timestep: int, 
                       current_episode: int = None,
                       loss: float = None, 
                       reward: float = None,
                       mean_reward: float = None,
                       episode_length: int = None,
                       learning_rate: float = None):
        """Update training progress"""
        with self._state_lock:
            if not self._training_state['is_training']:
                return
            
            self._training_state['current_timestep'] = current_timestep
            self._training_state['progress'] = min(100.0, (current_timestep / self._training_state['total_timesteps']) * 100)
            self._training_state['status'] = 'training'
            
            if current_episode is not None:
                self._training_state['current_episode'] = current_episode
            if loss is not None:
                self._training_state['loss'] = loss
            if reward is not None:
                self._training_state['reward'] = reward
            if mean_reward is not None:
                self._training_state['mean_reward'] = mean_reward
            if episode_length is not None:
                self._training_state['episode_length'] = episode_length
            if learning_rate is not None:
                self._training_state['learning_rate'] = learning_rate
            
            # Calculate estimated time remaining
            if self._training_state['start_time'] and current_timestep > 0:
                elapsed = datetime.now() - self._training_state['start_time']
                elapsed_seconds = elapsed.total_seconds()
                steps_per_second = current_timestep / elapsed_seconds
                remaining_steps = self._training_state['total_timesteps'] - current_timestep
                
                if steps_per_second > 0:
                    remaining_seconds = remaining_steps / steps_per_second
                    hours = int(remaining_seconds // 3600)
                    minutes = int((remaining_seconds % 3600) // 60)
                    seconds = int(remaining_seconds % 60)
                    self._training_state['estimated_time_remaining'] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def complete_training(self, success: bool = True, error_message: str = None):
        """Mark training as completed"""
        with self._state_lock:
            self._training_state.update({
                'is_training': False,
                'progress': 100.0 if success else self._training_state['progress'],
                'status': 'completed' if success else 'failed',
                'error_message': error_message,
                'estimated_time_remaining': '00:00:00'
            })
            logger.info(f"Training {'completed' if success else 'failed'}: {error_message or 'Success'}")
    
    def stop_training(self):
        """Stop training (user requested)"""
        with self._state_lock:
            if self._training_state['is_training']:
                self._training_state.update({
                    'is_training': False,
                    'status': 'stopped',
                    'estimated_time_remaining': '00:00:00'
                })
                logger.info("Training stopped by user")
    
    def get_state(self) -> Dict:
        """Get current training state (thread-safe copy)"""
        with self._state_lock:
            state = self._training_state.copy()
            
            # Format time elapsed
            if state['start_time'] and state['is_training']:
                elapsed = datetime.now() - state['start_time']
                hours = int(elapsed.total_seconds() // 3600)
                minutes = int((elapsed.total_seconds() % 3600) // 60)
                seconds = int(elapsed.total_seconds() % 60)
                state['time_elapsed'] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            else:
                state['time_elapsed'] = "00:00:00"
            
            # Remove start_time from response (not JSON serializable)
            state.pop('start_time', None)
            
            return state
    
    def reset_state(self):
        """Reset training state to initial values"""
        with self._state_lock:
            self._training_state.update({
                'is_training': False,
                'progress': 0.0,
                'current_timestep': 0,
                'total_timesteps': 0,
                'current_episode': 0,
                'loss': 0.0,
                'reward': 0.0,
                'mean_reward': 0.0,
                'episode_length': 0,
                'learning_rate': 0.0,
                'start_time': None,
                'estimated_time_remaining': '00:00:00',
                'status': 'idle',
                'error_message': None
            })


# Global instance
training_state = TrainingStateManager()