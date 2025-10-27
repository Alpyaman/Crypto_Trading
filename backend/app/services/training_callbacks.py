"""
Training Progress Callback
Custom callback for Stable Baselines3 to report training progress to shared state
"""
from stable_baselines3.common.callbacks import BaseCallback
from app.services.training_state import training_state
import logging

logger = logging.getLogger(__name__)


class ProgressCallback(BaseCallback):
    """
    Custom callback to track training progress and update shared state
    """
    
    def __init__(self, update_frequency: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.update_frequency = update_frequency
        self.last_update = 0
        
    def _on_training_start(self) -> None:
        """Called before the first step"""
        logger.info("Training callback: Training started")
        
    def _on_step(self) -> bool:
        """
        Called after each step
        Returns True to continue training, False to stop
        """
        # Update progress every update_frequency steps
        if self.num_timesteps - self.last_update >= self.update_frequency:
            self._update_training_state()
            self.last_update = self.num_timesteps
        
        return True  # Continue training
    
    def _on_training_end(self) -> None:
        """Called at the end of training"""
        logger.info("Training callback: Training ended")
        self._update_training_state()  # Final update
    
    def _update_training_state(self):
        """Update the shared training state with current metrics"""
        try:
            # Get current metrics from the model
            current_timestep = self.num_timesteps
            
            # Try to get episode info from logger
            episode_reward = 0.0
            episode_length = 0
            mean_reward = 0.0
            loss = 0.0
            learning_rate = 0.0
            
            # Extract metrics from model's logger if available
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                # Get recent values from logger
                if hasattr(self.model.logger, 'name_to_value'):
                    values = self.model.logger.name_to_value
                    episode_reward = values.get('rollout/ep_rew_mean', 0.0)
                    episode_length = values.get('rollout/ep_len_mean', 0)
                    learning_rate = values.get('train/learning_rate', 0.0)
                    loss = values.get('train/loss', 0.0)
                    mean_reward = episode_reward  # Same as ep_rew_mean
            
            # Try to extract from locals if logger is not available
            if hasattr(self, 'locals') and self.locals:
                infos = self.locals.get('infos', [])
                if infos:
                    # Get the most recent episode info
                    for info in reversed(infos):
                        if isinstance(info, dict) and 'episode' in info:
                            episode_info = info['episode']
                            episode_reward = episode_info.get('r', episode_reward)
                            episode_length = episode_info.get('l', episode_length)
                            break
            
            # Calculate current episode estimate
            current_episode = max(1, current_timestep // max(1, int(episode_length))) if episode_length > 0 else current_timestep // 1000
            
            # Update training state
            training_state.update_progress(
                current_timestep=current_timestep,
                current_episode=current_episode,
                loss=float(loss),
                reward=float(episode_reward),
                mean_reward=float(mean_reward),
                episode_length=int(episode_length),
                learning_rate=float(learning_rate)
            )
            
            if self.verbose > 0:
                logger.info(f"Training progress: {current_timestep} steps, "
                           f"episode reward: {episode_reward:.2f}, "
                           f"loss: {loss:.4f}")
                
        except Exception as e:
            logger.error(f"Error updating training state: {e}")


class EnhancedProgressCallback(BaseCallback):
    """
    Enhanced callback with more detailed progress tracking
    """
    
    def __init__(self, 
                 update_frequency: int = 1000,
                 log_frequency: int = 10000,
                 verbose: int = 1):
        super().__init__(verbose)
        self.update_frequency = update_frequency
        self.log_frequency = log_frequency
        self.last_update = 0
        self.last_log = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_training_start(self) -> None:
        """Called before the first step"""
        logger.info("Enhanced training callback: Training started")
        
    def _on_step(self) -> bool:
        """Called after each step"""
        # Collect episode info
        if len(self.locals.get('infos', [])) > 0:
            for info in self.locals['infos']:
                if isinstance(info, dict) and 'episode' in info:
                    ep_info = info['episode']
                    self.episode_rewards.append(ep_info['r'])
                    self.episode_lengths.append(ep_info['l'])
        
        # Update progress
        if self.num_timesteps - self.last_update >= self.update_frequency:
            self._update_training_state()
            self.last_update = self.num_timesteps
        
        # Log progress
        if self.num_timesteps - self.last_log >= self.log_frequency:
            self._log_progress()
            self.last_log = self.num_timesteps
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at the end of training"""
        logger.info("Enhanced training callback: Training completed")
        self._update_training_state()
        self._log_progress()
    
    def _update_training_state(self):
        """Update training state with enhanced metrics"""
        try:
            current_timestep = self.num_timesteps
            
            # Calculate episode statistics
            mean_reward = sum(self.episode_rewards[-100:]) / len(self.episode_rewards[-100:]) if self.episode_rewards else 0.0
            mean_length = sum(self.episode_lengths[-100:]) / len(self.episode_lengths[-100:]) if self.episode_lengths else 0
            recent_reward = self.episode_rewards[-1] if self.episode_rewards else 0.0
            
            # Get training metrics from model
            loss = 0.0
            learning_rate = 0.0
            
            if hasattr(self.model, 'logger') and self.model.logger:
                if hasattr(self.model.logger, 'name_to_value'):
                    values = self.model.logger.name_to_value
                    loss = values.get('train/loss', 0.0)
                    learning_rate = values.get('train/learning_rate', 0.0)
            
            current_episode = len(self.episode_rewards)
            
            training_state.update_progress(
                current_timestep=current_timestep,
                current_episode=current_episode,
                loss=float(loss),
                reward=float(recent_reward),
                mean_reward=float(mean_reward),
                episode_length=int(mean_length),
                learning_rate=float(learning_rate)
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced progress update: {e}")
    
    def _log_progress(self):
        """Log training progress"""
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
            mean_recent = sum(recent_rewards) / len(recent_rewards)
            
            logger.info(f"Training Progress - Step: {self.num_timesteps}, "
                       f"Episodes: {len(self.episode_rewards)}, "
                       f"Mean Reward (last 10): {mean_recent:.2f}, "
                       f"Total Episodes: {len(self.episode_rewards)}")