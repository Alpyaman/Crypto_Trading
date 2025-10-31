"""
Environment Configuration Loader
Handles loading environment-specific configurations with fallbacks
"""
import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EnvironmentLoader:
    """Load environment-specific configuration files"""
    
    def __init__(self, base_dir: Optional[Path] = None):
        # Default to the backend directory (two levels up from utils)
        self.base_dir = base_dir or Path(__file__).parent.parent.parent
        self.environment = os.getenv('ENVIRONMENT', 'development')
        self.loaded_files = []
        
    def load_environment(self, environment: Optional[str] = None) -> bool:
        """
        Load environment configuration with fallback chain
        
        Priority order:
        1. .env.{environment}.local (highest priority, git-ignored)
        2. .env.{environment}
        3. .env.local (git-ignored)
        4. .env (lowest priority, may be git-ignored)
        """
        env = environment or self.environment
        logger.info(f"Loading environment configuration for: {env}")
        
        # List of env files to try loading (in order of priority)
        env_files = [
            f'.env.{env}.local',
            f'.env.{env}',
            '.env.local',
            '.env'
        ]
        
        loaded_count = 0
        
        for env_file in env_files:
            file_path = self.base_dir / env_file
            if self._load_env_file(file_path):
                loaded_count += 1
        
        if loaded_count == 0:
            logger.warning("No environment files found! Using system environment variables only.")
            return False
        
        logger.info(f"Successfully loaded {loaded_count} environment file(s)")
        logger.info(f"Loaded files: {', '.join(self.loaded_files)}")
        return True
    
    def _load_env_file(self, file_path: Path) -> bool:
        """Load a single .env file"""
        if not file_path.exists():
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse KEY=VALUE pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        
                        # Only set if not already set (respects priority)
                        if key not in os.environ:
                            os.environ[key] = value
                    else:
                        logger.warning(f"Invalid line in {file_path}:{line_num}: {line}")
            
            self.loaded_files.append(file_path.name)
            logger.debug(f"Loaded environment file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return False
    
    def get_config_summary(self) -> dict:
        """Get a summary of current configuration (without sensitive data)"""
        sensitive_keys = {
            'BINANCE_API_KEY', 'BINANCE_API_SECRET', 'SECRET_KEY', 
            'TELEGRAM_BOT_TOKEN', 'DATABASE_URL', 'REDIS_URL'
        }
        
        config = {}
        for key, value in os.environ.items():
            if key.startswith(('BINANCE_', 'ENVIRONMENT', 'DEBUG', 'LOG_LEVEL', 'HOST', 'PORT')):
                if key in sensitive_keys:
                    config[key] = '***' if value else 'Not set'
                else:
                    config[key] = value
        
        return config
    
    def validate_required_config(self) -> list:
        """Validate that required configuration is present"""
        required_keys = [
            'BINANCE_API_KEY',
            'BINANCE_API_SECRET',
            'ENVIRONMENT'
        ]
        
        missing = []
        for key in required_keys:
            if not os.getenv(key):
                missing.append(key)
        
        return missing
    
    @staticmethod
    def get_bool(key: str, default: bool = False) -> bool:
        """Get boolean value from environment"""
        value = os.getenv(key, '').lower()
        return value in ('true', '1', 'yes', 'on')
    
    @staticmethod
    def get_int(key: str, default: int = 0) -> int:
        """Get integer value from environment"""
        try:
            return int(os.getenv(key, default))
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def get_float(key: str, default: float = 0.0) -> float:
        """Get float value from environment"""
        try:
            return float(os.getenv(key, default))
        except (ValueError, TypeError):
            return default


# Global instance
env_loader = EnvironmentLoader()


def load_environment(environment: Optional[str] = None) -> bool:
    """Convenience function to load environment configuration"""
    return env_loader.load_environment(environment)


def get_config_summary() -> dict:
    """Get configuration summary"""
    return env_loader.get_config_summary()


def validate_config() -> list:
    """Validate required configuration"""
    return env_loader.validate_required_config()


def load_credentials() -> dict:
    """
    Load Binance API credentials with security enhancement
    Attempts to use secure credential manager first, falls back to plain text
    """
    try:
        # Try to use secure credential manager
        from app.security.credential_manager import get_credential_manager
        credential_manager = get_credential_manager()
        
        if credential_manager.validate_credentials():
            logger.info("Using secure encrypted credentials")
            return credential_manager.get_binance_credentials()
            
    except ImportError:
        logger.warning("Secure credential manager not available")
    except Exception as e:
        logger.warning(f"Secure credential manager failed: {e}")
    
    # Fallback to plain text credentials
    logger.warning("Falling back to plain text credentials - UPGRADE TO ENCRYPTED!")
    return {
        'api_key': os.getenv('BINANCE_API_KEY'),
        'api_secret': os.getenv('BINANCE_API_SECRET'),
        'testnet': env_loader.get_bool('BINANCE_TESTNET', False)
    }


def secure_load_credentials() -> dict:
    """
    Load credentials using secure credential manager (preferred method)
    Raises exception if secure credentials are not available
    """
    from app.security.credential_manager import get_credential_manager
    credential_manager = get_credential_manager()
    return credential_manager.get_binance_credentials()


# Auto-load on import
if __name__ != "__main__":
    env_loader.load_environment()