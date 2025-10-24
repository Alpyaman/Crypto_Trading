"""
Configuration management for the trading application
"""
import os
from typing import Dict, Any
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class TradingConfig:
    """Trading configuration parameters"""
    
    # Risk Management
    conservative_mode: Dict[str, float] = None
    balanced_mode: Dict[str, float] = None
    aggressive_mode: Dict[str, float] = None
    
    # Model Parameters
    model_path: str = "models/ppo_crypto_trader.zip"
    training_timesteps: int = 100000
    learning_rate: float = 3e-4
    
    # Trading Parameters
    default_symbol: str = "BTCUSDT"
    trading_interval: int = 300  # 5 minutes
    data_window_size: int = 30
    
    # API Configuration
    api_rate_limit: int = 100  # requests per minute
    request_timeout: int = 30  # seconds
    
    def __post_init__(self):
        if self.conservative_mode is None:
            self.conservative_mode = {
                'min_confidence': 0.8,
                'max_position_size': 0.1,
                'stop_loss': 0.02,
                'take_profit': 0.04
            }
        
        if self.balanced_mode is None:
            self.balanced_mode = {
                'min_confidence': 0.65,
                'max_position_size': 0.25,
                'stop_loss': 0.03,
                'take_profit': 0.06
            }
        
        if self.aggressive_mode is None:
            self.aggressive_mode = {
                'min_confidence': 0.55,
                'max_position_size': 0.5,
                'stop_loss': 0.05,
                'take_profit': 0.1
            }


@dataclass
class AppConfig:
    """Main application configuration"""
    
    # Environment
    debug: bool = False
    environment: str = "development"
    log_level: str = "INFO"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    # Binance API
    binance_api_key: str = ""
    binance_api_secret: str = ""
    binance_testnet: bool = True
    
    # Security
    secret_key: str = ""
    
    # Database (for future use)
    database_url: str = "sqlite:///./crypto_trading.db"
    
    # Redis (for future use)
    redis_url: str = "redis://localhost:6379/0"
    
    # Trading Configuration
    trading: TradingConfig = None
    
    def __post_init__(self):
        if self.trading is None:
            self.trading = TradingConfig()


def load_config() -> AppConfig:
    """Load configuration from environment variables"""
    
    config = AppConfig(
        # Environment settings
        debug=os.getenv("DEBUG", "false").lower() == "true",
        environment=os.getenv("ENVIRONMENT", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        
        # Server settings
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "true").lower() == "true",
        
        # Binance API
        binance_testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true",
        binance_api_key="",  # Will be set below based on testnet mode
        binance_api_secret="",  # Will be set below based on testnet mode
        
        # Security
        secret_key=os.getenv("SECRET_KEY", "your-secret-key-here"),
        
        # Database
        database_url=os.getenv("DATABASE_URL", "sqlite:///./crypto_trading.db"),
        
        # Redis
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    
    # Set appropriate API credentials based on testnet mode
    if config.binance_testnet:
        # Use testnet credentials
        config.binance_api_key = os.getenv("BINANCE_TESTNET_API", "")
        config.binance_api_secret = os.getenv("BINANCE_TESTNET_SECRET", "")
        print("🧪 Using Binance TESTNET credentials")
    else:
        # Use live credentials
        config.binance_api_key = os.getenv("BINANCE_API_KEY", "")
        config.binance_api_secret = os.getenv("BINANCE_API_SECRET", "")
        print("⚠️ Using Binance LIVE credentials")
    
    return config


def get_trading_symbols() -> Dict[str, Dict[str, Any]]:
    """Get supported trading symbols configuration"""
    return {
        "BTCUSDT": {
            "name": "Bitcoin",
            "base_asset": "BTC",
            "quote_asset": "USDT",
            "min_quantity": 0.00001,
            "tick_size": 0.01,
            "recommended_modes": ["conservative", "balanced", "aggressive"]
        },
        "ETHUSDT": {
            "name": "Ethereum", 
            "base_asset": "ETH",
            "quote_asset": "USDT",
            "min_quantity": 0.0001,
            "tick_size": 0.01,
            "recommended_modes": ["conservative", "balanced", "aggressive"]
        },
        "ADAUSDT": {
            "name": "Cardano",
            "base_asset": "ADA", 
            "quote_asset": "USDT",
            "min_quantity": 1,
            "tick_size": 0.0001,
            "recommended_modes": ["balanced", "aggressive"]
        },
        "DOTUSDT": {
            "name": "Polkadot",
            "base_asset": "DOT",
            "quote_asset": "USDT", 
            "min_quantity": 0.1,
            "tick_size": 0.001,
            "recommended_modes": ["balanced", "aggressive"]
        }
    }


# Global configuration instance
config = load_config()