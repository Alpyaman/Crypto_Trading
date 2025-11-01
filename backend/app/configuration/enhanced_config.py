"""
Enhanced Configuration Management with Pydantic Validation
Production-ready configuration system with type validation and environment overrides
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Dict, List, Optional, Any
from enum import Enum
import yaml
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class EnvironmentType(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging" 
    PRODUCTION = "production"

class ModelType(str, Enum):
    PPO = "PPO"
    A2C = "A2C"
    SAC = "SAC"
    TD3 = "TD3"

class PolicyType(str, Enum):
    MLP = "MlpPolicy"
    CNN = "CnnPolicy"

# API Configuration
class APIConfig(BaseModel):
    host: str = Field(default="127.0.0.1", description="API server host")
    port: int = Field(default=8000, ge=1000, le=65535, description="API server port")
    reload: bool = Field(default=True, description="Enable auto-reload in development")
    cors_origins: List[str] = Field(default_factory=list, description="CORS allowed origins")

# Binance Configuration
class BinanceConfig(BaseModel):
    testnet: bool = Field(default=True, description="Use Binance testnet")
    base_url: str = Field(default="https://testnet.binance.vision", description="Binance API base URL")
    futures_base_url: str = Field(default="https://testnet.binancefuture.com", description="Binance Futures API URL")
    rate_limit: int = Field(default=1200, ge=100, le=6000, description="Rate limit per minute")
    timeout: int = Field(default=30, ge=5, le=120, description="Request timeout in seconds")

# Risk Management Configuration
class RiskManagementConfig(BaseModel):
    max_position_size: float = Field(default=0.25, ge=0.01, le=1.0, description="Maximum position size as portfolio fraction")
    max_daily_loss: float = Field(default=0.05, ge=0.01, le=0.5, description="Maximum daily loss threshold")
    max_drawdown: float = Field(default=0.20, ge=0.05, le=0.9, description="Maximum portfolio drawdown")
    stop_loss_percentage: float = Field(default=0.03, ge=0.005, le=0.2, description="Stop loss percentage")
    take_profit_percentage: float = Field(default=0.06, ge=0.01, le=0.5, description="Take profit percentage")
    max_concurrent_trades: int = Field(default=3, ge=1, le=10, description="Maximum concurrent positions")

    @field_validator('take_profit_percentage')
    @classmethod
    def take_profit_must_exceed_stop_loss(cls, v, info):
        if info.data and 'stop_loss_percentage' in info.data and v <= info.data['stop_loss_percentage']:
            raise ValueError('Take profit must be greater than stop loss')
        return v

# Position Sizing Configuration
class PositionSizingModeConfig(BaseModel):
    base_size: float = Field(ge=0.01, le=1.0, description="Base position size")
    volatility_adjustment: bool = Field(default=True, description="Adjust size based on volatility")
    max_size: float = Field(ge=0.01, le=1.0, description="Maximum position size")
    min_size: float = Field(ge=0.001, le=0.5, description="Minimum position size")

    @field_validator('max_size')
    @classmethod
    def max_size_must_exceed_base(cls, v, info):
        if info.data and 'base_size' in info.data and v < info.data['base_size']:
            raise ValueError('Maximum size must be >= base size')
        return v

    @field_validator('min_size')
    @classmethod
    def min_size_must_be_less_than_base(cls, v, info):
        if info.data and 'base_size' in info.data and v > info.data['base_size']:
            raise ValueError('Minimum size must be <= base size')
        return v

class PositionSizingConfig(BaseModel):
    conservative: PositionSizingModeConfig
    balanced: PositionSizingModeConfig
    aggressive: PositionSizingModeConfig

# Regime Parameters Configuration
class RegimeParameterConfig(BaseModel):
    momentum_threshold: Optional[float] = Field(default=0.02, ge=0.001, le=0.1)
    trend_strength_min: Optional[float] = Field(default=0.6, ge=0.1, le=1.0)
    volatility_threshold: Optional[float] = Field(default=0.15, ge=0.01, le=1.0)
    mean_reversion_strength: Optional[float] = Field(default=0.7, ge=0.1, le=1.0)
    position_multiplier: float = Field(default=1.0, ge=0.1, le=2.0)
    stop_loss_tightening: Optional[float] = Field(default=1.0, ge=0.1, le=1.0)

class RegimeParametersConfig(BaseModel):
    trending: RegimeParameterConfig
    ranging: RegimeParameterConfig
    volatile: RegimeParameterConfig

# Trading Configuration
class TradingConfig(BaseModel):
    risk_management: RiskManagementConfig
    position_sizing: PositionSizingConfig
    regime_parameters: RegimeParametersConfig

# ML Model Configuration
class ModelConfig(BaseModel):
    type: ModelType = Field(default=ModelType.PPO, description="ML model type")
    policy: PolicyType = Field(default=PolicyType.MLP, description="Policy network type")
    learning_rate: float = Field(default=0.0003, ge=1e-6, le=1e-1, description="Learning rate")
    n_steps: int = Field(default=2048, ge=64, le=16384, description="Steps per update")
    batch_size: int = Field(default=64, ge=8, le=512, description="Batch size")
    n_epochs: int = Field(default=10, ge=1, le=50, description="Training epochs")
    gamma: float = Field(default=0.99, ge=0.9, le=0.999, description="Discount factor")
    gae_lambda: float = Field(default=0.95, ge=0.8, le=0.99, description="GAE lambda")
    clip_range: float = Field(default=0.2, ge=0.05, le=0.5, description="PPO clip range")

class TrainingConfig(BaseModel):
    total_timesteps: int = Field(default=100000, ge=1000, le=10000000, description="Total training timesteps")
    eval_freq: int = Field(default=10000, ge=1000, le=100000, description="Evaluation frequency")
    n_eval_episodes: int = Field(default=5, ge=1, le=20, description="Episodes per evaluation")
    save_freq: int = Field(default=20000, ge=1000, le=200000, description="Model save frequency")
    patience: int = Field(default=5, ge=1, le=20, description="Early stopping patience")
    min_improvement: float = Field(default=0.01, ge=0.001, le=0.1, description="Minimum improvement threshold")

class FeaturesConfig(BaseModel):
    lookback_window: int = Field(default=50, ge=10, le=200, description="Historical data window")
    technical_indicators: List[str] = Field(default_factory=list, description="Technical indicators to use")
    regime_indicators: List[str] = Field(default_factory=list, description="Regime detection indicators")

class MachineLearningConfig(BaseModel):
    model: ModelConfig
    training: TrainingConfig
    features: FeaturesConfig

# Backtesting Configuration
class BacktestingDefaultConfig(BaseModel):
    initial_balance: float = Field(default=10000.0, ge=100.0, le=1000000.0, description="Starting capital")
    commission_rate: float = Field(default=0.001, ge=0.0, le=0.01, description="Commission rate")
    slippage: float = Field(default=0.0005, ge=0.0, le=0.01, description="Slippage rate")

class BacktestingMetricsConfig(BaseModel):
    risk_free_rate: float = Field(default=0.02, ge=0.0, le=0.1, description="Risk-free rate for Sharpe calculation")
    benchmark_symbol: str = Field(default="BTCUSDT", description="Benchmark symbol")

class GradingThresholdConfig(BaseModel):
    excellent: float
    good: float
    average: float
    poor: float

class BacktestingGradingConfig(BaseModel):
    sharpe_ratio: GradingThresholdConfig
    win_rate: GradingThresholdConfig
    max_drawdown: GradingThresholdConfig

class BacktestingConfig(BaseModel):
    default: BacktestingDefaultConfig
    metrics: BacktestingMetricsConfig
    grading: BacktestingGradingConfig

# Database Configuration
class DatabaseRetentionConfig(BaseModel):
    price_data_days: int = Field(default=365, ge=30, le=3650, description="Price data retention days")
    trade_history_days: int = Field(default=730, ge=90, le=3650, description="Trade history retention days")
    backtest_results_days: int = Field(default=90, ge=7, le=365, description="Backtest results retention days")
    log_retention_days: int = Field(default=30, ge=1, le=365, description="Log retention days")

class DatabaseConfig(BaseModel):
    url: str = Field(default="sqlite:///./crypto_trading.db", description="Database URL")
    echo: bool = Field(default=False, description="Enable SQL query logging")
    pool_size: int = Field(default=10, ge=1, le=50, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Max pool overflow")
    pool_timeout: int = Field(default=30, ge=5, le=120, description="Pool timeout seconds")
    retention: DatabaseRetentionConfig

# Monitoring Configuration
class PrometheusConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable Prometheus monitoring")
    port: int = Field(default=9090, ge=1000, le=65535, description="Prometheus port")
    metrics_path: str = Field(default="/metrics", description="Metrics endpoint path")

class AlertThresholdsConfig(BaseModel):
    high_drawdown: float = Field(default=0.15, ge=0.05, le=0.5, description="High drawdown alert threshold")
    low_win_rate: float = Field(default=0.40, ge=0.1, le=0.8, description="Low win rate alert threshold")
    api_error_rate: float = Field(default=0.05, ge=0.01, le=0.2, description="API error rate threshold")
    system_cpu_usage: float = Field(default=0.85, ge=0.5, le=0.99, description="CPU usage alert threshold")
    system_memory_usage: float = Field(default=0.85, ge=0.5, le=0.99, description="Memory usage alert threshold")

class HealthCheckConfig(BaseModel):
    interval_seconds: int = Field(default=60, ge=10, le=300, description="Health check interval")
    timeout_seconds: int = Field(default=30, ge=5, le=120, description="Health check timeout")

class MonitoringConfig(BaseModel):
    prometheus: PrometheusConfig
    alerts: AlertThresholdsConfig
    health_check: HealthCheckConfig

# Security Configuration
class EncryptionConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable data encryption")
    algorithm: str = Field(default="AES-256", description="Encryption algorithm")
    key_rotation_days: int = Field(default=30, ge=1, le=365, description="Key rotation period")

class SecurityConfig(BaseModel):
    api_key_rotation_days: int = Field(default=90, ge=1, le=365, description="API key rotation period")
    session_timeout_minutes: int = Field(default=60, ge=5, le=480, description="Session timeout")
    max_login_attempts: int = Field(default=5, ge=1, le=20, description="Max login attempts")
    encryption: EncryptionConfig

# Performance Configuration
class CacheConfig(BaseModel):
    enabled: bool = Field(default=True, description="Enable caching")
    ttl_seconds: int = Field(default=300, ge=60, le=3600, description="Cache TTL")
    max_size: int = Field(default=1000, ge=100, le=10000, description="Max cache entries")

class AsyncConfig(BaseModel):
    worker_threads: int = Field(default=4, ge=1, le=16, description="Worker thread count")
    queue_size: int = Field(default=1000, ge=100, le=10000, description="Task queue size")
    timeout_seconds: int = Field(default=120, ge=30, le=600, description="Task timeout")

class ResourceLimitsConfig(BaseModel):
    max_memory_mb: int = Field(default=2048, ge=512, le=16384, description="Memory limit MB")
    max_cpu_percent: int = Field(default=80, ge=10, le=100, description="CPU limit percent")
    max_disk_gb: int = Field(default=10, ge=1, le=100, description="Disk limit GB")

class PerformanceConfig(BaseModel):
    cache: CacheConfig
    async_config: AsyncConfig = Field(alias="async")
    limits: ResourceLimitsConfig

# Main Configuration Model
class TradingSystemConfig(BaseModel):
    environment: EnvironmentType = Field(default=EnvironmentType.DEVELOPMENT, description="Environment type")
    debug: bool = Field(default=True, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    api: APIConfig
    binance: BinanceConfig
    trading: TradingConfig
    machine_learning: MachineLearningConfig
    backtesting: BacktestingConfig
    database: DatabaseConfig
    monitoring: MonitoringConfig
    security: SecurityConfig
    performance: PerformanceConfig

    @model_validator(mode='after')
    def validate_environment_consistency(self):
        """Ensure configuration is consistent with environment"""
        if self.environment == EnvironmentType.PRODUCTION and self.debug:
            raise ValueError("Debug mode should be disabled in production")
        
        return self

    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Log level must be one of: {valid_levels}')
        return v.upper()

class ConfigManager:
    """Configuration manager with environment-specific overrides"""
    
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            # Default to config directory relative to backend folder
            backend_dir = Path(__file__).parent.parent.parent
            self.config_dir = backend_dir / "config"
        else:
            self.config_dir = Path(config_dir)
        self.config: Optional[TradingSystemConfig] = None
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML files with environment overrides"""
        try:
            # Load base configuration
            base_config_path = self.config_dir / "trading_config.yaml"
            if not base_config_path.exists():
                logger.warning(f"Base config file not found: {base_config_path}")
                self.config = self._create_default_config()
                return
            
            with open(base_config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Get environment from env var or config
            environment = os.getenv('TRADING_ENVIRONMENT', config_data.get('environment', 'development'))
            
            # Apply environment-specific overrides
            env_overrides = config_data.get(environment, {})
            if env_overrides:
                config_data = self._merge_config(config_data, env_overrides)
            
            # Apply environment variable overrides
            config_data = self._apply_env_overrides(config_data)
            
            # Validate and create config object
            self.config = TradingSystemConfig(**config_data)
            
            logger.info(f"Configuration loaded successfully for environment: {environment}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Fall back to default configuration
            self.config = self._create_default_config()
    
    def _create_default_config(self) -> TradingSystemConfig:
        """Create a default configuration when loading fails"""
        default_config = {
            "environment": "development",
            "debug": True,
            "log_level": "INFO",
            "api": {
                "host": "127.0.0.1",
                "port": 8000,
                "reload": True,
                "cors_origins": ["http://localhost:3000", "http://127.0.0.1:3000"]
            },
            "binance": {
                "testnet": False,
                "base_url": "https://api.binance.com",
                "futures_base_url": "https://fapi.binance.com",
                "rate_limit": 1200,
                "timeout": 30
            },
            "trading": {
                "risk_management": {
                    "max_position_size": 0.1,
                    "max_daily_loss": 0.05,
                    "max_drawdown": 0.15,
                    "stop_loss_percentage": 0.02,
                    "take_profit_percentage": 0.04,
                    "max_concurrent_trades": 3
                },
                "position_sizing": {
                    "conservative": {
                        "base_size": 0.01,
                        "max_size": 0.05,
                        "min_size": 0.001,
                        "volatility_adjustment": True
                    },
                    "balanced": {
                        "base_size": 0.02,
                        "max_size": 0.1,
                        "min_size": 0.005,
                        "volatility_adjustment": True
                    },
                    "aggressive": {
                        "base_size": 0.05,
                        "max_size": 0.2,
                        "min_size": 0.01,
                        "volatility_adjustment": False
                    }
                },
                "regime_parameters": {
                    "trending": {
                        "momentum_threshold": 0.02,
                        "trend_strength_min": 0.6,
                        "position_multiplier": 1.2
                    },
                    "ranging": {
                        "volatility_threshold": 0.15,
                        "mean_reversion_strength": 0.7,
                        "position_multiplier": 0.8
                    },
                    "volatile": {
                        "volatility_threshold": 0.30,
                        "position_multiplier": 0.6,
                        "stop_loss_tightening": 0.5
                    }
                }
            },
            "machine_learning": {
                "model": {
                    "type": "PPO",
                    "policy": "MlpPolicy",
                    "learning_rate": 0.0003,
                    "n_steps": 2048,
                    "batch_size": 64,
                    "n_epochs": 10,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2
                },
                "training": {
                    "total_timesteps": 100000,
                    "eval_freq": 10000,
                    "n_eval_episodes": 5,
                    "save_freq": 20000,
                    "patience": 5,
                    "min_improvement": 0.01
                },
                "features": {
                    "lookback_window": 50,
                    "technical_indicators": ["RSI", "MACD", "Bollinger_Bands", "EMA_12", "EMA_26", "Volume_SMA", "ATR"],
                    "regime_indicators": ["trend_strength", "volatility_regime", "momentum_state", "volume_profile"]
                }
            },
            "backtesting": {
                "default": {
                    "initial_balance": 10000.0,
                    "commission_rate": 0.001,
                    "slippage": 0.0005
                },
                "metrics": {
                    "risk_free_rate": 0.02,
                    "benchmark_symbol": "BTCUSDT"
                },
                "grading": {
                    "sharpe_ratio": {
                        "excellent": 2.0,
                        "good": 1.5,
                        "average": 1.0,
                        "poor": 0.5
                    },
                    "win_rate": {
                        "excellent": 0.65,
                        "good": 0.55,
                        "average": 0.45,
                        "poor": 0.35
                    },
                    "max_drawdown": {
                        "excellent": 0.05,
                        "good": 0.10,
                        "average": 0.20,
                        "poor": 0.30
                    }
                }
            },
            "database": {
                "url": "sqlite:///./trading.db",
                "echo": False,
                "pool_size": 10,
                "max_overflow": 20,
                "pool_timeout": 30,
                "retention": {
                    "price_data_days": 365,
                    "trade_history_days": 730,
                    "backtest_results_days": 90,
                    "log_retention_days": 30
                }
            },
            "monitoring": {
                "prometheus": {
                    "enabled": True,
                    "port": 9090,
                    "metrics_path": "/metrics"
                },
                "alerts": {
                    "high_drawdown": 0.15,
                    "low_win_rate": 0.40,
                    "api_error_rate": 0.05,
                    "system_cpu_usage": 0.85,
                    "system_memory_usage": 0.85
                },
                "health_check": {
                    "interval_seconds": 60,
                    "timeout_seconds": 30
                }
            },
            "security": {
                "api_key_rotation_days": 90,
                "session_timeout_minutes": 60,
                "max_login_attempts": 5,
                "encryption": {
                    "enabled": True,
                    "algorithm": "AES-256",
                    "key_rotation_days": 30
                }
            },
            "performance": {
                "cache": {
                    "enabled": True,
                    "ttl_seconds": 300,
                    "max_size": 1000
                },
                "async": {
                    "worker_threads": 4,
                    "queue_size": 1000,
                    "timeout_seconds": 120
                },
                "limits": {
                    "max_memory_mb": 2048,
                    "max_cpu_percent": 80,
                    "max_disk_gb": 10
                }
            }
        }
        return TradingSystemConfig(**default_config)
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        # Environment variable mappings
        env_mappings = {
            'TRADING_DEBUG': ('debug', bool),
            'TRADING_LOG_LEVEL': ('log_level', str),
            'TRADING_API_HOST': ('api.host', str),
            'TRADING_API_PORT': ('api.port', int),
            'BINANCE_TESTNET': ('binance.testnet', bool),
            'BINANCE_RATE_LIMIT': ('binance.rate_limit', int),
            'MAX_POSITION_SIZE': ('trading.risk_management.max_position_size', float),
            'MAX_DAILY_LOSS': ('trading.risk_management.max_daily_loss', float),
            'MAX_DRAWDOWN': ('trading.risk_management.max_drawdown', float),
            'INITIAL_BALANCE': ('backtesting.default.initial_balance', float),
            'COMMISSION_RATE': ('backtesting.default.commission_rate', float),
        }
        
        for env_var, (config_path, value_type) in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    # Convert to appropriate type
                    if value_type is bool:
                        typed_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif value_type is int:
                        typed_value = int(env_value)
                    elif value_type is float:
                        typed_value = float(env_value)
                    else:
                        typed_value = env_value
                    
                    # Set nested configuration value
                    self._set_nested_value(config_data, config_path, typed_value)
                    logger.info(f"Applied environment override: {env_var} -> {config_path}")
                    
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment variable {env_var}: {e}")
        
        return config_data
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """Set a nested dictionary value using dot notation"""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def get_config(self) -> TradingSystemConfig:
        """Get the current configuration"""
        if self.config is None:
            self._load_config()
        return self.config
    
    def reload_config(self):
        """Reload configuration from files"""
        self._load_config()
    
    def get_risk_config(self) -> RiskManagementConfig:
        """Get risk management configuration"""
        return self.get_config().trading.risk_management
    
    def get_position_sizing_config(self, mode: str = "balanced") -> PositionSizingModeConfig:
        """Get position sizing configuration for specific mode"""
        position_config = self.get_config().trading.position_sizing
        
        if mode == "conservative":
            return position_config.conservative
        elif mode == "aggressive":
            return position_config.aggressive
        else:
            return position_config.balanced
    
    def get_regime_config(self, regime: str) -> RegimeParameterConfig:
        """Get regime-specific configuration"""
        regime_config = self.get_config().trading.regime_parameters
        
        if regime == "trending":
            return regime_config.trending
        elif regime == "volatile":
            return regime_config.volatile
        else:
            return regime_config.ranging
    
    def validate_config(self) -> bool:
        """Validate the current configuration"""
        try:
            self.get_config()
            # Configuration is validated by Pydantic during creation
            logger.info("Configuration validation passed")
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

# Global configuration manager instance
config_manager = ConfigManager()

# Convenience function to get configuration
def get_trading_config() -> TradingSystemConfig:
    """Get the global trading configuration"""
    return config_manager.get_config()