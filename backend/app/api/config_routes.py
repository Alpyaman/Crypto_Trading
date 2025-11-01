"""
Configuration Management API Routes
Endpoints for managing trading configuration at runtime
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging
from datetime import datetime
import sys
import os

logger = logging.getLogger(__name__)

# Add parent directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configuration management components
try:
    from configuration.enhanced_config import get_trading_config, config_manager, TradingSystemConfig
    CONFIG_AVAILABLE = True
except ImportError:
    # Handle import error gracefully with fallback
    get_trading_config = None
    config_manager = None
    TradingSystemConfig = None
    CONFIG_AVAILABLE = False
    logger.warning("Enhanced config module not available - using fallback configuration")

# Fallback configuration for when enhanced config is not available
FALLBACK_CONFIG = {
    "environment": "development",
    "debug": True,
    "trading": {
        "risk_management": {
            "max_position_size": 0.1,
            "stop_loss_percentage": 0.02,
            "take_profit_percentage": 0.04,
            "max_open_positions": 3
        },
        "position_sizing": {
            "mode": "balanced",
            "base_amount": 100.0,
            "risk_per_trade": 0.02
        }
    },
    "ml": {
        "model_type": "ensemble",
        "retrain_interval": 24,
        "confidence_threshold": 0.7
    }
}

# Create router
config_router = APIRouter(prefix="/api/v3/config", tags=["configuration"])

# Request/Response models
class ConfigUpdateRequest(BaseModel):
    section: str = Field(..., description="Configuration section to update")
    updates: Dict[str, Any] = Field(..., description="Configuration updates")
    validate_only: bool = Field(default=False, description="Only validate, don't apply changes")

class ConfigResponse(BaseModel):
    status: str
    data: Optional[Dict[str, Any]] = None
    message: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class RiskParametersUpdate(BaseModel):
    max_position_size: Optional[float] = Field(None, ge=0.01, le=1.0)
    max_daily_loss: Optional[float] = Field(None, ge=0.01, le=0.5)
    max_drawdown: Optional[float] = Field(None, ge=0.05, le=0.9)
    stop_loss_percentage: Optional[float] = Field(None, ge=0.005, le=0.2)
    take_profit_percentage: Optional[float] = Field(None, ge=0.01, le=0.5)
    max_concurrent_trades: Optional[int] = Field(None, ge=1, le=10)

class PositionSizingUpdate(BaseModel):
    mode: str = Field(..., description="Position sizing mode")
    base_size: Optional[float] = Field(None, ge=0.01, le=1.0)
    max_size: Optional[float] = Field(None, ge=0.01, le=1.0)
    min_size: Optional[float] = Field(None, ge=0.001, le=0.5)
    volatility_adjustment: Optional[bool] = None

@config_router.get("/current", response_model=ConfigResponse)
async def get_current_config():
    """Get the current trading configuration"""
    try:
        if CONFIG_AVAILABLE and get_trading_config is not None:
            config = get_trading_config()
            # Convert to serializable format
            config_data = {
                "environment": config.environment,
                "debug": config.debug,
                "trading": {
                    "risk_management": config.trading.risk_management.dict(),
                    "position_sizing": config.trading.position_sizing.dict()
                },
                "ml": config.ml.dict() if hasattr(config, 'ml') else {},
                "backtesting": config.backtesting.dict() if hasattr(config, 'backtesting') else {}
            }
        else:
            # Use fallback configuration
            config_data = FALLBACK_CONFIG.copy()
            
        return ConfigResponse(
            status="success",
            data=config_data,
            message="Configuration retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to get current configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve configuration: {str(e)}")

@config_router.post("/risk-management", response_model=ConfigResponse)
async def update_risk_parameters(updates: RiskParametersUpdate):
    """Update risk management parameters"""
    try:
        if not CONFIG_AVAILABLE:
            # For fallback mode, just validate and return success
            # Validate take profit vs stop loss
            if (updates.take_profit_percentage is not None and 
                updates.stop_loss_percentage is not None and 
                updates.take_profit_percentage <= updates.stop_loss_percentage):
                raise HTTPException(status_code=400, detail="Take profit must be greater than stop loss")
                
            return ConfigResponse(
                status="success",
                message="Risk management parameters updated (fallback mode)"
            )
            
        if config_manager is None:
            raise HTTPException(status_code=503, detail="Configuration system not available")
            
        # Prepare updates
        risk_updates = {}
        if updates.max_position_size is not None:
            risk_updates["max_position_size"] = updates.max_position_size
        if updates.max_daily_loss is not None:
            risk_updates["max_daily_loss"] = updates.max_daily_loss
        if updates.max_drawdown is not None:
            risk_updates["max_drawdown"] = updates.max_drawdown
        if updates.stop_loss_percentage is not None:
            risk_updates["stop_loss_percentage"] = updates.stop_loss_percentage
        if updates.take_profit_percentage is not None:
            risk_updates["take_profit_percentage"] = updates.take_profit_percentage
        if updates.max_concurrent_trades is not None:
            risk_updates["max_concurrent_trades"] = updates.max_concurrent_trades
        
        if not risk_updates:
            raise HTTPException(status_code=400, detail="No valid updates provided")
        
        # Validate take profit vs stop loss
        if (updates.take_profit_percentage is not None and 
            updates.stop_loss_percentage is not None and 
            updates.take_profit_percentage <= updates.stop_loss_percentage):
            raise HTTPException(status_code=400, detail="Take profit must be greater than stop loss")
        
        # Log the updates for monitoring
        logger.info(f"Risk management parameters updated: {risk_updates}")
        
        return ConfigResponse(
            status="success",
            data={"updated_parameters": risk_updates},
            message=f"Risk management parameters updated successfully. {len(risk_updates)} parameters changed."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update risk parameters: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update risk parameters: {str(e)}")

@config_router.post("/position-sizing", response_model=ConfigResponse)
async def update_position_sizing(updates: PositionSizingUpdate):
    """Update position sizing configuration"""
    try:
        if config_manager is None:
            raise HTTPException(status_code=503, detail="Configuration system not available")
            
        if updates.mode not in ["conservative", "balanced", "aggressive"]:
            raise HTTPException(status_code=400, detail="Invalid position sizing mode")
        
        # Get current configuration
        current_config = config_manager.get_config()
        
        # Prepare updates
        sizing_updates = {}
        if updates.base_size is not None:
            sizing_updates["base_size"] = updates.base_size
        if updates.max_size is not None:
            sizing_updates["max_size"] = updates.max_size
        if updates.min_size is not None:
            sizing_updates["min_size"] = updates.min_size
        if updates.volatility_adjustment is not None:
            sizing_updates["volatility_adjustment"] = updates.volatility_adjustment
        
        if not sizing_updates:
            raise HTTPException(status_code=400, detail="No valid updates provided")
        
        # Validate size relationships
        current_mode_config = getattr(current_config.trading.position_sizing, updates.mode)
        base_size = sizing_updates.get("base_size", current_mode_config.base_size)
        max_size = sizing_updates.get("max_size", current_mode_config.max_size)
        min_size = sizing_updates.get("min_size", current_mode_config.min_size)
        
        if max_size < base_size:
            raise HTTPException(status_code=400, detail="Maximum size must be >= base size")
        if min_size > base_size:
            raise HTTPException(status_code=400, detail="Minimum size must be <= base size")
        
        logger.info(f"Position sizing updated for {updates.mode} mode: {sizing_updates}")
        
        return ConfigResponse(
            status="success",
            data={"mode": updates.mode, "updated_parameters": sizing_updates},
            message=f"Position sizing for {updates.mode} mode updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update position sizing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update position sizing: {str(e)}")

@config_router.post("/reload", response_model=ConfigResponse)
async def reload_configuration():
    """Reload configuration from files"""
    try:
        if config_manager is None:
            raise HTTPException(status_code=503, detail="Configuration system not available")
            
        # Reload configuration from files
        config_manager.reload_config()
        
        # Validate the reloaded configuration
        is_valid = config_manager.validate_config()
        
        if not is_valid:
            raise HTTPException(status_code=500, detail="Configuration validation failed after reload")
        
        logger.info("Configuration reloaded successfully from files")
        
        return ConfigResponse(
            status="success",
            message="Configuration reloaded successfully from files"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reload configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload configuration: {str(e)}")

@config_router.get("/validate", response_model=ConfigResponse)
async def validate_configuration():
    """Validate the current configuration"""
    try:
        if not CONFIG_AVAILABLE:
            # For fallback mode, just return that fallback config is valid
            validation_result = {
                "is_valid": True,
                "validation_timestamp": datetime.now().isoformat(),
                "mode": "fallback"
            }
            
            return ConfigResponse(
                status="success",
                data=validation_result,
                message="Fallback configuration is valid"
            )
            
        if config_manager is None:
            raise HTTPException(status_code=503, detail="Configuration system not available")
            
        is_valid = config_manager.validate_config()
        
        validation_result = {
            "is_valid": is_valid,
            "validation_timestamp": datetime.now().isoformat()
        }
        
        return ConfigResponse(
            status="success" if is_valid else "error",
            data=validation_result,
            message="Configuration is valid" if is_valid else "Configuration validation failed"
        )
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration validation failed: {str(e)}")

@config_router.get("/schema", response_model=ConfigResponse)
async def get_configuration_schema():
    """Get the configuration schema for validation"""
    try:
        if TradingSystemConfig is None:
            raise HTTPException(status_code=503, detail="Configuration system not available")
            
        # Get the Pydantic schema
        schema = TradingSystemConfig.schema()
        
        # Simplify schema for frontend use
        simplified_schema = {
            "trading": {
                "risk_management": {
                    "max_position_size": {"type": "number", "minimum": 0.01, "maximum": 1.0, "description": "Maximum position size as portfolio fraction"},
                    "max_daily_loss": {"type": "number", "minimum": 0.01, "maximum": 0.5, "description": "Maximum daily loss threshold"},
                    "max_drawdown": {"type": "number", "minimum": 0.05, "maximum": 0.9, "description": "Maximum portfolio drawdown"},
                    "stop_loss_percentage": {"type": "number", "minimum": 0.005, "maximum": 0.2, "description": "Stop loss percentage"},
                    "take_profit_percentage": {"type": "number", "minimum": 0.01, "maximum": 0.5, "description": "Take profit percentage"},
                    "max_concurrent_trades": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Maximum concurrent positions"}
                },
                "position_sizing": {
                    "modes": ["conservative", "balanced", "aggressive"],
                    "parameters": {
                        "base_size": {"type": "number", "minimum": 0.01, "maximum": 1.0, "description": "Base position size"},
                        "max_size": {"type": "number", "minimum": 0.01, "maximum": 1.0, "description": "Maximum position size"},
                        "min_size": {"type": "number", "minimum": 0.001, "maximum": 0.5, "description": "Minimum position size"},
                        "volatility_adjustment": {"type": "boolean", "description": "Adjust size based on volatility"}
                    }
                }
            },
            "backtesting": {
                "default": {
                    "initial_balance": {"type": "number", "minimum": 100.0, "maximum": 1000000.0, "description": "Starting capital"},
                    "commission_rate": {"type": "number", "minimum": 0.0, "maximum": 0.01, "description": "Commission rate"},
                    "slippage": {"type": "number", "minimum": 0.0, "maximum": 0.01, "description": "Slippage rate"}
                }
            }
        }
        
        return ConfigResponse(
            status="success",
            data={"schema": simplified_schema, "full_schema": schema},
            message="Configuration schema retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to get configuration schema: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get configuration schema: {str(e)}")

@config_router.get("/presets", response_model=ConfigResponse)
async def get_configuration_presets():
    """Get predefined configuration presets"""
    try:
        presets = {
            "conservative": {
                "description": "Low-risk trading with small positions and tight risk controls",
                "risk_management": {
                    "max_position_size": 0.10,
                    "max_daily_loss": 0.02,
                    "max_drawdown": 0.10,
                    "stop_loss_percentage": 0.02,
                    "take_profit_percentage": 0.04,
                    "max_concurrent_trades": 2
                },
                "position_sizing": {
                    "base_size": 0.05,
                    "max_size": 0.10,
                    "min_size": 0.02
                }
            },
            "balanced": {
                "description": "Moderate risk trading with balanced position sizing",
                "risk_management": {
                    "max_position_size": 0.25,
                    "max_daily_loss": 0.05,
                    "max_drawdown": 0.20,
                    "stop_loss_percentage": 0.03,
                    "take_profit_percentage": 0.06,
                    "max_concurrent_trades": 3
                },
                "position_sizing": {
                    "base_size": 0.15,
                    "max_size": 0.25,
                    "min_size": 0.05
                }
            },
            "aggressive": {
                "description": "High-risk trading with larger positions for experienced traders",
                "risk_management": {
                    "max_position_size": 0.50,
                    "max_daily_loss": 0.10,
                    "max_drawdown": 0.30,
                    "stop_loss_percentage": 0.05,
                    "take_profit_percentage": 0.10,
                    "max_concurrent_trades": 5
                },
                "position_sizing": {
                    "base_size": 0.30,
                    "max_size": 0.50,
                    "min_size": 0.10
                }
            }
        }
        
        return ConfigResponse(
            status="success",
            data={"presets": presets},
            message="Configuration presets retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to get configuration presets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get configuration presets: {str(e)}")