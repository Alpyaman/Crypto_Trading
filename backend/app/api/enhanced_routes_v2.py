"""
Enhanced API routes with improved error handling and user experience
"""
from fastapi import APIRouter, Depends, BackgroundTasks
from typing import Optional
import logging
from datetime import datetime
import asyncio

from app.core.error_handling import (
    APIResponse, handle_trading_exception, TradingAPIException, 
    MLServiceException, BinanceAPIException
)
from app.services.binance_service import BinanceService
from app.services.enhanced_ml_service import EnhancedMLService
from app.services.enhanced_trading_service import EnhancedTradingService
from app.services.training_state import TrainingStateManager

router = APIRouter(prefix="/api/v2", tags=["Enhanced Trading API"])
logger = logging.getLogger(__name__)

# Service dependencies
_binance_service: Optional[BinanceService] = None
_ml_service: Optional[EnhancedMLService] = None
_trading_service: Optional[EnhancedTradingService] = None
_training_state: Optional[TrainingStateManager] = None

def get_binance_service() -> BinanceService:
    """Dependency to get Binance service with error handling"""
    if _binance_service is None:
        raise BinanceAPIException("Binance service not initialized", "SERVICE_NOT_AVAILABLE")
    return _binance_service

def get_ml_service() -> EnhancedMLService:
    """Dependency to get ML service with error handling"""
    if _ml_service is None:
        raise MLServiceException("ML service not initialized", "SERVICE_NOT_AVAILABLE")
    return _ml_service

def get_trading_service() -> EnhancedTradingService:
    """Dependency to get trading service with error handling"""
    if _trading_service is None:
        raise TradingAPIException("Trading service not initialized", "SERVICE_NOT_AVAILABLE")
    return _trading_service

def set_services(
    binance_service: BinanceService,
    ml_service: EnhancedMLService,
    trading_service: EnhancedTradingService,
    training_state: TrainingStateManager
):
    """Set service instances"""
    global _binance_service, _ml_service, _trading_service, _training_state
    _binance_service = binance_service
    _ml_service = ml_service
    _trading_service = trading_service
    _training_state = training_state

@router.get("/health")
async def health_check():
    """Enhanced health check with service status"""
    try:
        services_status = {
            "binance_service": _binance_service is not None,
            "ml_service": _ml_service is not None,
            "trading_service": _trading_service is not None,
            "training_state": _training_state is not None
        }
        
        # Test Binance connection if available
        binance_connected = False
        if _binance_service:
            try:
                await _binance_service.get_account_info()
                binance_connected = True
            except Exception as e:
                logger.warning(f"Binance connection test failed: {e}")
        
        health_data = {
            "services": services_status,
            "binance_connected": binance_connected,
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": "Available via system metrics"
        }
        
        return APIResponse.success("Service is healthy", health_data)
    
    except Exception as e:
        raise handle_trading_exception(e)

@router.get("/market/status")
async def get_market_status(
    symbol: str = "BTCUSDT",
    binance_service: BinanceService = Depends(get_binance_service)
):
    """Get enhanced market status with better error handling"""
    try:
        # Get market data with timeout
        market_data = await asyncio.wait_for(
            binance_service.get_ticker_price(symbol),
            timeout=10.0
        )
        
        if not market_data:
            raise BinanceAPIException(
                f"No market data available for {symbol}",
                "NO_MARKET_DATA",
                {"symbol": symbol}
            )
        
        # Get additional market info
        try:
            klines = await binance_service.get_klines(symbol, "1h", 24)
            volume_24h = sum(float(k[5]) for k in klines) if klines else 0
        except Exception as e:
            logger.warning(f"Could not fetch volume data: {e}")
            volume_24h = 0
        
        enhanced_data = {
            "symbol": symbol,
            "price": float(market_data.get("price", 0)),
            "price_change_24h": float(market_data.get("priceChangePercent", 0)),
            "volume_24h": volume_24h,
            "last_updated": datetime.utcnow().isoformat(),
            "market_status": "OPEN"  # You can enhance this with actual market hours
        }
        
        return APIResponse.success(f"Market data for {symbol}", enhanced_data)
        
    except asyncio.TimeoutError:
        raise APIResponse.error(
            "Market data request timed out",
            "REQUEST_TIMEOUT",
            status_code=408,
            details={"symbol": symbol, "timeout": "10s"}
        )
    except Exception as e:
        raise handle_trading_exception(e)

@router.get("/account/balance")
async def get_account_balance(
    binance_service: BinanceService = Depends(get_binance_service)
):
    """Get account balance with enhanced error handling"""
    try:
        account_info = await asyncio.wait_for(
            binance_service.get_account_info(),
            timeout=10.0
        )
        
        if not account_info:
            raise BinanceAPIException(
                "Could not retrieve account information",
                "ACCOUNT_INFO_UNAVAILABLE"
            )
        
        # Parse balance information
        balances = []
        total_value_usdt = 0
        
        if "assets" in account_info:
            for asset in account_info["assets"]:
                free_balance = float(asset.get("walletBalance", 0))
                if free_balance > 0:
                    balances.append({
                        "asset": asset.get("asset"),
                        "free": free_balance,
                        "locked": float(asset.get("marginBalance", 0)) - free_balance
                    })
                    
                    # Estimate USDT value (simplified)
                    if asset.get("asset") == "USDT":
                        total_value_usdt += free_balance
        
        balance_data = {
            "balances": balances,
            "total_value_usdt": total_value_usdt,
            "last_updated": datetime.utcnow().isoformat(),
            "account_type": "futures" if "assets" in account_info else "spot"
        }
        
        return APIResponse.success("Account balance retrieved", balance_data)
        
    except asyncio.TimeoutError:
        raise APIResponse.error(
            "Account balance request timed out",
            "REQUEST_TIMEOUT",
            status_code=408
        )
    except Exception as e:
        raise handle_trading_exception(e)

@router.get("/ml/status")
async def get_ml_status(
    ml_service: EnhancedMLService = Depends(get_ml_service)
):
    """Get ML service status with detailed information"""
    try:
        # Get training progress if available
        training_progress = None
        if _training_state:
            training_progress = _training_state.get_progress()
        
        model_status = {
            "model_loaded": ml_service.model is not None,
            "model_path": ml_service.model_path,
            "last_prediction": getattr(ml_service, 'last_prediction_time', None),
            "training_progress": training_progress,
            "features_count": 86,  # Your enhanced feature count
            "model_type": "PPO with VecNormalize"
        }
        
        # Check if model is ready for trading
        ready_for_trading = (
            ml_service.model is not None and 
            (training_progress is None or training_progress.get("status") != "training")
        )
        
        model_status["ready_for_trading"] = ready_for_trading
        
        return APIResponse.success("ML service status", model_status)
        
    except Exception as e:
        raise handle_trading_exception(e)

@router.post("/ml/train")
async def start_training(
    background_tasks: BackgroundTasks,
    symbol: str = "BTCUSDT",
    timesteps: int = 100000,
    ml_service: EnhancedMLService = Depends(get_ml_service)
):
    """Start ML training with progress tracking"""
    try:
        # Check if already training
        if _training_state and _training_state.get_progress().get("status") == "training":
            raise MLServiceException(
                "Training is already in progress",
                "TRAINING_IN_PROGRESS",
                {"current_progress": _training_state.get_progress()}
            )
        
        # Validate parameters
        if timesteps < 1000:
            raise MLServiceException(
                "Minimum training timesteps is 1000",
                "INVALID_TIMESTEPS",
                {"provided": timesteps, "minimum": 1000}
            )
        
        # Start training in background
        background_tasks.add_task(
            ml_service.train_model,
            symbol=symbol,
            total_timesteps=timesteps
        )
        
        training_info = {
            "symbol": symbol,
            "timesteps": timesteps,
            "estimated_duration": f"{timesteps // 1000} minutes",
            "started_at": datetime.utcnow().isoformat()
        }
        
        return APIResponse.success("Training started successfully", training_info)
        
    except Exception as e:
        raise handle_trading_exception(e)

@router.post("/trading/start")
async def start_trading(
    symbol: str = "BTCUSDT",
    mode: str = "balanced",
    trading_service: EnhancedTradingService = Depends(get_trading_service),
    ml_service: EnhancedMLService = Depends(get_ml_service)
):
    """Start trading with enhanced validation"""
    try:
        # Validate ML model is ready
        if ml_service.model is None:
            raise MLServiceException(
                "ML model must be trained before starting trading",
                "MODEL_NOT_READY"
            )
        
        # Validate trading mode
        valid_modes = ["conservative", "balanced", "aggressive"]
        if mode not in valid_modes:
            raise TradingAPIException(
                f"Invalid trading mode: {mode}",
                "INVALID_TRADING_MODE",
                {"provided": mode, "valid_modes": valid_modes}
            )
        
        # Check account balance
        account_info = await _binance_service.get_account_info()
        if not account_info or not account_info.get("assets"):
            raise BinanceAPIException(
                "Could not verify account balance",
                "ACCOUNT_VERIFICATION_FAILED"
            )
        
        # Start trading
        success = await trading_service.start_trading(symbol, mode)
        
        if not success:
            raise TradingAPIException(
                "Failed to start trading service",
                "TRADING_START_FAILED"
            )
        
        trading_info = {
            "symbol": symbol,
            "mode": mode,
            "started_at": datetime.utcnow().isoformat(),
            "status": "active"
        }
        
        return APIResponse.success("Trading started successfully", trading_info)
        
    except Exception as e:
        raise handle_trading_exception(e)

@router.post("/trading/stop")
async def stop_trading(
    trading_service: EnhancedTradingService = Depends(get_trading_service)
):
    """Stop trading with proper cleanup"""
    try:
        success = await trading_service.stop_trading()
        
        if not success:
            raise TradingAPIException(
                "Failed to stop trading service",
                "TRADING_STOP_FAILED"
            )
        
        stop_info = {
            "stopped_at": datetime.utcnow().isoformat(),
            "status": "stopped"
        }
        
        return APIResponse.success("Trading stopped successfully", stop_info)
        
    except Exception as e:
        raise handle_trading_exception(e)

@router.get("/trading/logs")
async def get_trading_logs(
    limit: int = 50,
    trading_service: EnhancedTradingService = Depends(get_trading_service)
):
    """Get recent trading logs with pagination"""
    try:
        if limit > 500:
            raise TradingAPIException(
                "Maximum log limit is 500",
                "INVALID_LIMIT",
                {"provided": limit, "maximum": 500}
            )
        
        logs = trading_service.get_trading_logs(limit)
        
        log_data = {
            "logs": logs,
            "count": len(logs),
            "limit": limit,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        return APIResponse.success("Trading logs retrieved", log_data)
        
    except Exception as e:
        raise handle_trading_exception(e)