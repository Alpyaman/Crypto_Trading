"""
Enhanced API routes with comprehensive request/response validation
"""
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, status
from typing import Optional
import logging
from datetime import datetime
import asyncio
import uuid

from app.core.error_handling import handle_trading_exception
from app.models.schemas import (
    # Request models
    TradeRequest, TrainModelRequest,
    # Response models
    TradeResponse, TrainingResponse, MarketDataResponse,
    AccountBalanceResponse, MLStatusResponse, TrainingProgressResponse,
    ErrorResponse, create_error_response, create_success_response,
    # Enums
    TradingSymbol, TrainingStatus
)
from app.services.binance_service import BinanceService
from app.services.enhanced_ml_service import EnhancedMLService
from app.services.enhanced_trading_service import EnhancedTradingService
from app.services.training_state import TrainingStateManager

router = APIRouter(prefix="/api/v3", tags=["Validated Trading API"])
logger = logging.getLogger(__name__)

# Service dependencies
_binance_service: Optional[BinanceService] = None
_ml_service: Optional[EnhancedMLService] = None
_trading_service: Optional[EnhancedTradingService] = None
_training_state: Optional[TrainingStateManager] = None

def get_binance_service() -> BinanceService:
    """Dependency to get Binance service with validation"""
    if _binance_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=create_error_response(
                "Binance service not available", 
                "SERVICE_NOT_AVAILABLE"
            ).model_dump()
        )
    return _binance_service

def get_ml_service() -> EnhancedMLService:
    """Dependency to get ML service with validation"""
    if _ml_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=create_error_response(
                "ML service not available", 
                "SERVICE_NOT_AVAILABLE"
            ).model_dump()
        )
    return _ml_service

def get_trading_service() -> EnhancedTradingService:
    """Dependency to get trading service with validation"""
    if _trading_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=create_error_response(
                "Trading service not available", 
                "SERVICE_NOT_AVAILABLE"
            ).model_dump()
        )
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

@router.get(
    "/health",
    summary="Health Check",
    description="Check the health status of all services",
    responses={
        200: {"model": MLStatusResponse, "description": "Service health status"},
        503: {"model": ErrorResponse, "description": "Service unavailable"}
    }
)
async def health_check():
    """Enhanced health check with comprehensive service validation"""
    try:
        services_status = {
            "binance_service": _binance_service is not None,
            "ml_service": _ml_service is not None,
            "trading_service": _trading_service is not None,
            "training_state": _training_state is not None
        }
        
        # Test Binance connection if available
        binance_connected = False
        binance_error = None
        if _binance_service:
            try:
                await asyncio.wait_for(_binance_service.get_account_info(), timeout=5.0)
                binance_connected = True
            except Exception as e:
                binance_error = str(e)
                logger.warning(f"Binance connection test failed: {e}")
        
        health_data = {
            "services": services_status,
            "binance_connected": binance_connected,
            "binance_error": binance_error,
            "timestamp": datetime.now(datetime.timezone.utc),
            "version": "3.0.0"
        }
        
        all_healthy = all(services_status.values()) and binance_connected
        
        if not all_healthy:
            return HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=create_error_response(
                    "Some services are not healthy",
                    "PARTIAL_SERVICE_UNAVAILABLE",
                    health_data
                ).model_dump()
            )
        
        return create_success_response(
            MLStatusResponse,
            "All services are healthy",
            model_loaded=_ml_service.model is not None if _ml_service else False,
            model_type="Enhanced PPO with VecNormalize",
            features_count=86,
            training_status=TrainingStatus.NOT_STARTED,
            ready_for_trading=all_healthy
        )
    
    except Exception as e:
        raise handle_trading_exception(e)

@router.get(
    "/market/{symbol}",
    response_model=MarketDataResponse,
    summary="Get Market Data",
    description="Retrieve real-time market data for a trading symbol",
    responses={
        200: {"model": MarketDataResponse, "description": "Market data retrieved successfully"},
        400: {"model": ErrorResponse, "description": "Invalid symbol"},
        503: {"model": ErrorResponse, "description": "Market data unavailable"}
    }
)
async def get_market_data(
    symbol: TradingSymbol,
    binance_service: BinanceService = Depends(get_binance_service)
):
    """Get enhanced market data with full validation"""
    try:
        # Get market data with timeout
        market_data = await asyncio.wait_for(
            binance_service.get_ticker_price(symbol.value),
            timeout=10.0
        )
        
        if not market_data:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=create_error_response(
                    f"No market data available for {symbol.value}",
                    "NO_MARKET_DATA",
                    {"symbol": symbol.value}
                ).model_dump()
            )
        
        # Get additional market info
        try:
            klines = await binance_service.get_klines(symbol.value, "1h", 24)
            volume_24h = sum(float(k[5]) for k in klines) if klines else 0
            high_24h = max(float(k[2]) for k in klines) if klines else None
            low_24h = min(float(k[3]) for k in klines) if klines else None
        except Exception as e:
            logger.warning(f"Could not fetch additional market data: {e}")
            volume_24h = 0
            high_24h = None
            low_24h = None
        
        return MarketDataResponse(
            success=True,
            message=f"Market data for {symbol.value} retrieved successfully",
            symbol=symbol.value,
            price=float(market_data.get("price", 0)),
            price_change_24h=float(market_data.get("priceChangePercent", 0)),
            volume_24h=volume_24h,
            high_24h=high_24h,
            low_24h=low_24h,
            last_updated=datetime.utcnow()
        )
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=create_error_response(
                "Market data request timed out",
                "REQUEST_TIMEOUT",
                {"symbol": symbol.value, "timeout": "10s"}
            ).model_dump()
        )
    except Exception as e:
        raise handle_trading_exception(e)

@router.get(
    "/account/balance",
    response_model=AccountBalanceResponse,
    summary="Get Account Balance",
    description="Retrieve account balance and position information",
    responses={
        200: {"model": AccountBalanceResponse, "description": "Account balance retrieved"},
        401: {"model": ErrorResponse, "description": "Invalid credentials"},
        503: {"model": ErrorResponse, "description": "Account service unavailable"}
    }
)
async def get_account_balance(
    binance_service: BinanceService = Depends(get_binance_service)
):
    """Get account balance with comprehensive validation"""
    try:
        account_info = await asyncio.wait_for(
            binance_service.get_account_info(),
            timeout=10.0
        )
        
        if not account_info:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=create_error_response(
                    "Could not retrieve account information",
                    "ACCOUNT_INFO_UNAVAILABLE"
                ).model_dump()
            )
        
        # Parse balance information
        balances = []
        total_balance = 0
        available_balance = 0
        locked_balance = 0
        
        if "assets" in account_info:
            for asset in account_info["assets"]:
                wallet_balance = float(asset.get("walletBalance", 0))
                margin_balance = float(asset.get("marginBalance", 0))
                
                if wallet_balance > 0:
                    asset_info = {
                        "asset": asset.get("asset"),
                        "free": wallet_balance,
                        "locked": max(0, margin_balance - wallet_balance)
                    }
                    balances.append(asset_info)
                    
                    # Calculate totals (assuming USDT equivalent)
                    if asset.get("asset") == "USDT":
                        total_balance += wallet_balance
                        available_balance += wallet_balance
                        locked_balance += asset_info["locked"]
        
        return AccountBalanceResponse(
            success=True,
            message="Account balance retrieved successfully",
            total_balance=total_balance,
            available_balance=available_balance,
            locked_balance=locked_balance,
            balances=balances,
            account_type="futures" if "assets" in account_info else "spot"
        )
        
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail=create_error_response(
                "Account balance request timed out",
                "REQUEST_TIMEOUT"
            ).dict()
        )
    except Exception as e:
        raise handle_trading_exception(e)

@router.post(
    "/trading/start",
    response_model=TradeResponse,
    summary="Start Trading",
    description="Start automated trading with specified parameters",
    responses={
        200: {"model": TradeResponse, "description": "Trading started successfully"},
        400: {"model": ErrorResponse, "description": "Invalid trading parameters"},
        503: {"model": ErrorResponse, "description": "Trading service unavailable"}
    }
)
async def start_trading(
    request: TradeRequest,
    trading_service: EnhancedTradingService = Depends(get_trading_service),
    ml_service: EnhancedMLService = Depends(get_ml_service)
):
    """Start trading with comprehensive validation"""
    try:
        # Validate ML model is ready
        if ml_service.model is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=create_error_response(
                    "ML model must be trained before starting trading",
                    "MODEL_NOT_READY"
                ).model_dump()
            )
        
        # Check account balance
        account_info = await _binance_service.get_account_info()
        if not account_info or not account_info.get("assets"):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=create_error_response(
                    "Could not verify account balance",
                    "ACCOUNT_VERIFICATION_FAILED"
                ).model_dump()
            )
        
        # Generate unique trade ID
        trade_id = f"trade_{uuid.uuid4().hex[:8]}"
        
        # Start trading
        success = await trading_service.start_trading(
            request.symbol.value, 
            request.mode.value
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=create_error_response(
                    "Failed to start trading service",
                    "TRADING_START_FAILED"
                ).model_dump()
            )
        
        return TradeResponse(
            success=True,
            message="Trading started successfully",
            trade_id=trade_id,
            symbol=request.symbol.value,
            mode=request.mode.value,
            position_size=request.position_size,
            details={
                "stop_loss": request.stop_loss,
                "take_profit": request.take_profit,
                "risk_level": request.mode.value
            }
        )
        
    except Exception as e:
        raise handle_trading_exception(e)

@router.post(
    "/ml/train",
    response_model=TrainingResponse,
    summary="Start Model Training",
    description="Start ML model training with specified parameters",
    responses={
        200: {"model": TrainingResponse, "description": "Training started successfully"},
        400: {"model": ErrorResponse, "description": "Invalid training parameters"},
        409: {"model": ErrorResponse, "description": "Training already in progress"}
    }
)
async def start_training(
    request: TrainModelRequest,
    background_tasks: BackgroundTasks,
    ml_service: EnhancedMLService = Depends(get_ml_service)
):
    """Start ML training with comprehensive validation"""
    try:
        # Check if already training
        if _training_state and _training_state.get_progress().get("status") == "training":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=create_error_response(
                    "Training is already in progress",
                    "TRAINING_IN_PROGRESS",
                    {"current_progress": _training_state.get_progress()}
                ).model_dump()
            )
        
        # Generate unique training ID
        training_id = f"train_{uuid.uuid4().hex[:8]}"
        
        # Calculate estimated duration (rough estimate: 1000 steps per minute)
        estimated_minutes = request.timesteps / 1000
        estimated_duration = f"{int(estimated_minutes)} minutes"
        
        # Start training in background
        background_tasks.add_task(
            ml_service.train_model,
            symbol=request.symbol.value,
            total_timesteps=request.timesteps
        )
        
        return TrainingResponse(
            success=True,
            message="Training started successfully",
            training_id=training_id,
            symbol=request.symbol.value,
            timesteps=request.timesteps,
            estimated_duration=estimated_duration,
            progress={
                "status": "starting",
                "current_step": 0,
                "total_steps": request.timesteps
            }
        )
        
    except Exception as e:
        raise handle_trading_exception(e)

@router.get(
    "/ml/status",
    response_model=MLStatusResponse,
    summary="Get ML Status",
    description="Get current ML service and training status",
    responses={
        200: {"model": MLStatusResponse, "description": "ML status retrieved"},
        503: {"model": ErrorResponse, "description": "ML service unavailable"}
    }
)
async def get_ml_status(
    ml_service: EnhancedMLService = Depends(get_ml_service)
):
    """Get ML service status with comprehensive information"""
    try:
        # Get training progress if available
        training_status = TrainingStatus.NOT_STARTED
        training_progress = None
        
        if _training_state:
            progress = _training_state.get_progress()
            training_status = TrainingStatus(progress.get("status", "not_started"))
            training_progress = progress
        
        # Check if model is ready for trading
        ready_for_trading = (
            ml_service.model is not None and 
            training_status != TrainingStatus.TRAINING
        )
        
        return MLStatusResponse(
            success=True,
            message="ML status retrieved successfully",
            model_loaded=ml_service.model is not None,
            model_path=ml_service.model_path,
            model_type="PPO with VecNormalize",
            features_count=86,
            training_status=training_status,
            ready_for_trading=ready_for_trading,
            last_prediction=getattr(ml_service, 'last_prediction_time', None),
            performance_metrics=training_progress
        )
        
    except Exception as e:
        raise handle_trading_exception(e)

@router.get(
    "/training/progress",
    response_model=TrainingProgressResponse,
    summary="Get Training Progress",
    description="Get real-time training progress information",
    responses={
        200: {"model": TrainingProgressResponse, "description": "Training progress retrieved"},
        404: {"model": ErrorResponse, "description": "No training in progress"}
    }
)
async def get_training_progress():
    """Get training progress with detailed metrics"""
    try:
        if not _training_state:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=create_error_response(
                    "No training state available",
                    "NO_TRAINING_STATE"
                ).model_dump()
            )
        
        progress = _training_state.get_progress()
        
        # Calculate progress percentage
        progress_percentage = None
        if progress.get("current_step") and progress.get("total_steps"):
            progress_percentage = (progress["current_step"] / progress["total_steps"]) * 100
        
        return TrainingProgressResponse(
            success=True,
            message="Training progress retrieved successfully",
            status=TrainingStatus(progress.get("status", "not_started")),
            current_step=progress.get("current_step"),
            total_steps=progress.get("total_steps"),
            progress_percentage=progress_percentage,
            current_loss=progress.get("current_loss"),
            elapsed_time=progress.get("elapsed_time"),
            eta=progress.get("eta"),
            episode_count=progress.get("episode_count")
        )
        
    except Exception as e:
        raise handle_trading_exception(e)