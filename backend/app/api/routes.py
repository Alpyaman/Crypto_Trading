"""
API Routes
RESTful endpoints for the trading application
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Global references to services (will be set from main.py)
binance_service = None
ml_service = None
trading_service = None


# Health check endpoints
@router.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "message": "API is running"}


@router.get("/health/binance")
async def binance_health_check():
    """Check Binance API connectivity"""
    if not binance_service:
        raise HTTPException(status_code=503, detail="Binance service not initialized")
    
    try:
        is_connected = binance_service.check_api_connectivity()
        if is_connected:
            return {"status": "healthy", "message": "Binance API is accessible"}
        else:
            raise HTTPException(status_code=503, detail="Binance API is not accessible")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Binance API health check failed: {str(e)}")


@router.get("/health/binance/status")
async def binance_service_status():
    """Get detailed Binance service status"""
    if not binance_service:
        raise HTTPException(status_code=503, detail="Binance service not initialized")
    
    try:
        status = binance_service.get_service_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service status: {str(e)}")


# Pydantic models for request/response
class StartTradingRequest(BaseModel):
    symbol: str = "BTCUSDT"
    mode: str = "balanced"  # conservative, balanced, aggressive
    leverage: int = 10  # Leverage for futures trading


class TrainModelRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timesteps: int = 100000


class OrderRequest(BaseModel):
    symbol: str
    side: str  # BUY or SELL
    quantity: float


class FuturesOrderRequest(BaseModel):
    symbol: str
    side: str  # BUY or SELL
    quantity: float
    order_type: str = "MARKET"  # MARKET or LIMIT
    price: Optional[float] = None
    leverage: Optional[int] = None


class LeverageRequest(BaseModel):
    symbol: str
    leverage: int


class MarginTypeRequest(BaseModel):
    symbol: str
    margin_type: str = "CROSSED"  # CROSSED or ISOLATED


class TradingStatusResponse(BaseModel):
    is_trading: bool
    current_position: Optional[Dict] = None
    total_trades: int
    recent_trades: List[Dict]


# Market Data Endpoints
@router.get("/market/price/{symbol}")
async def get_price(symbol: str):
    """Get current price for a symbol"""
    if not binance_service:
        raise HTTPException(status_code=503, detail="Binance service not initialized")
    
    try:
        price = binance_service.get_current_price(symbol)
        if price is None:
            raise HTTPException(
                status_code=503, 
                detail=f"Unable to fetch price for {symbol}. This could be due to network issues or API timeout."
            )
        
        return {"symbol": symbol, "price": price}
    except Exception as e:
        logger.error(f"Unexpected error in get_price endpoint: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error while fetching price for {symbol}"
        )


@router.get("/market/ticker/{symbol}")
async def get_ticker(symbol: str):
    """Get 24h ticker statistics"""
    if not binance_service:
        raise HTTPException(status_code=503, detail="Binance service not initialized")
    
    ticker = binance_service.get_24h_ticker(symbol)
    if ticker is None:
        raise HTTPException(status_code=404, detail=f"Ticker not found for {symbol}")
    
    return ticker


@router.get("/market/klines/{symbol}")
async def get_klines(symbol: str, interval: str = "1h", limit: int = 100):
    """Get historical candlestick data"""
    if not binance_service:
        raise HTTPException(status_code=503, detail="Binance service not initialized")
    
    klines = binance_service.get_historical_klines(symbol, interval, limit)
    if not klines:
        raise HTTPException(status_code=404, detail=f"Klines not found for {symbol}")
    
    return {
        "symbol": symbol,
        "interval": interval,
        "data": klines
    }


# Account Endpoints
@router.get("/account/balance")
async def get_balance():
    """Get account balances"""
    if not binance_service:
        raise HTTPException(status_code=503, detail="Binance service not initialized")
    
    try:
        balance = binance_service.get_account_balance()
        return balance
    except Exception as e:
        logger.error(f"Unexpected error in get_balance endpoint: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error while fetching account balance"
        )


@router.post("/account/order")
async def place_order(order_request: OrderRequest):
    """Place a manual spot order (deprecated - use futures endpoints)"""
    if not binance_service:
        raise HTTPException(status_code=503, detail="Binance service not initialized")
    
    try:
        result = binance_service.place_futures_order(
            symbol=order_request.symbol,
            side=order_request.side,
            quantity=order_request.quantity,
            order_type='MARKET'
        )
        
        if result is None:
            raise HTTPException(status_code=400, detail="Failed to place order")
        
        return result
        
    except Exception as e:
        logger.error(f"Error placing order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Futures Trading Endpoints
@router.post("/futures/order")
async def place_futures_order(order_request: FuturesOrderRequest):
    """Place a futures order with leverage support"""
    if not binance_service:
        raise HTTPException(status_code=503, detail="Binance service not initialized")
    
    try:
        # Validate the order first
        is_valid, message = binance_service.validate_futures_order(
            symbol=order_request.symbol,
            side=order_request.side,
            quantity=order_request.quantity,
            order_type=order_request.order_type
        )
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=f"Order validation failed: {message}")
        
        result = binance_service.place_futures_order(
            symbol=order_request.symbol,
            side=order_request.side,
            quantity=order_request.quantity,
            order_type=order_request.order_type,
            price=order_request.price,
            leverage=order_request.leverage
        )
        
        if result is None:
            raise HTTPException(status_code=400, detail="Failed to place futures order")
        
        return result
        
    except Exception as e:
        logger.error(f"Error placing futures order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/futures/leverage")
async def set_leverage(leverage_request: LeverageRequest):
    """Set leverage for a futures symbol"""
    if not binance_service:
        raise HTTPException(status_code=503, detail="Binance service not initialized")
    
    try:
        success = binance_service.set_leverage(
            symbol=leverage_request.symbol,
            leverage=leverage_request.leverage
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to set leverage")
        
        return {
            "symbol": leverage_request.symbol,
            "leverage": leverage_request.leverage,
            "message": "Leverage set successfully"
        }
        
    except Exception as e:
        logger.error(f"Error setting leverage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/futures/margin-type")
async def set_margin_type(margin_request: MarginTypeRequest):
    """Set margin type for a futures symbol"""
    if not binance_service:
        raise HTTPException(status_code=503, detail="Binance service not initialized")
    
    try:
        success = binance_service.set_margin_type(
            symbol=margin_request.symbol,
            margin_type=margin_request.margin_type
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to set margin type")
        
        return {
            "symbol": margin_request.symbol,
            "margin_type": margin_request.margin_type,
            "message": "Margin type set successfully"
        }
        
    except Exception as e:
        logger.error(f"Error setting margin type: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/futures/positions")
async def get_positions(symbol: Optional[str] = None):
    """Get current futures positions"""
    if not binance_service:
        raise HTTPException(status_code=503, detail="Binance service not initialized")
    
    try:
        positions = binance_service.get_position_info(symbol)
        return {
            "positions": positions,
            "total_positions": len(positions)
        }
        
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/futures/close-position/{symbol}")
async def close_position(symbol: str):
    """Close all positions for a symbol"""
    if not binance_service:
        raise HTTPException(status_code=503, detail="Binance service not initialized")
    
    try:
        result = binance_service.close_position(symbol)
        
        if result is None:
            return {"message": f"No open positions for {symbol}"}
        
        return {
            "message": f"Position closed for {symbol}",
            "order": result
        }
        
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/futures/portfolio")
async def get_futures_portfolio():
    """Get futures portfolio statistics"""
    if not binance_service:
        raise HTTPException(status_code=503, detail="Binance service not initialized")
    
    try:
        portfolio = binance_service.get_portfolio_value()
        return portfolio
        
    except Exception as e:
        logger.error(f"Error getting futures portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ML Model Endpoints
@router.post("/ml/train")
async def train_model(
    train_request: TrainModelRequest,
    background_tasks: BackgroundTasks
):
    """Train ML model in background"""
    if not ml_service:
        raise HTTPException(status_code=503, detail="ML service not initialized")
    
    # Add training task to background
    background_tasks.add_task(
        _train_model_background,
        train_request.symbol,
        train_request.timesteps
    )
    
    return {
        "message": f"Model training started for {train_request.symbol}",
        "timesteps": train_request.timesteps
    }


@router.get("/ml/status")
async def get_model_status():
    """Get ML model status"""
    if not ml_service:
        raise HTTPException(status_code=503, detail="ML service not initialized")
    
    return {
        "model_loaded": ml_service.model is not None,
        "model_path": ml_service.model_path
    }


@router.post("/ml/predict/{symbol}")
async def get_prediction(symbol: str):
    """Get trading prediction for a symbol"""
    if not ml_service or not ml_service.model:
        raise HTTPException(status_code=503, detail="ML model not available")
    
    if not binance_service:
        raise HTTPException(status_code=503, detail="Binance service not initialized")
    
    try:
        # Get recent market data
        klines = binance_service.get_historical_klines(symbol, '1h', 100)
        if not klines:
            raise HTTPException(status_code=404, detail="Market data not available")
        
        # Prepare observation (simplified)
        observation = _prepare_observation_simple(klines)
        
        # Get prediction
        action, confidence = ml_service.predict(observation)
        
        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
        
        return {
            "symbol": symbol,
            "prediction": action_names.get(action, "UNKNOWN"),
            "confidence": float(confidence),
            "action_code": int(action)
        }
        
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        # Return a simple fallback response for debugging
        return {
            "symbol": symbol,
            "prediction": "BUY",
            "confidence": 0.5,
            "action_code": 1,
            "error": str(e)
        }


# Trading Endpoints
@router.post("/trading/start")
async def start_trading(trading_request: StartTradingRequest):
    """Start automated futures trading"""
    if not trading_service:
        raise HTTPException(status_code=503, detail="Trading service not initialized")
    
    success = trading_service.start_trading(
        symbol=trading_request.symbol,
        mode=trading_request.mode,
        leverage=trading_request.leverage
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to start futures trading")
    
    return {
        "message": f"Futures trading started for {trading_request.symbol}",
        "mode": trading_request.mode,
        "leverage": f"{trading_request.leverage}x",
        "trading_type": "futures"
    }


@router.post("/trading/stop")
async def stop_trading():
    """Stop automated trading"""
    if not trading_service:
        raise HTTPException(status_code=503, detail="Trading service not initialized")
    
    success = trading_service.stop_trading()
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to stop trading")
    
    return {"message": "Trading stopped"}


@router.get("/trading/status", response_model=TradingStatusResponse)
async def get_trading_status():
    """Get current trading status"""
    if not trading_service:
        raise HTTPException(status_code=503, detail="Trading service not initialized")
    
    status = trading_service.get_trading_status()
    return TradingStatusResponse(**status)


@router.get("/trading/history")
async def get_trading_history():
    """Get trading history"""
    if not trading_service:
        raise HTTPException(status_code=503, detail="Trading service not initialized")
    
    history = trading_service.get_trading_history()
    return {"history": history, "total_trades": len(history)}


# Monitoring Endpoints
@router.get("/monitoring/performance")
async def get_performance_metrics(hours: int = 24):
    """Get trading performance metrics"""
    from app.monitor import monitor
    return monitor.get_trading_performance(hours)


@router.get("/monitoring/system")
async def get_system_status():
    """Get system health status"""
    from app.monitor import monitor
    return monitor.get_system_status()


@router.get("/monitoring/export")
async def export_metrics():
    """Export all metrics to file"""
    from app.monitor import monitor
    from datetime import datetime
    
    filename = f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = f"logs/{filename}"
    
    monitor.export_metrics(filepath)
    
    return {
        "message": "Metrics exported successfully",
        "filename": filename,
        "filepath": filepath
    }


# Helper functions
async def _train_model_background(symbol: str, timesteps: int):
    """Background task for model training"""
    import os
    
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    
    if not api_key or not api_secret:
        logger.error("API credentials not available for training")
        return
    
    success = ml_service.train_model(
        api_key=api_key,
        api_secret=api_secret,
        symbol=symbol,
        total_timesteps=timesteps
    )
    
    if success:
        logger.info(f"Model training completed for {symbol}")
    else:
        logger.error(f"Model training failed for {symbol}")


def _prepare_observation_simple(klines: List[List]) -> List[float]:
    """Prepare observation matching the training environment format"""
    import pandas as pd
    import numpy as np
    import ta
    
    try:
        # Convert klines to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate indicators (matching trading_env.py)
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_30'] = ta.trend.sma_indicator(df['close'], window=30)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ATR
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # CCI
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        
        # Volume indicators
        df['volume_sma'] = ta.volume.volume_sma(df['close'], df['volume'], window=10)
        
        # Fill NaN values
        df = df.fillna(0)
        
        # Prepare observation for last 30 steps
        window_size = 30
        observation = []
        
        # Get last 30 rows
        recent_data = df.tail(window_size)
        
        for _, row in recent_data.iterrows():
            # OHLCV (5 features)
            step_data = [
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ]
            
            # Technical indicators (15 features)
            step_data.extend([
                float(row['sma_10']),
                float(row['sma_30']),
                float(row['ema_12']),
                float(row['macd']),
                float(row['macd_signal']),
                float(row['rsi']),
                float(row['bb_high']),
                float(row['bb_low']),
                float(row['bb_mid']),
                float(row['stoch_k']),
                float(row['stoch_d']),
                float(row['atr']),
                float(row['williams_r']),
                float(row['cci']),
                float(row['volume_sma'])
            ])
            
            # Portfolio state (3 features) - simplified for prediction
            step_data.extend([
                1000.0,  # balance (dummy)
                0.0,     # crypto_held (dummy)
                1.0      # position_ratio (dummy)
            ])
            
            observation.extend(step_data)
        
        # Ensure we have exactly 23 * 30 = 690 features
        target_length = 23 * window_size
        while len(observation) < target_length:
            observation.append(0.0)
        
        return np.array(observation[:target_length], dtype=np.float32)
        
    except Exception as e:
        logger.error(f"Error preparing observation: {e}")
        # Fallback to simple observation
        prices = []
        for kline in klines[-30:]:
            # Basic OHLCV + dummy indicators + dummy portfolio
            step_data = [
                float(kline[1]),  # open
                float(kline[2]),  # high  
                float(kline[3]),  # low
                float(kline[4]),  # close
                float(kline[5])   # volume
            ]
            # Add 15 dummy indicators
            step_data.extend([0.0] * 15)
            # Add 3 dummy portfolio values
            step_data.extend([1000.0, 0.0, 1.0])
            
            prices.extend(step_data)
        
        # Ensure exact length
        target_length = 23 * 30
        while len(prices) < target_length:
            prices.append(0.0)
        
        return np.array(prices[:target_length], dtype=np.float32)


def set_services(binance_svc, ml_svc, trading_svc):
    """Set service references from main application"""
    global binance_service, ml_service, trading_service
    binance_service = binance_svc
    ml_service = ml_svc
    trading_service = trading_svc