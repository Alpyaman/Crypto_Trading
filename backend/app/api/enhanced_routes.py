"""
Enhanced API Routes for Advanced ML Futures Trading
New endpoints for enhanced ML model training, prediction, and trading
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
import logging

from app.services.enhanced_ml_service import EnhancedMLService
from app.services.enhanced_trading_service import EnhancedTradingService
from app.config import get_binance_credentials

logger = logging.getLogger(__name__)

# Initialize enhanced services
enhanced_ml_service = EnhancedMLService()
enhanced_trading_service = None  # Will be initialized when needed

router = APIRouter(prefix="/enhanced", tags=["Enhanced ML Trading"])


class EnhancedTrainingRequest(BaseModel):
    symbol: str = "BTCUSDT"
    total_timesteps: int = 200000
    algorithm: str = "PPO"  # PPO or A2C
    

class EnhancedTradingRequest(BaseModel):
    symbol: str = "BTCUSDT"
    mode: str = "balanced"  # conservative, balanced, aggressive
    leverage: int = 10


class EnhancedPredictionRequest(BaseModel):
    symbol: str = "BTCUSDT"
    account_balance: float = 10000.0


@router.post("/train")
async def train_enhanced_model(request: EnhancedTrainingRequest, background_tasks: BackgroundTasks):
    """Train enhanced ML model with advanced features"""
    try:
        api_key, api_secret = get_binance_credentials()
        
        # Start training in background
        background_tasks.add_task(
            enhanced_ml_service.train_enhanced_model,
            api_key=api_key,
            api_secret=api_secret,
            symbol=request.symbol,
            total_timesteps=request.total_timesteps,
            algorithm=request.algorithm
        )
        
        return {
            "status": "success",
            "message": f"Enhanced {request.algorithm} model training started for {request.symbol}",
            "training_params": {
                "symbol": request.symbol,
                "timesteps": request.total_timesteps,
                "algorithm": request.algorithm
            }
        }
        
    except Exception as e:
        logger.error(f"Error starting enhanced model training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/load-model")
async def load_enhanced_model():
    """Load trained enhanced model"""
    try:
        success = enhanced_ml_service.load_enhanced_model()
        
        if success:
            model_info = enhanced_ml_service.get_enhanced_model_info()
            return {
                "status": "success",
                "message": "Enhanced model loaded successfully",
                "model_info": model_info
            }
        else:
            raise HTTPException(status_code=404, detail="Enhanced model not found")
            
    except Exception as e:
        logger.error(f"Error loading enhanced model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-info")
async def get_enhanced_model_info():
    """Get enhanced model information"""
    try:
        info = enhanced_ml_service.get_enhanced_model_info()
        return {
            "status": "success",
            "model_info": info
        }
        
    except Exception as e:
        logger.error(f"Error getting enhanced model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
async def get_enhanced_prediction(request: EnhancedPredictionRequest):
    """Get enhanced prediction with comprehensive analysis"""
    try:
        if not enhanced_ml_service.model:
            raise HTTPException(status_code=400, detail="Enhanced model not loaded")
        
        api_key, api_secret = get_binance_credentials()
        
        # Get market data
        from binance.client import Client
        client = Client(api_key, api_secret)
        
        klines = client.get_klines(
            symbol=request.symbol,
            interval='1h',
            limit=200
        )
        
        if not klines:
            raise HTTPException(status_code=400, detail="Could not fetch market data")
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Extract enhanced features
        enhanced_data = enhanced_ml_service.extract_enhanced_features(df, request.symbol)
        
        # Create observation (simplified)
        observation = enhanced_data.tail(1).values.flatten()[:100]  # Take first 100 features
        
        # Pad if needed
        if len(observation) < 100:
            observation = list(observation) + [0.0] * (100 - len(observation))
        
        import numpy as np
        observation = np.array(observation[:100], dtype=np.float32)
        
        # Get enhanced prediction
        action, confidence, position_size, analysis = enhanced_ml_service.predict_enhanced(
            observation=observation,
            market_data=enhanced_data,
            account_balance=request.account_balance
        )
        
        # Map action to human-readable format
        action_names = {0: 'CLOSE', 1: 'LONG', 2: 'SHORT', 3: 'HOLD'}
        
        return {
            "status": "success",
            "prediction": {
                "action": action,
                "action_name": action_names.get(action, 'UNKNOWN'),
                "confidence": confidence,
                "position_size": position_size,
                "current_price": float(enhanced_data['close'].iloc[-1])
            },
            "analysis": analysis,
            "symbol": request.symbol
        }
        
    except Exception as e:
        logger.error(f"Error getting enhanced prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trading/start")
async def start_enhanced_trading(request: EnhancedTradingRequest):
    """Start enhanced automated trading"""
    try:
        global enhanced_trading_service
        
        if not enhanced_ml_service.model:
            raise HTTPException(status_code=400, detail="Enhanced model not loaded")
        
        # Initialize trading service
        from app.services.binance_service import BinanceService
        api_key, api_secret = get_binance_credentials()
        binance_service = BinanceService(api_key, api_secret)
        
        enhanced_trading_service = EnhancedTradingService(binance_service, enhanced_ml_service)
        
        # Start trading
        success = await enhanced_trading_service.start_enhanced_trading(
            symbol=request.symbol,
            mode=request.mode,
            leverage=request.leverage
        )
        
        if success:
            return {
                "status": "success",
                "message": f"Enhanced trading started for {request.symbol}",
                "config": {
                    "symbol": request.symbol,
                    "mode": request.mode,
                    "leverage": request.leverage
                }
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to start enhanced trading")
            
    except Exception as e:
        logger.error(f"Error starting enhanced trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trading/stop")
async def stop_enhanced_trading():
    """Stop enhanced automated trading"""
    try:
        global enhanced_trading_service
        
        if not enhanced_trading_service:
            return {"status": "success", "message": "Enhanced trading not active"}
        
        success = enhanced_trading_service.stop_enhanced_trading()
        
        if success:
            return {
                "status": "success",
                "message": "Enhanced trading stopped"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to stop enhanced trading")
            
    except Exception as e:
        logger.error(f"Error stopping enhanced trading: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trading/status")
async def get_enhanced_trading_status():
    """Get enhanced trading status and performance"""
    try:
        global enhanced_trading_service
        
        if not enhanced_trading_service:
            return {
                "status": "success",
                "trading_status": {
                    "is_trading": False,
                    "message": "Enhanced trading service not initialized"
                }
            }
        
        status = enhanced_trading_service.get_enhanced_status()
        
        return {
            "status": "success",
            "trading_status": status
        }
        
    except Exception as e:
        logger.error(f"Error getting enhanced trading status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market-analysis/{symbol}")
async def get_enhanced_market_analysis(symbol: str):
    """Get comprehensive market analysis"""
    try:
        api_key, api_secret = get_binance_credentials()
        
        # Get market data
        from binance.client import Client
        client = Client(api_key, api_secret)
        
        klines = client.get_klines(
            symbol=symbol,
            interval='1h',
            limit=200
        )
        
        if not klines:
            raise HTTPException(status_code=400, detail="Could not fetch market data")
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Extract enhanced features
        enhanced_data = enhanced_ml_service.extract_enhanced_features(df, symbol)
        
        # Detect market regime
        regime = enhanced_ml_service.detect_market_regime(enhanced_data)
        
        # Get current market metrics
        current_data = enhanced_data.iloc[-1]
        
        analysis = {
            "symbol": symbol,
            "current_price": float(current_data['close']),
            "market_regime": regime,
            "technical_indicators": {
                "rsi_14": float(current_data.get('rsi_14', 0)),
                "macd": float(current_data.get('macd', 0)),
                "bb_position": float(current_data.get('bb_position', 0)),
                "volume_ratio": float(current_data.get('volume_ratio', 0)),
                "volatility": float(current_data.get('volatility_ratio', 0))
            },
            "trend_analysis": {
                "sma_20": float(current_data.get('sma_20', 0)),
                "ema_20": float(current_data.get('ema_20', 0)),
                "price_change": float(current_data.get('price_change', 0)),
                "trend_strength": enhanced_ml_service._calculate_trend_strength(enhanced_data)
            },
            "risk_metrics": {
                "support_resistance": enhanced_ml_service._get_support_resistance(enhanced_data),
                "risk_score": enhanced_ml_service._calculate_risk_score(enhanced_data)
            }
        }
        
        return {
            "status": "success",
            "analysis": analysis
        }
        
    except Exception as e:
        logger.error(f"Error getting enhanced market analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/extract/{symbol}")
async def extract_enhanced_features(symbol: str, limit: int = 100):
    """Extract and return enhanced features for analysis"""
    try:
        api_key, api_secret = get_binance_credentials()
        
        # Get market data
        from binance.client import Client
        client = Client(api_key, api_secret)
        
        klines = client.get_klines(
            symbol=symbol,
            interval='1h',
            limit=limit
        )
        
        if not klines:
            raise HTTPException(status_code=400, detail="Could not fetch market data")
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Extract enhanced features
        enhanced_data = enhanced_ml_service.extract_enhanced_features(df, symbol)
        
        # Return feature summary
        feature_summary = {
            "symbol": symbol,
            "data_points": len(enhanced_data),
            "feature_count": len(enhanced_data.columns),
            "feature_names": list(enhanced_data.columns),
            "latest_values": enhanced_data.iloc[-1].to_dict(),
            "statistics": {
                "mean_price": float(enhanced_data['close'].mean()),
                "std_price": float(enhanced_data['close'].std()),
                "min_price": float(enhanced_data['close'].min()),
                "max_price": float(enhanced_data['close'].max())
            }
        }
        
        return {
            "status": "success",
            "features": feature_summary
        }
        
    except Exception as e:
        logger.error(f"Error extracting enhanced features: {e}")
        raise HTTPException(status_code=500, detail=str(e))