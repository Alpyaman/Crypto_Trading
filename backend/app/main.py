"""
FastAPI Backend Server with Prometheus Monitoring
Main entry point for the trading application
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel
from dotenv import load_dotenv
import logging
import os
import time
from datetime import datetime

from app.config import config
from app.core.error_handling import handle_trading_exception

from app.services.binance_service import BinanceService
from app.services.ml_service import MLService
from app.services.trading_service import TradingService
from app.services.monitoring_service import monitoring_service
# from app.services.enhanced_ml_service import EnhancedMLService
# from app.services.training_state import TrainingStateManager
from app.api.routes import router, set_services
from app.api.enhanced_routes import router as enhanced_router
from app.api.v1.api import router as api_v1_router, set_ml_services as set_websocket_services
from app.api.validated_routes import router as validated_router, set_services as set_validated_services
from app.api.database_routes import router as database_router

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=getattr(logging, config.log_level.upper()))
logger = logging.getLogger(__name__)

# Lifespan event handler
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    await startup_event()
    yield
    # Shutdown
    await shutdown_event()

# Initialize FastAPI with lifespan
app = FastAPI(
    title="Crypto Trading AI", 
    version="1.0.0",
    debug=config.debug,
    description="AI-powered cryptocurrency trading application with machine learning",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus monitoring middleware
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Monitor API requests for Prometheus metrics"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Record metrics
        monitoring_service.record_api_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration=duration
        )
        
        return response
    except Exception as e:
        duration = time.time() - start_time
        
        # Record error
        monitoring_service.record_api_error(type(e).__name__)
        monitoring_service.record_api_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=500,
            duration=duration
        )
        
        raise

# Initialize services
binance_service = None
ml_service = None
trading_service = None

# Global state
trading_active = False
current_symbol = "BTCUSDT"


# Pydantic models
class StartTradingRequest(BaseModel):
    symbol: str = "BTCUSDT"
    mode: str = "balanced"  # conservative, balanced, aggressive


class TrainModelRequest(BaseModel):
    symbol: str = "BTCUSDT"
    timesteps: int = 100000


class OrderRequest(BaseModel):
    symbol: str
    side: str  # BUY or SELL
    quantity: float


# Include API routes
app.include_router(router)
app.include_router(enhanced_router)
app.include_router(api_v1_router)
app.include_router(validated_router)
app.include_router(database_router)

# Add enhanced error handling middleware
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for better error responses"""
    try:
        return handle_trading_exception(exc)
    except Exception:
        # Fallback for any exception in the exception handler
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "error_code": "INTERNAL_ERROR",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
app.include_router(enhanced_router)

# Mount static files for frontend
frontend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "frontend")
logger.info(f"Looking for frontend at: {frontend_path}")
logger.info(f"Frontend exists: {os.path.exists(frontend_path)}")

if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")
    logger.info("Frontend static files mounted successfully")
    
    # Serve the main GUI page at root
    @app.get("/")
    async def serve_gui():
        """Serve the main GUI page"""
        gui_file = os.path.join(frontend_path, "index.html")
        if os.path.exists(gui_file):
            return FileResponse(gui_file)
        else:
            raise HTTPException(status_code=404, detail="GUI not found")
else:
    logger.warning(f"Frontend directory not found at {frontend_path}")
    
    @app.get("/")
    async def root():
        return {"message": "Crypto Trading AI Backend", "frontend": f"Not found at {frontend_path}", "status": "running"}

async def startup_event():
    """Initialize services on startup"""
    global binance_service, ml_service, trading_service
    
    logger.info("Starting Crypto Trading AI application...")
    
    # Start monitoring service
    monitoring_service.start_system_monitoring()
    logger.info("📊 Prometheus monitoring service started")
    
    # Initialize database
    try:
        from app.models.database import init_database
        init_database()
        logger.info("📊 Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    
    api_key = config.binance_api_key
    api_secret = config.binance_api_secret
    
    if not api_key or not api_secret or api_key == "" or api_secret == "":
        logger.warning("Binance API credentials not found in configuration")
        logger.warning("Some features will not be available without API credentials")
        logger.warning("Please check your .env file and ensure BINANCE_API_KEY and BINANCE_API_SECRET are set")
        return
    
    try:
        # Initialize services
        binance_service = BinanceService(api_key, api_secret, testnet=config.binance_testnet)
        ml_service = MLService(model_path=config.trading.model_path)
        trading_service = TradingService(binance_service, ml_service)
        
        # Initialize enhanced services for WebSocket support
        try:
            from app.services.enhanced_ml_service import EnhancedMLService
            from app.services.enhanced_trading_service import EnhancedTradingService
            from app.services.training_state import TrainingStateManager
            
            enhanced_ml_service = EnhancedMLService(model_path=config.trading.model_path)
            training_state = TrainingStateManager()
            enhanced_trading_service = EnhancedTradingService(binance_service, enhanced_ml_service)
            
            # Set WebSocket services
            set_websocket_services(enhanced_ml_service, training_state)
            logger.info("WebSocket services initialized successfully")
            
            # Set validated API services
            set_validated_services(binance_service, enhanced_ml_service, enhanced_trading_service, training_state)
            logger.info("Validated API services initialized successfully")
            
        except ImportError as e:
            logger.warning(f"Enhanced services not available: {e}")
        except Exception as e:
            logger.warning(f"Failed to initialize enhanced services: {e}")
        
        # Set services in routes module
        set_services(binance_service, ml_service, trading_service)
        
        # Try to load existing model
        model_loaded = ml_service.load_model()
        if model_loaded:
            logger.info("Pre-trained model loaded successfully")
        else:
            logger.info("No pre-trained model found - you can train one via the API")
        
        logger.info("Services initialized successfully")
        logger.info(f"Environment: {config.environment}")
        logger.info(f"Testnet mode: {config.binance_testnet}")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")


async def shutdown_event():
    """Cleanup on shutdown"""
    global trading_service
    
    logger.info("Shutting down Crypto Trading AI application...")
    
    # Stop monitoring service
    monitoring_service.stop_system_monitoring()
    logger.info("📊 Prometheus monitoring service stopped")
    
    # Stop any active trading
    if trading_service and trading_service.is_trading:
        logger.info("Stopping active trading...")
        trading_service.stop_trading()
    
    logger.info("Application shutdown complete")


# Health check
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "version": "1.0.0",
        "environment": config.environment,
        "testnet_mode": config.binance_testnet,
        "services": {
            "binance": binance_service is not None,
            "ml": ml_service is not None,
            "trading": trading_service is not None
        },
        "model_status": {
            "loaded": ml_service.model is not None if ml_service else False,
            "path": config.trading.model_path
        }
    }


@app.get("/config")
async def get_config():
    """Get application configuration (safe values only)"""
    return {
        "environment": config.environment,
        "debug": config.debug,
        "testnet_mode": config.binance_testnet,
        "api_credentials_configured": bool(config.binance_api_key and config.binance_api_secret),
        "trading_config": {
            "default_symbol": config.trading.default_symbol,
            "trading_interval": config.trading.trading_interval,
            "conservative_mode": config.trading.conservative_mode,
            "balanced_mode": config.trading.balanced_mode,
            "aggressive_mode": config.trading.aggressive_mode
        }
    }


# Legacy endpoints (for backward compatibility)
@app.get("/balance")
async def get_balance_legacy():
    """Get account balance (legacy endpoint)"""
    if not binance_service:
        raise HTTPException(status_code=503, detail="Service not available")
    return binance_service.get_account_balance()


@app.get("/price/{symbol}")
async def get_price_legacy(symbol: str):
    """Get current price (legacy endpoint)"""
    if not binance_service:
        raise HTTPException(status_code=503, detail="Service not available")
    
    price = binance_service.get_current_price(symbol)
    if price is None:
        raise HTTPException(status_code=404, detail="Price not found")
    
    return {"symbol": symbol, "price": price}


# === MONITORING AND OBSERVABILITY ENDPOINTS ===

@app.get("/metrics", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """
    Prometheus metrics endpoint
    Returns metrics in Prometheus text format for scraping
    """
    return monitoring_service.get_prometheus_metrics()

@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint
    Returns system health status with detailed metrics
    """
    return monitoring_service.get_health_status()

@app.get("/metrics/snapshot")
async def get_metrics_snapshot():
    """
    Get current metrics snapshot in JSON format
    Useful for dashboards and debugging
    """
    return monitoring_service.get_metrics_snapshot()

@app.post("/metrics/export")
async def export_metrics(request: Request):
    """
    Export current metrics to JSON file
    """
    try:
        body = await request.json()
        filepath = body.get("filepath", f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        monitoring_service.export_metrics_json(filepath)
        
        return {
            "status": "success",
            "message": f"Metrics exported to {filepath}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error exporting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Use import string for reload functionality
    if config.reload and config.debug:
        uvicorn.run(
            "app.main:app",
            host=config.host,
            port=config.port,
            reload=True,
            log_level=config.log_level.lower()
        )
    else:
        uvicorn.run(
            app,
            host=config.host,
            port=config.port,
            reload=False,
            log_level=config.log_level.lower()
        )