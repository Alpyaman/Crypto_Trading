"""
WebSocket-enabled ML endpoints for real-time training updates
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import List, Dict, Any, Optional
import asyncio
import json
import logging
from datetime import datetime

from app.services.enhanced_ml_service import EnhancedMLService
from app.services.training_state import TrainingStateManager
from app.core.error_handling import APIResponse, MLServiceException

router = APIRouter(prefix="/api/v1/ml", tags=["ML WebSocket API"])
logger = logging.getLogger(__name__)

# Global service references
_ml_service: Optional[EnhancedMLService] = None
_training_state: Optional[TrainingStateManager] = None

# WebSocket connection manager
class TrainingWebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.last_progress: Dict[str, Any] = {}
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
        
        # Send current state immediately
        if self.last_progress:
            try:
                await websocket.send_json(self.last_progress)
            except Exception as e:
                logger.error(f"Failed to send initial progress: {e}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast_progress(self, progress_data: Dict[str, Any]):
        """Broadcast training progress to all connected clients"""
        if not self.active_connections:
            return
        
        self.last_progress = progress_data
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_json(progress_data)
            except Exception as e:
                logger.warning(f"Failed to send to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

# Global WebSocket manager
websocket_manager = TrainingWebSocketManager()

def set_ml_services(ml_service: EnhancedMLService, training_state: TrainingStateManager):
    """Set service instances for WebSocket endpoints"""
    global _ml_service, _training_state
    _ml_service = ml_service
    _training_state = training_state

def get_ml_service() -> EnhancedMLService:
    """Dependency to get ML service"""
    if _ml_service is None:
        raise MLServiceException("ML service not initialized", "SERVICE_NOT_AVAILABLE")
    return _ml_service

def get_training_state() -> TrainingStateManager:
    """Dependency to get training state manager"""
    if _training_state is None:
        raise MLServiceException("Training state manager not initialized", "SERVICE_NOT_AVAILABLE")
    return _training_state

@router.websocket("/ws/training")
async def training_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time training progress updates"""
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Keep connection alive and handle incoming messages
            try:
                # Wait for ping or client messages with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                
                # Handle client requests
                try:
                    message = json.loads(data)
                    if message.get("type") == "get_progress":
                        # Send current progress immediately
                        progress = get_current_training_progress()
                        await websocket.send_json(progress)
                except json.JSONDecodeError:
                    # Ignore non-JSON messages (like pings)
                    pass
                    
            except asyncio.TimeoutError:
                # Send periodic heartbeat
                await websocket.send_json({"type": "heartbeat", "timestamp": datetime.utcnow().isoformat()})
                
    except WebSocketDisconnect:
        logger.info("Training WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Training WebSocket error: {e}")
    finally:
        websocket_manager.disconnect(websocket)

@router.websocket("/ws/market")
async def market_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time market data updates"""
    await websocket.accept()
    
    try:
        while True:
            # Get current market data
            market_data = await get_current_market_data()
            await websocket.send_json(market_data)
            
            # Wait 5 seconds before next update
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        logger.info("Market WebSocket client disconnected")
    except Exception as e:
        logger.error(f"Market WebSocket error: {e}")

@router.get("/training/progress")
async def get_training_progress_http(training_state: TrainingStateManager = Depends(get_training_state)):
    """HTTP endpoint for training progress (fallback for WebSocket)"""
    try:
        progress = training_state.get_progress()
        return APIResponse.success("Training progress retrieved", progress)
    except Exception as e:
        logger.error(f"Failed to get training progress: {e}")
        raise MLServiceException("Failed to retrieve training progress", "PROGRESS_ERROR")

def get_current_training_progress() -> Dict[str, Any]:
    """Get current training progress with error handling"""
    try:
        if _training_state is None:
            return {
                "status": "not_initialized",
                "message": "Training state manager not available",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        progress = _training_state.get_progress()
        
        # Enhance progress data with additional info
        enhanced_progress = {
            **progress,
            "timestamp": datetime.utcnow().isoformat(),
            "websocket_active": len(websocket_manager.active_connections) > 0,
            "connected_clients": len(websocket_manager.active_connections)
        }
        
        return enhanced_progress
        
    except Exception as e:
        logger.error(f"Failed to get training progress: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

async def get_current_market_data() -> Dict[str, Any]:
    """Get current market data for WebSocket streaming"""
    try:
        # This would integrate with your Binance service
        # For now, return mock data structure
        return {
            "type": "market_update",
            "symbol": "BTCUSDT",
            "price": 35000.0,  # This would come from real API
            "change_24h": 2.5,
            "volume": 1000000,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get market data: {e}")
        return {
            "type": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Background task to broadcast training updates
async def broadcast_training_updates():
    """Background task that periodically broadcasts training updates"""
    while True:
        try:
            if websocket_manager.active_connections:
                progress = get_current_training_progress()
                await websocket_manager.broadcast_progress(progress)
        except Exception as e:
            logger.error(f"Failed to broadcast training updates: {e}")
        
        await asyncio.sleep(2)  # Update every 2 seconds

# Function to manually trigger progress broadcast (called from training service)
async def trigger_progress_broadcast(progress_data: Dict[str, Any]):
    """Manually trigger a progress broadcast"""
    try:
        await websocket_manager.broadcast_progress(progress_data)
    except Exception as e:
        logger.error(f"Failed to trigger progress broadcast: {e}")

# Export manager for external use
__all__ = ["router", "set_ml_services", "websocket_manager", "trigger_progress_broadcast"]