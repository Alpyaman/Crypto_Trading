"""
API v1 main router
"""
from fastapi import APIRouter
from app.api.v1.endpoints import ml

router = APIRouter()

# Include endpoint routers
router.include_router(ml.router)

# Export WebSocket manager for external use
websocket_manager = ml.websocket_manager
set_ml_services = ml.set_ml_services
trigger_progress_broadcast = ml.trigger_progress_broadcast