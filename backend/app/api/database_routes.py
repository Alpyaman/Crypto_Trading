"""
Database API endpoints for accessing stored data
Provides endpoints for trades, training sessions, and analytics
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, Any
from datetime import datetime

from app.services.database_service import db_service

router = APIRouter(prefix="/api/database", tags=["database"])

# Trade Endpoints
@router.get("/trades", response_model=Dict[str, Any])
async def get_trades(
    symbol: Optional[str] = Query(None, description="Trading symbol filter"),
    limit: int = Query(100, description="Maximum number of trades to return"),
    days_back: int = Query(30, description="Number of days to look back")
):
    """Get trade history with optional filtering"""
    try:
        trades = db_service.get_trades(symbol=symbol, limit=limit, days_back=days_back)
        
        return {
            "success": True,
            "message": f"Retrieved {len(trades)} trades",
            "data": {
                "trades": [trade.to_dict() for trade in trades],
                "count": len(trades),
                "filters": {
                    "symbol": symbol,
                    "limit": limit,
                    "days_back": days_back
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve trades: {str(e)}")

@router.get("/trades/statistics", response_model=Dict[str, Any])
async def get_trade_statistics(
    symbol: Optional[str] = Query(None, description="Trading symbol filter"),
    days_back: int = Query(30, description="Number of days to analyze")
):
    """Get trading performance statistics"""
    try:
        stats = db_service.get_trade_statistics(symbol=symbol, days_back=days_back)
        
        return {
            "success": True,
            "message": "Trade statistics retrieved successfully",
            "data": {
                "statistics": stats,
                "filters": {
                    "symbol": symbol,
                    "days_back": days_back
                },
                "period": f"Last {days_back} days"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve trade statistics: {str(e)}")

# Training Session Endpoints
@router.get("/training/sessions", response_model=Dict[str, Any])
async def get_training_sessions(
    symbol: Optional[str] = Query(None, description="Training symbol filter"),
    algorithm: Optional[str] = Query(None, description="Algorithm filter (PPO, A2C, SAC, DQN)"),
    limit: int = Query(50, description="Maximum number of sessions to return")
):
    """Get training session history"""
    try:
        sessions = db_service.get_training_sessions(
            symbol=symbol, 
            algorithm=algorithm, 
            limit=limit
        )
        
        return {
            "success": True,
            "message": f"Retrieved {len(sessions)} training sessions",
            "data": {
                "sessions": [session.to_dict() for session in sessions],
                "count": len(sessions),
                "filters": {
                    "symbol": symbol,
                    "algorithm": algorithm,
                    "limit": limit
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve training sessions: {str(e)}")

@router.get("/training/active", response_model=Dict[str, Any])
async def get_active_training_sessions():
    """Get currently active training sessions"""
    try:
        sessions = db_service.get_active_training_sessions()
        
        return {
            "success": True,
            "message": f"Found {len(sessions)} active training sessions",
            "data": {
                "active_sessions": [session.to_dict() for session in sessions],
                "count": len(sessions)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve active training sessions: {str(e)}")

@router.get("/training/{session_id}/metrics", response_model=Dict[str, Any])
async def get_training_metrics(
    session_id: str,
    limit: int = Query(1000, description="Maximum number of metric points to return")
):
    """Get training metrics for a specific session"""
    try:
        # First, find the training session by session_id
        sessions = db_service.get_training_sessions()
        training_session = None
        for session in sessions:
            if session.session_id == session_id:
                training_session = session
                break
        
        if not training_session:
            raise HTTPException(status_code=404, detail=f"Training session {session_id} not found")
        
        metrics = db_service.get_training_metrics(
            training_session_id=training_session.id, 
            limit=limit
        )
        
        return {
            "success": True,
            "message": f"Retrieved {len(metrics)} metric points for session {session_id}",
            "data": {
                "session_info": training_session.to_dict(),
                "metrics": [metric.to_dict() for metric in metrics],
                "count": len(metrics)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve training metrics: {str(e)}")

# Account Balance Endpoints
@router.get("/balance/current", response_model=Dict[str, Any])
async def get_current_balance(
    account_type: str = Query("futures", description="Account type (spot, futures)")
):
    """Get latest account balance"""
    try:
        balance = db_service.get_latest_balance(account_type=account_type)
        
        if not balance:
            return {
                "success": True,
                "message": f"No balance records found for {account_type} account",
                "data": None,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return {
            "success": True,
            "message": f"Latest {account_type} balance retrieved",
            "data": {
                "balance": balance.to_dict(),
                "account_type": account_type
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve account balance: {str(e)}")

# Analytics Endpoints
@router.get("/analytics/performance", response_model=Dict[str, Any])
async def get_performance_summary(
    days_back: int = Query(30, description="Number of days to analyze")
):
    """Get comprehensive performance analytics"""
    try:
        summary = db_service.get_performance_summary(days_back=days_back)
        
        return {
            "success": True,
            "message": f"Performance summary for last {days_back} days",
            "data": summary,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve performance summary: {str(e)}")

@router.get("/analytics/dashboard", response_model=Dict[str, Any])
async def get_dashboard_data():
    """Get consolidated dashboard data"""
    try:
        # Get recent data for dashboard
        recent_trades = db_service.get_trades(limit=10, days_back=7)
        trade_stats = db_service.get_trade_statistics(days_back=7)
        training_sessions = db_service.get_training_sessions(limit=5)
        active_trainings = db_service.get_active_training_sessions()
        latest_balance = db_service.get_latest_balance()
        
        return {
            "success": True,
            "message": "Dashboard data retrieved successfully",
            "data": {
                "recent_trades": {
                    "trades": [trade.to_dict() for trade in recent_trades],
                    "count": len(recent_trades)
                },
                "trading_stats": trade_stats,
                "training_overview": {
                    "recent_sessions": [session.to_dict() for session in training_sessions],
                    "active_sessions": [session.to_dict() for session in active_trainings],
                    "active_count": len(active_trainings)
                },
                "account_balance": latest_balance.to_dict() if latest_balance else None
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dashboard data: {str(e)}")

# Data Management Endpoints
@router.post("/trades", response_model=Dict[str, Any])
async def create_trade_record(trade_data: Dict[str, Any]):
    """Create a new trade record"""
    try:
        trade = db_service.create_trade(trade_data)
        
        return {
            "success": True,
            "message": "Trade record created successfully",
            "data": {
                "trade": trade.to_dict()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create trade record: {str(e)}")

@router.post("/training/sessions", response_model=Dict[str, Any])
async def create_training_session_record(session_data: Dict[str, Any]):
    """Create a new training session record"""
    try:
        session = db_service.create_training_session(session_data)
        
        return {
            "success": True,
            "message": "Training session created successfully",
            "data": {
                "session": session.to_dict()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create training session: {str(e)}")

@router.put("/training/sessions/{session_id}", response_model=Dict[str, Any])
async def update_training_session_record(session_id: str, updates: Dict[str, Any]):
    """Update a training session record"""
    try:
        session = db_service.update_training_session(session_id, updates)
        
        if not session:
            raise HTTPException(status_code=404, detail=f"Training session {session_id} not found")
        
        return {
            "success": True,
            "message": "Training session updated successfully",
            "data": {
                "session": session.to_dict()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update training session: {str(e)}")

@router.post("/balance", response_model=Dict[str, Any])
async def save_balance_snapshot(balance_data: Dict[str, Any]):
    """Save account balance snapshot"""
    try:
        balance = db_service.save_account_balance(balance_data)
        
        return {
            "success": True,
            "message": "Balance snapshot saved successfully",
            "data": {
                "balance": balance.to_dict()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save balance snapshot: {str(e)}")

@router.post("/metrics", response_model=Dict[str, Any])
async def add_system_metric_record(metric_data: Dict[str, Any]):
    """Add a system metric record"""
    try:
        metric = db_service.add_system_metric(metric_data)
        
        return {
            "success": True,
            "message": "System metric added successfully",
            "data": {
                "metric": metric.to_dict()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add system metric: {str(e)}")

# Health Check
@router.get("/health", response_model=Dict[str, Any])
async def database_health_check():
    """Check database connectivity and health"""
    try:
        # Try to perform a simple query to test connectivity
        db_service.get_training_sessions(limit=1)
        
        return {
            "success": True,
            "message": "Database is healthy and accessible",
            "data": {
                "status": "healthy",
                "connection": "active",
                "last_check": datetime.utcnow().isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database health check failed: {str(e)}")