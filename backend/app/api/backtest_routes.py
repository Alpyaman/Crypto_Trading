"""
Backtesting API Routes
Provides endpoints for historical trading simulation and performance analytics
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Create router
backtest_router = APIRouter(prefix="/api/v3/backtest", tags=["backtesting"])

# Request/Response models
class BacktestRequest(BaseModel):
    symbol: str = Field(..., description="Trading pair (e.g., 'BTCUSDT')")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")
    model_type: str = Field(default="enhanced", description="Model type: 'enhanced' or 'basic'")
    initial_balance: float = Field(default=10000.0, description="Starting balance in USDT")
    commission_rate: float = Field(default=0.001, description="Commission rate (0.001 = 0.1%)")
    position_size_mode: str = Field(default="balanced", description="Position sizing: 'conservative', 'balanced', or 'aggressive'")

class BacktestResponse(BaseModel):
    status: str
    data: Optional[Dict[str, Any]] = None
    message: str = ""
    execution_time: float = 0.0

class BacktestListResponse(BaseModel):
    status: str
    backtests: List[Dict[str, Any]]
    total: int

# Dependency to get services
async def get_services():
    from app.main import app
    
    # Get services from app state
    try:
        binance_service = app.state.binance_service
        enhanced_ml_service = app.state.enhanced_ml_service
        backtesting_service = app.state.backtesting_service
        return binance_service, enhanced_ml_service, backtesting_service
    except AttributeError:
        # Fallback for when app state is not available
        from app.main import binance_service
        try:
            from app.services.enhanced_ml_service import EnhancedMLService
            enhanced_ml_service = EnhancedMLService()
        except Exception:
            enhanced_ml_service = None
            
        if not all([binance_service, enhanced_ml_service]):
            raise HTTPException(status_code=503, detail="Required services not available")
        
        # Create backtesting service on demand
        from app.services.backtesting_service import BacktestingService
        backtesting_service = BacktestingService(binance_service, enhanced_ml_service)
        return binance_service, enhanced_ml_service, backtesting_service

@backtest_router.post("/run", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    services = Depends(get_services)
):
    """
    Run a comprehensive backtest simulation
    
    This endpoint performs historical trading simulation using the specified model
    and parameters, returning detailed performance analytics.
    """
    try:
        start_time = datetime.now()
        binance_service, enhanced_ml_service, backtesting_service = services
        
        logger.info(f"Starting backtest: {request.symbol} ({request.start_date} to {request.end_date})")
        
        # Validate parameters
        try:
            start_dt = datetime.strptime(request.start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(request.end_date, "%Y-%m-%d")
            
            if start_dt >= end_dt:
                raise ValueError("Start date must be before end date")
            
            if end_dt > datetime.now():
                raise ValueError("End date cannot be in the future")
            
            if (end_dt - start_dt).days < 1:
                raise ValueError("Backtest period must be at least 1 day")
                
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid date format or range: {e}")
        
        # Validate other parameters
        if request.initial_balance <= 0:
            raise HTTPException(status_code=400, detail="Initial balance must be positive")
        
        if not 0 <= request.commission_rate <= 0.1:
            raise HTTPException(status_code=400, detail="Commission rate must be between 0 and 0.1 (10%)")
        
        if request.model_type not in ["enhanced", "basic"]:
            raise HTTPException(status_code=400, detail="Model type must be 'enhanced' or 'basic'")
        
        if request.position_size_mode not in ["conservative", "balanced", "aggressive"]:
            raise HTTPException(status_code=400, detail="Position size mode must be 'conservative', 'balanced', or 'aggressive'")
        
        # Run backtest
        results = await backtesting_service.run_backtest(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            model_type=request.model_type,
            initial_balance=request.initial_balance,
            commission_rate=request.commission_rate,
            position_size_mode=request.position_size_mode
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Convert dataclass to dict for JSON response
        results_dict = {
            "performance_summary": {
                "total_trades": results.total_trades,
                "win_rate": round(results.win_rate, 2),
                "total_pnl": round(results.total_pnl, 2),
                "total_return": round(results.total_return, 2),
                "annualized_return": round(results.annualized_return, 2),
                "sharpe_ratio": round(results.sharpe_ratio, 3),
                "max_drawdown": round(results.max_drawdown, 2),
                "max_drawdown_percentage": round(results.max_drawdown_percentage, 2)
            },
            "detailed_metrics": {
                "profit_metrics": {
                    "gross_profit": round(results.gross_profit, 2),
                    "gross_loss": round(results.gross_loss, 2),
                    "profit_factor": round(results.profit_factor, 3),
                    "best_trade": round(results.best_trade, 2),
                    "worst_trade": round(results.worst_trade, 2),
                    "avg_trade": round(results.avg_trade, 2),
                    "avg_winning_trade": round(results.avg_winning_trade, 2),
                    "avg_losing_trade": round(results.avg_losing_trade, 2)
                },
                "risk_metrics": {
                    "volatility": round(results.volatility, 4),
                    "calmar_ratio": round(results.calmar_ratio, 3),
                    "sortino_ratio": round(results.sortino_ratio, 3),
                    "largest_win_streak": results.largest_win_streak,
                    "largest_loss_streak": results.largest_loss_streak
                },
                "trading_metrics": {
                    "winning_trades": results.winning_trades,
                    "losing_trades": results.losing_trades,
                    "trades_per_day": round(results.trades_per_day, 2),
                    "commission_paid": round(results.commission_paid, 2)
                }
            },
            "regime_analysis": results.regime_performance,
            "backtest_info": {
                "symbol": request.symbol,
                "start_date": results.start_date,
                "end_date": results.end_date,
                "duration_days": results.duration_days,
                "model_type": request.model_type,
                "initial_balance": request.initial_balance,
                "commission_rate": request.commission_rate,
                "position_size_mode": request.position_size_mode
            },
            "equity_curve": results.equity_curve[-1000:],  # Limit to last 1000 points for response size
            "trade_history": results.trades[-100:]  # Limit to last 100 trades
        }
        
        logger.info(f"Backtest completed in {execution_time:.2f}s: {results.total_trades} trades, "
                   f"{results.win_rate:.1f}% win rate, {results.total_pnl:.2f} P&L")
        
        return BacktestResponse(
            status="success",
            data=results_dict,
            message=f"Backtest completed successfully with {results.total_trades} trades",
            execution_time=execution_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest execution failed: {str(e)}")

@backtest_router.get("/quick-run", response_model=BacktestResponse)
async def quick_backtest(
    symbol: str = Query(..., description="Trading pair (e.g., 'BTCUSDT')"),
    days: int = Query(30, description="Number of days to backtest", ge=1, le=365),
    model_type: str = Query("enhanced", description="Model type: 'enhanced' or 'basic'"),
    services = Depends(get_services)
):
    """
    Run a quick backtest for the specified number of days
    
    This endpoint provides a simplified way to run backtests for recent periods
    with default parameters.
    """
    try:
        start_time = datetime.now()
        binance_service, enhanced_ml_service, backtesting_service = services
        
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        logger.info(f"Running quick backtest: {symbol} for {days} days")
        
        # Run backtest with default parameters
        results = await backtesting_service.run_backtest(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            model_type=model_type,
            initial_balance=10000.0,
            commission_rate=0.001,
            position_size_mode="balanced"
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Return summary results
        summary = {
            "quick_summary": {
                "symbol": symbol,
                "period_days": days,
                "total_trades": results.total_trades,
                "win_rate": round(results.win_rate, 2),
                "total_return": round(results.total_return, 2),
                "sharpe_ratio": round(results.sharpe_ratio, 3),
                "max_drawdown_pct": round(results.max_drawdown_percentage, 2),
                "profit_factor": round(results.profit_factor, 3)
            },
            "performance_grade": _calculate_performance_grade(results),
            "recommendations": _generate_recommendations(results)
        }
        
        return BacktestResponse(
            status="success",
            data=summary,
            message=f"Quick backtest completed: {results.total_trades} trades in {days} days",
            execution_time=execution_time
        )
        
    except Exception as e:
        logger.error(f"Quick backtest failed: {e}")
        raise HTTPException(status_code=500, detail=f"Quick backtest failed: {str(e)}")

@backtest_router.get("/performance-summary/{symbol}")
async def get_performance_summary(
    symbol: str,
    period: str = Query("30d", description="Period: '7d', '30d', '90d', '1y'"),
    services = Depends(get_services)
):
    """
    Get performance summary for different time periods
    """
    try:
        binance_service, enhanced_ml_service, backtesting_service = services
        
        # Define periods
        period_days = {
            "7d": 7,
            "30d": 30,
            "90d": 90,
            "1y": 365
        }
        
        if period not in period_days:
            raise HTTPException(status_code=400, detail="Invalid period. Use '7d', '30d', '90d', or '1y'")
        
        days = period_days[period]
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Run backtest
        results = await backtesting_service.run_backtest(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            model_type="enhanced",
            initial_balance=10000.0,
            commission_rate=0.001,
            position_size_mode="balanced"
        )
        
        # Create performance summary
        summary = {
            "symbol": symbol,
            "period": period,
            "summary": {
                "total_return": round(results.total_return, 2),
                "annualized_return": round(results.annualized_return, 2),
                "win_rate": round(results.win_rate, 2),
                "sharpe_ratio": round(results.sharpe_ratio, 3),
                "max_drawdown": round(results.max_drawdown_percentage, 2),
                "total_trades": results.total_trades,
                "profit_factor": round(results.profit_factor, 3)
            },
            "risk_assessment": _assess_risk(results),
            "market_regimes": results.regime_performance
        }
        
        return BacktestResponse(
            status="success",
            data=summary,
            message=f"Performance summary for {symbol} over {period}"
        )
        
    except Exception as e:
        logger.error(f"Performance summary failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance summary: {str(e)}")

@backtest_router.get("/compare")
async def compare_strategies(
    symbol: str = Query(..., description="Trading pair"),
    days: int = Query(30, description="Backtest period in days"),
    services = Depends(get_services)
):
    """
    Compare enhanced vs basic model performance
    """
    try:
        binance_service, enhanced_ml_service, backtesting_service = services
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Run both backtests in parallel
        enhanced_results = await backtesting_service.run_backtest(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            model_type="enhanced",
            initial_balance=10000.0,
            commission_rate=0.001,
            position_size_mode="balanced"
        )
        
        basic_results = await backtesting_service.run_backtest(
            symbol=symbol,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            model_type="basic",
            initial_balance=10000.0,
            commission_rate=0.001,
            position_size_mode="balanced"
        )
        
        # Compare results
        comparison = {
            "symbol": symbol,
            "period_days": days,
            "enhanced_model": {
                "total_return": round(enhanced_results.total_return, 2),
                "win_rate": round(enhanced_results.win_rate, 2),
                "sharpe_ratio": round(enhanced_results.sharpe_ratio, 3),
                "max_drawdown": round(enhanced_results.max_drawdown_percentage, 2),
                "total_trades": enhanced_results.total_trades,
                "profit_factor": round(enhanced_results.profit_factor, 3)
            },
            "basic_model": {
                "total_return": round(basic_results.total_return, 2),
                "win_rate": round(basic_results.win_rate, 2),
                "sharpe_ratio": round(basic_results.sharpe_ratio, 3),
                "max_drawdown": round(basic_results.max_drawdown_percentage, 2),
                "total_trades": basic_results.total_trades,
                "profit_factor": round(basic_results.profit_factor, 3)
            },
            "performance_difference": {
                "return_improvement": round(enhanced_results.total_return - basic_results.total_return, 2),
                "win_rate_improvement": round(enhanced_results.win_rate - basic_results.win_rate, 2),
                "sharpe_improvement": round(enhanced_results.sharpe_ratio - basic_results.sharpe_ratio, 3),
                "recommended_model": "enhanced" if enhanced_results.sharpe_ratio > basic_results.sharpe_ratio else "basic"
            }
        }
        
        return BacktestResponse(
            status="success",
            data=comparison,
            message="Strategy comparison completed"
        )
        
    except Exception as e:
        logger.error(f"Strategy comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Strategy comparison failed: {str(e)}")

# Helper functions
def _calculate_performance_grade(results) -> str:
    """Calculate a performance grade based on key metrics"""
    score = 0
    
    # Win rate (0-30 points)
    if results.win_rate >= 60:
        score += 30
    elif results.win_rate >= 50:
        score += 20
    elif results.win_rate >= 40:
        score += 10
    
    # Sharpe ratio (0-25 points)
    if results.sharpe_ratio >= 2.0:
        score += 25
    elif results.sharpe_ratio >= 1.5:
        score += 20
    elif results.sharpe_ratio >= 1.0:
        score += 15
    elif results.sharpe_ratio >= 0.5:
        score += 10
    
    # Profit factor (0-25 points)
    if results.profit_factor >= 2.0:
        score += 25
    elif results.profit_factor >= 1.5:
        score += 20
    elif results.profit_factor >= 1.2:
        score += 15
    elif results.profit_factor >= 1.0:
        score += 10
    
    # Max drawdown (0-20 points)
    if results.max_drawdown_percentage <= 5:
        score += 20
    elif results.max_drawdown_percentage <= 10:
        score += 15
    elif results.max_drawdown_percentage <= 20:
        score += 10
    elif results.max_drawdown_percentage <= 30:
        score += 5
    
    # Grade assignment
    if score >= 85:
        return "A+"
    elif score >= 75:
        return "A"
    elif score >= 65:
        return "B+"
    elif score >= 55:
        return "B"
    elif score >= 45:
        return "C+"
    elif score >= 35:
        return "C"
    else:
        return "D"

def _generate_recommendations(results) -> List[str]:
    """Generate trading recommendations based on backtest results"""
    recommendations = []
    
    if results.win_rate < 40:
        recommendations.append("Consider improving entry/exit criteria - win rate is below optimal")
    
    if results.sharpe_ratio < 1.0:
        recommendations.append("Risk-adjusted returns are low - consider position sizing optimization")
    
    if results.max_drawdown_percentage > 20:
        recommendations.append("Maximum drawdown is high - implement stricter risk management")
    
    if results.profit_factor < 1.2:
        recommendations.append("Profit factor is low - review trade selection and timing")
    
    if results.total_trades < 10:
        recommendations.append("Limited trading activity - consider more active strategy or longer backtest period")
    
    if not recommendations:
        recommendations.append("Strategy performance looks good - consider live testing with small position sizes")
    
    return recommendations

def _assess_risk(results) -> str:
    """Assess overall risk level"""
    risk_score = 0
    
    if results.max_drawdown_percentage > 30:
        risk_score += 3
    elif results.max_drawdown_percentage > 20:
        risk_score += 2
    elif results.max_drawdown_percentage > 10:
        risk_score += 1
    
    if results.volatility > 0.5:
        risk_score += 2
    elif results.volatility > 0.3:
        risk_score += 1
    
    if results.sharpe_ratio < 0.5:
        risk_score += 2
    elif results.sharpe_ratio < 1.0:
        risk_score += 1
    
    if risk_score >= 5:
        return "HIGH"
    elif risk_score >= 3:
        return "MEDIUM"
    else:
        return "LOW"