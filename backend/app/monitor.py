"""
Enhanced Monitoring and Logging System with Prometheus Integration
Tracks trading performance, errors, and system health
"""
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Import monitoring service for Prometheus integration
try:
    from app.services.monitoring_service import monitoring_service
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    monitoring_service = None


@dataclass
class TradeMetrics:
    """Trading performance metrics"""
    symbol: str
    timestamp: datetime
    action: str  # BUY, SELL, HOLD
    price: float
    quantity: float
    confidence: float
    profit_loss: Optional[float] = None
    balance_before: float = 0.0
    balance_after: float = 0.0


@dataclass
class SystemHealth:
    """System health metrics"""
    timestamp: datetime
    api_status: bool
    model_status: bool
    trading_status: bool
    memory_usage: float
    cpu_usage: float
    error_count: int = 0


class TradingMonitor:
    """Enhanced monitors trading activities and system performance with Prometheus integration"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup loggers
        self.setup_loggers()
        
        # Metrics storage
        self.trade_history: List[TradeMetrics] = []
        self.system_health_history: List[SystemHealth] = []
        self.error_log: List[Dict] = []
        
    def setup_loggers(self):
        """Setup structured logging"""
        
        # Trading logger
        self.trade_logger = logging.getLogger("trading")
        trade_handler = logging.FileHandler(self.log_dir / "trades.log")
        trade_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.trade_logger.addHandler(trade_handler)
        self.trade_logger.setLevel(logging.INFO)
        
        # System logger
        self.system_logger = logging.getLogger("system")
        system_handler = logging.FileHandler(self.log_dir / "system.log")
        system_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        self.system_logger.addHandler(system_handler)
        self.system_logger.setLevel(logging.INFO)
        
        # Error logger
        self.error_logger = logging.getLogger("errors")
        error_handler = logging.FileHandler(self.log_dir / "errors.log")
        error_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(exc_info)s')
        )
        self.error_logger.addHandler(error_handler)
        self.error_logger.setLevel(logging.ERROR)
    
    def log_trade(self, trade_metrics: TradeMetrics):
        """Log a trade execution with Prometheus integration"""
        self.trade_history.append(trade_metrics)
        
        # Log to file
        trade_data = {
            "timestamp": trade_metrics.timestamp.isoformat(),
            "symbol": trade_metrics.symbol,
            "action": trade_metrics.action,
            "price": trade_metrics.price,
            "quantity": trade_metrics.quantity,
            "confidence": trade_metrics.confidence,
            "profit_loss": trade_metrics.profit_loss,
            "balance_before": trade_metrics.balance_before,
            "balance_after": trade_metrics.balance_after
        }
        
        self.trade_logger.info(f"TRADE_EXECUTED: {json.dumps(trade_data)}")
        
        # Prometheus metrics integration
        if PROMETHEUS_AVAILABLE and monitoring_service:
            try:
                # Determine if trade was profitable
                is_win = trade_metrics.profit_loss and trade_metrics.profit_loss > 0
                pnl = trade_metrics.profit_loss or 0.0
                volume = trade_metrics.price * trade_metrics.quantity
                
                # Record trade metrics
                monitoring_service.record_trade(
                    is_win=is_win,
                    pnl=pnl,
                    volume=volume
                )
                
                # Update account balance
                monitoring_service.update_account_balance(trade_metrics.balance_after)
                
            except Exception as e:
                self.error_logger.error(f"Error updating Prometheus metrics: {e}")
        
        # Keep only last 1000 trades in memory
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-1000:]
    
    def log_system_health(self, health: SystemHealth):
        """Log system health metrics with Prometheus integration"""
        self.system_health_history.append(health)
        
        health_data = asdict(health)
        health_data['timestamp'] = health.timestamp.isoformat()
        
        self.system_logger.info(f"HEALTH_CHECK: {json.dumps(health_data)}")
        
        # Prometheus metrics integration
        if PROMETHEUS_AVAILABLE and monitoring_service:
            try:
                # Update system metrics (these are handled automatically by monitoring service)
                # We can add any additional custom metrics here if needed
                pass
            except Exception as e:
                self.error_logger.error(f"Error updating system health metrics: {e}")
        
        # Keep only last 100 health checks in memory
        if len(self.system_health_history) > 100:
            self.system_health_history = self.system_health_history[-100:]
    
    def log_error(self, error_type: str, message: str, details: Optional[Dict] = None):
        """Log system errors with Prometheus integration"""
        error_data = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": message,
            "details": details or {}
        }
        
        self.error_log.append(error_data)
        self.error_logger.error(f"ERROR: {json.dumps(error_data)}")
        
        # Prometheus metrics integration
        if PROMETHEUS_AVAILABLE and monitoring_service:
            try:
                monitoring_service.record_api_error(error_type)
            except Exception as e:
                self.error_logger.error(f"Error recording error metric: {e}")
        
        # Keep only last 500 errors in memory
        if len(self.error_log) > 500:
            self.error_log = self.error_log[-500:]
    
    def log_regime_detection(self, regime_type: str, confidence: float, metrics: Dict[str, Any] = None):
        """Log regime detection with Prometheus integration"""
        detection_data = {
            "timestamp": datetime.now().isoformat(),
            "regime_type": regime_type,
            "confidence": confidence,
            "metrics": metrics or {}
        }
        
        self.system_logger.info(f"REGIME_DETECTION: {json.dumps(detection_data)}")
        
        # Prometheus metrics integration
        if PROMETHEUS_AVAILABLE and monitoring_service:
            try:
                monitoring_service.record_regime_detection(regime_type, confidence)
            except Exception as e:
                self.error_logger.error(f"Error recording regime detection metric: {e}")
    
    def log_regime_switch(self, from_regime: str, to_regime: str):
        """Log regime switch with Prometheus integration"""
        switch_data = {
            "timestamp": datetime.now().isoformat(),
            "from_regime": from_regime,
            "to_regime": to_regime
        }
        
        self.system_logger.info(f"REGIME_SWITCH: {json.dumps(switch_data)}")
        
        # Prometheus metrics integration
        if PROMETHEUS_AVAILABLE and monitoring_service:
            try:
                monitoring_service.record_regime_switch(from_regime, to_regime)
            except Exception as e:
                self.error_logger.error(f"Error recording regime switch metric: {e}")
    
    def log_ml_prediction(self, model_type: str, regime: str, confidence: float = None):
        """Log ML model prediction with Prometheus integration"""
        prediction_data = {
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "regime": regime,
            "confidence": confidence
        }
        
        self.system_logger.info(f"ML_PREDICTION: {json.dumps(prediction_data)}")
        
        # Prometheus metrics integration
        if PROMETHEUS_AVAILABLE and monitoring_service:
            try:
                monitoring_service.record_ml_prediction(model_type, regime)
            except Exception as e:
                self.error_logger.error(f"Error recording ML prediction metric: {e}")
    
    def log_model_training(self, model_type: str, duration: float, accuracy: float = None):
        """Log ML model training with Prometheus integration"""
        training_data = {
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "duration": duration,
            "accuracy": accuracy
        }
        
        self.system_logger.info(f"MODEL_TRAINING: {json.dumps(training_data)}")
        
        # Prometheus metrics integration
        if PROMETHEUS_AVAILABLE and monitoring_service:
            try:
                monitoring_service.record_model_training(model_type, duration)
                if accuracy is not None:
                    monitoring_service.update_model_accuracy(model_type, accuracy)
            except Exception as e:
                self.error_logger.error(f"Error recording model training metrics: {e}")
    
    def update_positions(self, count: int):
        """Update active positions count with Prometheus integration"""
        position_data = {
            "timestamp": datetime.now().isoformat(),
            "active_positions": count
        }
        
        self.system_logger.info(f"POSITION_UPDATE: {json.dumps(position_data)}")
        
        # Prometheus metrics integration
        if PROMETHEUS_AVAILABLE and monitoring_service:
            try:
                monitoring_service.update_positions(count)
            except Exception as e:
                self.error_logger.error(f"Error updating positions metric: {e}")
    
    def update_unrealized_pnl(self, unrealized_pnl: float):
        """Update unrealized PnL with Prometheus integration"""
        pnl_data = {
            "timestamp": datetime.now().isoformat(),
            "unrealized_pnl": unrealized_pnl
        }
        
        self.system_logger.info(f"UNREALIZED_PNL: {json.dumps(pnl_data)}")
        
        # Prometheus metrics integration
        if PROMETHEUS_AVAILABLE and monitoring_service:
            try:
                monitoring_service.update_unrealized_pnl(unrealized_pnl)
            except Exception as e:
                self.error_logger.error(f"Error updating unrealized PnL metric: {e}")
    
    def update_risk_metrics(self, sharpe_ratio: float, max_drawdown: float):
        """Update risk metrics with Prometheus integration"""
        risk_data = {
            "timestamp": datetime.now().isoformat(),
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        }
        
        self.system_logger.info(f"RISK_METRICS: {json.dumps(risk_data)}")
        
        # Prometheus metrics integration
        if PROMETHEUS_AVAILABLE and monitoring_service:
            try:
                monitoring_service.update_risk_metrics(sharpe_ratio, max_drawdown)
            except Exception as e:
                self.error_logger.error(f"Error updating risk metrics: {e}")
    
    
    def get_trading_performance(self, hours: int = 24) -> Dict[str, Any]:
        """Get trading performance metrics for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_trades = [
            trade for trade in self.trade_history 
            if trade.timestamp > cutoff_time
        ]
        
        if not recent_trades:
            return {
                "total_trades": 0,
                "total_profit_loss": 0,
                "win_rate": 0,
                "avg_profit_per_trade": 0,
                "best_trade": None,
                "worst_trade": None
            }
        
        # Calculate metrics
        profitable_trades = [t for t in recent_trades if t.profit_loss and t.profit_loss > 0]
        total_profit_loss = sum(t.profit_loss for t in recent_trades if t.profit_loss)
        
        performance = {
            "total_trades": len(recent_trades),
            "buy_trades": len([t for t in recent_trades if t.action == "BUY"]),
            "sell_trades": len([t for t in recent_trades if t.action == "SELL"]),
            "total_profit_loss": total_profit_loss,
            "win_rate": len(profitable_trades) / len(recent_trades) * 100 if recent_trades else 0,
            "avg_profit_per_trade": total_profit_loss / len(recent_trades) if recent_trades else 0,
            "avg_confidence": sum(t.confidence for t in recent_trades) / len(recent_trades),
        }
        
        # Best and worst trades
        if recent_trades:
            trades_with_pnl = [t for t in recent_trades if t.profit_loss is not None]
            if trades_with_pnl:
                best_trade = max(trades_with_pnl, key=lambda t: t.profit_loss)
                worst_trade = min(trades_with_pnl, key=lambda t: t.profit_loss)
                
                performance["best_trade"] = {
                    "symbol": best_trade.symbol,
                    "action": best_trade.action,
                    "profit_loss": best_trade.profit_loss,
                    "timestamp": best_trade.timestamp.isoformat()
                }
                
                performance["worst_trade"] = {
                    "symbol": worst_trade.symbol,
                    "action": worst_trade.action,
                    "profit_loss": worst_trade.profit_loss,
                    "timestamp": worst_trade.timestamp.isoformat()
                }
        
        return performance
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        recent_health = self.system_health_history[-10:] if self.system_health_history else []
        recent_errors = self.error_log[-10:] if self.error_log else []
        
        if not recent_health:
            return {
                "status": "unknown",
                "last_check": None,
                "avg_cpu": 0,
                "avg_memory": 0,
                "recent_errors": len(recent_errors)
            }
        
        latest_health = recent_health[-1]
        
        return {
            "status": "healthy" if latest_health.api_status and latest_health.model_status else "degraded",
            "last_check": latest_health.timestamp.isoformat(),
            "api_status": latest_health.api_status,
            "model_status": latest_health.model_status,
            "trading_status": latest_health.trading_status,
            "avg_cpu": sum(h.cpu_usage for h in recent_health) / len(recent_health),
            "avg_memory": sum(h.memory_usage for h in recent_health) / len(recent_health),
            "recent_errors": len(recent_errors),
            "total_trades_today": len([
                t for t in self.trade_history 
                if t.timestamp.date() == datetime.now().date()
            ])
        }
    
    def export_metrics(self, filepath: str):
        """Export all metrics to JSON file"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "trades": [
                {
                    "timestamp": t.timestamp.isoformat(),
                    "symbol": t.symbol,
                    "action": t.action,
                    "price": t.price,
                    "quantity": t.quantity,
                    "confidence": t.confidence,
                    "profit_loss": t.profit_loss,
                    "balance_before": t.balance_before,
                    "balance_after": t.balance_after
                }
                for t in self.trade_history
            ],
            "system_health": [
                {
                    "timestamp": h.timestamp.isoformat(),
                    "api_status": h.api_status,
                    "model_status": h.model_status,
                    "trading_status": h.trading_status,
                    "memory_usage": h.memory_usage,
                    "cpu_usage": h.cpu_usage,
                    "error_count": h.error_count
                }
                for h in self.system_health_history
            ],
            "errors": self.error_log
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)


# Global monitor instance
monitor = TradingMonitor()