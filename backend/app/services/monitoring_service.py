"""
Enhanced Monitoring Service with Prometheus Integration
Implements comprehensive observability for the trading system
"""

import time
import psutil
from typing import Dict, Any
from prometheus_client import Counter, Gauge, Histogram, Info, CollectorRegistry, generate_latest
from datetime import datetime
import logging
import json
import threading
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class TradingMetrics:
    """Trading performance metrics data class"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_volume: float = 0.0
    current_positions: int = 0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0

@dataclass
class SystemMetrics:
    """System health metrics data class"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    uptime_seconds: float = 0.0
    active_connections: int = 0
    error_rate: float = 0.0

class PrometheusMonitoringService:
    """
    Comprehensive monitoring service with Prometheus metrics
    Tracks trading performance, system health, and API usage
    """
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self.start_time = time.time()
        self._setup_metrics()
        self._system_monitor_thread = None
        self._monitoring_active = False
        
        # Internal tracking
        self._trading_metrics = TradingMetrics()
        self._system_metrics = SystemMetrics()
        self._recent_errors = []
        self._api_call_times = []
        
        logger.info("🔍 Prometheus monitoring service initialized")
    
    def _setup_metrics(self):
        """Initialize all Prometheus metrics"""
        
        # === TRADING PERFORMANCE METRICS ===
        
        # Counters (monotonically increasing)
        self.total_trades_counter = Counter(
            'trading_total_trades_total',
            'Total number of trades executed',
            registry=self.registry
        )
        
        self.winning_trades_counter = Counter(
            'trading_winning_trades_total',
            'Total number of winning trades',
            registry=self.registry
        )
        
        self.losing_trades_counter = Counter(
            'trading_losing_trades_total',
            'Total number of losing trades',
            registry=self.registry
        )
        
        self.trade_volume_counter = Counter(
            'trading_volume_total',
            'Total trading volume in USD',
            registry=self.registry
        )
        
        # Gauges (can go up and down)
        self.current_pnl_gauge = Gauge(
            'trading_current_pnl',
            'Current profit/loss in USD',
            registry=self.registry
        )
        
        self.unrealized_pnl_gauge = Gauge(
            'trading_unrealized_pnl',
            'Unrealized profit/loss in USD',
            registry=self.registry
        )
        
        self.active_positions_gauge = Gauge(
            'trading_active_positions',
            'Number of currently active positions',
            registry=self.registry
        )
        
        self.account_balance_gauge = Gauge(
            'trading_account_balance',
            'Current account balance in USD',
            registry=self.registry
        )
        
        self.win_rate_gauge = Gauge(
            'trading_win_rate',
            'Current win rate percentage',
            registry=self.registry
        )
        
        self.sharpe_ratio_gauge = Gauge(
            'trading_sharpe_ratio',
            'Current Sharpe ratio',
            registry=self.registry
        )
        
        self.max_drawdown_gauge = Gauge(
            'trading_max_drawdown',
            'Maximum drawdown percentage',
            registry=self.registry
        )
        
        # === MARKET REGIME METRICS ===
        
        self.regime_detection_counter = Counter(
            'regime_detections_total',
            'Number of regime detections by type',
            ['regime_type'],
            registry=self.registry
        )
        
        self.regime_confidence_gauge = Gauge(
            'regime_confidence',
            'Current regime detection confidence',
            ['regime_type'],
            registry=self.registry
        )
        
        self.regime_switches_counter = Counter(
            'regime_switches_total',
            'Number of regime switches',
            ['from_regime', 'to_regime'],
            registry=self.registry
        )
        
        # === SYSTEM HEALTH METRICS ===
        
        self.cpu_usage_gauge = Gauge(
            'system_cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage_gauge = Gauge(
            'system_memory_usage_percent',
            'Current memory usage percentage',
            registry=self.registry
        )
        
        self.disk_usage_gauge = Gauge(
            'system_disk_usage_percent',
            'Current disk usage percentage',
            registry=self.registry
        )
        
        self.uptime_gauge = Gauge(
            'system_uptime_seconds',
            'System uptime in seconds',
            registry=self.registry
        )
        
        # === API PERFORMANCE METRICS ===
        
        self.api_requests_counter = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.api_request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        self.api_errors_counter = Counter(
            'api_errors_total',
            'Total API errors',
            ['error_type'],
            registry=self.registry
        )
        
        # === ML MODEL METRICS ===
        
        self.model_predictions_counter = Counter(
            'ml_predictions_total',
            'Total ML model predictions',
            ['model_type', 'regime'],
            registry=self.registry
        )
        
        self.model_accuracy_gauge = Gauge(
            'ml_model_accuracy',
            'Current model accuracy',
            ['model_type'],
            registry=self.registry
        )
        
        self.model_training_duration = Histogram(
            'ml_training_duration_seconds',
            'Model training duration in seconds',
            ['model_type'],
            registry=self.registry
        )
        
        # === INFO METRICS ===
        
        self.app_info = Info(
            'app_info',
            'Application information',
            registry=self.registry
        )
        
        # Set application info
        self.app_info.info({
            'version': '1.0.0',
            'environment': 'production',
            'build_date': datetime.now().isoformat(),
            'features': 'regime_adaptation,ml_trading,risk_management'
        })
    
    def start_system_monitoring(self):
        """Start background system monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._system_monitor_thread = threading.Thread(
            target=self._system_monitor_loop,
            daemon=True
        )
        self._system_monitor_thread.start()
        logger.info("📊 System monitoring thread started")
    
    def stop_system_monitoring(self):
        """Stop background system monitoring"""
        self._monitoring_active = False
        if self._system_monitor_thread:
            self._system_monitor_thread.join(timeout=5)
        logger.info("🛑 System monitoring stopped")
    
    def _system_monitor_loop(self):
        """Background thread for system monitoring"""
        while self._monitoring_active:
            try:
                self._update_system_metrics()
                time.sleep(10)  # Update every 10 seconds
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(30)  # Wait longer on error
    
    def _update_system_metrics(self):
        """Update system health metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.cpu_usage_gauge.set(cpu_percent)
            self._system_metrics.cpu_usage = cpu_percent
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.memory_usage_gauge.set(memory_percent)
            self._system_metrics.memory_usage = memory_percent
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.disk_usage_gauge.set(disk_percent)
            self._system_metrics.disk_usage = disk_percent
            
            # Uptime
            uptime = time.time() - self.start_time
            self.uptime_gauge.set(uptime)
            self._system_metrics.uptime_seconds = uptime
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    # === TRADING METRICS METHODS ===
    
    def record_trade(self, is_win: bool, pnl: float, volume: float):
        """Record a completed trade"""
        self.total_trades_counter.inc()
        self._trading_metrics.total_trades += 1
        
        if is_win:
            self.winning_trades_counter.inc()
            self._trading_metrics.winning_trades += 1
        else:
            self.losing_trades_counter.inc()
            self._trading_metrics.losing_trades += 1
        
        # Update PnL
        self._trading_metrics.total_pnl += pnl
        self.current_pnl_gauge.set(self._trading_metrics.total_pnl)
        
        # Update volume
        self._trading_metrics.total_volume += volume
        self.trade_volume_counter.inc(volume)
        
        # Update win rate
        if self._trading_metrics.total_trades > 0:
            win_rate = (self._trading_metrics.winning_trades / self._trading_metrics.total_trades) * 100
            self._trading_metrics.win_rate = win_rate
            self.win_rate_gauge.set(win_rate)
        
        logger.info(f"📈 Trade recorded: {'WIN' if is_win else 'LOSS'}, PnL: ${pnl:.2f}, Volume: ${volume:.2f}")
    
    def update_positions(self, count: int):
        """Update active positions count"""
        self._trading_metrics.current_positions = count
        self.active_positions_gauge.set(count)
    
    def update_unrealized_pnl(self, unrealized_pnl: float):
        """Update unrealized PnL"""
        self._trading_metrics.unrealized_pnl = unrealized_pnl
        self.unrealized_pnl_gauge.set(unrealized_pnl)
    
    def update_account_balance(self, balance: float):
        """Update account balance"""
        self.account_balance_gauge.set(balance)
    
    def update_risk_metrics(self, sharpe_ratio: float, max_drawdown: float):
        """Update risk management metrics"""
        self._trading_metrics.sharpe_ratio = sharpe_ratio
        self._trading_metrics.max_drawdown = max_drawdown
        
        self.sharpe_ratio_gauge.set(sharpe_ratio)
        self.max_drawdown_gauge.set(max_drawdown)
    
    # === REGIME DETECTION METHODS ===
    
    def record_regime_detection(self, regime_type: str, confidence: float):
        """Record a regime detection"""
        self.regime_detection_counter.labels(regime_type=regime_type).inc()
        self.regime_confidence_gauge.labels(regime_type=regime_type).set(confidence)
        
        logger.debug(f"🎯 Regime detected: {regime_type} (confidence: {confidence:.2f})")
    
    def record_regime_switch(self, from_regime: str, to_regime: str):
        """Record a regime switch"""
        self.regime_switches_counter.labels(
            from_regime=from_regime,
            to_regime=to_regime
        ).inc()
        
        logger.info(f"🔄 Regime switch: {from_regime} → {to_regime}")
    
    # === ML MODEL METHODS ===
    
    def record_ml_prediction(self, model_type: str, regime: str):
        """Record an ML model prediction"""
        self.model_predictions_counter.labels(
            model_type=model_type,
            regime=regime
        ).inc()
    
    def update_model_accuracy(self, model_type: str, accuracy: float):
        """Update model accuracy metric"""
        self.model_accuracy_gauge.labels(model_type=model_type).set(accuracy)
    
    def record_model_training(self, model_type: str, duration: float):
        """Record model training duration"""
        self.model_training_duration.labels(model_type=model_type).observe(duration)
    
    # === API MONITORING METHODS ===
    
    def record_api_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record an API request"""
        status = f"{status_code // 100}xx"
        
        self.api_requests_counter.labels(
            method=method,
            endpoint=endpoint,
            status=status
        ).inc()
        
        self.api_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_api_error(self, error_type: str):
        """Record an API error"""
        self.api_errors_counter.labels(error_type=error_type).inc()
        
        # Track recent errors for error rate calculation
        self._recent_errors.append(time.time())
        # Keep only errors from last hour
        cutoff = time.time() - 3600
        self._recent_errors = [t for t in self._recent_errors if t > cutoff]
    
    # === DATA EXPORT METHODS ===
    
    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        return {
            'trading_metrics': asdict(self._trading_metrics),
            'system_metrics': asdict(self._system_metrics),
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': time.time() - self.start_time,
            'recent_error_count': len(self._recent_errors)
        }
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.registry).decode('utf-8')
    
    def export_metrics_json(self, filepath: str):
        """Export metrics to JSON file"""
        metrics = self.get_metrics_snapshot()
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"📊 Metrics exported to {filepath}")
    
    # === HEALTH CHECK METHODS ===
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        error_rate = len(self._recent_errors) / 3600 if self._recent_errors else 0
        
        health_score = 100
        issues = []
        
        # Check system resources
        if self._system_metrics.cpu_usage > 80:
            health_score -= 20
            issues.append("High CPU usage")
        
        if self._system_metrics.memory_usage > 85:
            health_score -= 20
            issues.append("High memory usage")
        
        if error_rate > 10:  # More than 10 errors per hour
            health_score -= 30
            issues.append("High error rate")
        
        status = "healthy" if health_score >= 80 else ("warning" if health_score >= 60 else "critical")
        
        return {
            'status': status,
            'health_score': health_score,
            'issues': issues,
            'system_metrics': asdict(self._system_metrics),
            'error_rate_per_hour': error_rate,
            'uptime_hours': (time.time() - self.start_time) / 3600
        }
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_system_monitoring()

# Global monitoring service instance
monitoring_service = PrometheusMonitoringService()