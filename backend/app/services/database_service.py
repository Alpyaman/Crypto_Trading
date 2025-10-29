"""
Database service for managing database operations
Handles CRUD operations for trades, training sessions, and metrics
"""
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import uuid
import json

from app.models.database import (
    Trade, ModelTraining, TrainingMetrics, AccountBalance, 
    SystemMetrics, TradingSession, get_db, init_database
)

class DatabaseService:
    """Service class for database operations"""
    
    def __init__(self):
        # Initialize database on service creation
        init_database()
        
    def get_session(self) -> Session:
        """Get database session"""
        return next(get_db())
    
    # Trade Operations
    def create_trade(self, trade_data: Dict[str, Any]) -> Trade:
        """Create a new trade record"""
        session = self.get_session()
        try:
            trade = Trade(
                symbol=trade_data.get('symbol'),
                side=trade_data.get('side'),
                quantity=trade_data.get('quantity'),
                price=trade_data.get('price'),
                pnl=trade_data.get('pnl', 0.0),
                commission=trade_data.get('commission', 0.0),
                order_id=trade_data.get('order_id'),
                status=trade_data.get('status', 'FILLED'),
                trading_mode=trade_data.get('trading_mode'),
                position_size=trade_data.get('position_size'),
                model_training_id=trade_data.get('model_training_id')
            )
            session.add(trade)
            session.commit()
            session.refresh(trade)
            return trade
        finally:
            session.close()
    
    def get_trades(self, 
                   symbol: Optional[str] = None, 
                   limit: int = 100,
                   days_back: int = 30) -> List[Trade]:
        """Get trades with optional filtering"""
        session = self.get_session()
        try:
            query = session.query(Trade)
            
            # Filter by symbol if provided
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            
            # Filter by date range
            start_date = datetime.utcnow() - timedelta(days=days_back)
            query = query.filter(Trade.timestamp >= start_date)
            
            # Order by timestamp desc and limit
            trades = query.order_by(desc(Trade.timestamp)).limit(limit).all()
            return trades
        finally:
            session.close()
    
    def get_trade_statistics(self, symbol: Optional[str] = None, days_back: int = 30) -> Dict[str, Any]:
        """Get trading statistics"""
        session = self.get_session()
        try:
            query = session.query(Trade)
            
            if symbol:
                query = query.filter(Trade.symbol == symbol)
            
            start_date = datetime.utcnow() - timedelta(days=days_back)
            query = query.filter(Trade.timestamp >= start_date)
            
            trades = query.all()
            
            if not trades:
                return {
                    'total_trades': 0,
                    'total_pnl': 0.0,
                    'win_rate': 0.0,
                    'avg_profit': 0.0,
                    'max_profit': 0.0,
                    'max_loss': 0.0
                }
            
            total_trades = len(trades)
            total_pnl = sum(trade.pnl for trade in trades)
            winning_trades = [trade for trade in trades if trade.pnl > 0]
            losing_trades = [trade for trade in trades if trade.pnl < 0]
            
            win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
            avg_profit = total_pnl / total_trades if total_trades > 0 else 0
            max_profit = max(trade.pnl for trade in trades) if trades else 0
            max_loss = min(trade.pnl for trade in trades) if trades else 0
            
            return {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'total_pnl': round(total_pnl, 4),
                'win_rate': round(win_rate, 2),
                'avg_profit': round(avg_profit, 4),
                'max_profit': round(max_profit, 4),
                'max_loss': round(max_loss, 4)
            }
        finally:
            session.close()
    
    # Training Operations
    def create_training_session(self, training_data: Dict[str, Any]) -> ModelTraining:
        """Create a new training session"""
        session = self.get_session()
        try:
            training = ModelTraining(
                session_id=training_data.get('session_id', str(uuid.uuid4())),
                algorithm=training_data.get('algorithm'),
                symbol=training_data.get('symbol'),
                timesteps=training_data.get('timesteps'),
                learning_rate=training_data.get('learning_rate', 0.0003),
                batch_size=training_data.get('batch_size', 64),
                n_steps=training_data.get('n_steps', 2048),
                features_count=training_data.get('features_count'),
                config=json.dumps(training_data.get('config', {})),
                status='RUNNING'
            )
            session.add(training)
            session.commit()
            session.refresh(training)
            return training
        finally:
            session.close()
    
    def update_training_session(self, session_id: str, updates: Dict[str, Any]) -> Optional[ModelTraining]:
        """Update training session"""
        session = self.get_session()
        try:
            training = session.query(ModelTraining).filter(
                ModelTraining.session_id == session_id
            ).first()
            
            if training:
                for key, value in updates.items():
                    if hasattr(training, key):
                        setattr(training, key, value)
                
                # Update completion timestamp if status is completed
                if updates.get('status') == 'COMPLETED':
                    training.completed_at = datetime.utcnow()
                    if training.started_at:
                        training.training_time = (training.completed_at - training.started_at).total_seconds()
                
                session.commit()
                session.refresh(training)
                return training
            return None
        finally:
            session.close()
    
    def get_training_sessions(self, 
                             symbol: Optional[str] = None,
                             algorithm: Optional[str] = None,
                             limit: int = 50) -> List[ModelTraining]:
        """Get training sessions with optional filtering"""
        session = self.get_session()
        try:
            query = session.query(ModelTraining)
            
            if symbol:
                query = query.filter(ModelTraining.symbol == symbol)
            if algorithm:
                query = query.filter(ModelTraining.algorithm == algorithm)
            
            sessions = query.order_by(desc(ModelTraining.created_at)).limit(limit).all()
            return sessions
        finally:
            session.close()
    
    def get_active_training_sessions(self) -> List[ModelTraining]:
        """Get currently active training sessions"""
        session = self.get_session()
        try:
            sessions = session.query(ModelTraining).filter(
                ModelTraining.status == 'RUNNING'
            ).all()
            return sessions
        finally:
            session.close()
    
    # Training Metrics Operations
    def add_training_metric(self, metric_data: Dict[str, Any]) -> TrainingMetrics:
        """Add training metric point"""
        session = self.get_session()
        try:
            metric = TrainingMetrics(
                training_session_id=metric_data.get('training_session_id'),
                episode=metric_data.get('episode'),
                reward=metric_data.get('reward'),
                loss=metric_data.get('loss'),
                entropy=metric_data.get('entropy'),
                value_loss=metric_data.get('value_loss'),
                policy_loss=metric_data.get('policy_loss'),
                learning_rate=metric_data.get('learning_rate'),
                explained_variance=metric_data.get('explained_variance'),
                fps=metric_data.get('fps')
            )
            session.add(metric)
            session.commit()
            session.refresh(metric)
            return metric
        finally:
            session.close()
    
    def get_training_metrics(self, training_session_id: int, limit: int = 1000) -> List[TrainingMetrics]:
        """Get training metrics for a session"""
        session = self.get_session()
        try:
            metrics = session.query(TrainingMetrics).filter(
                TrainingMetrics.training_session_id == training_session_id
            ).order_by(TrainingMetrics.episode).limit(limit).all()
            return metrics
        finally:
            session.close()
    
    # Account Balance Operations
    def save_account_balance(self, balance_data: Dict[str, Any]) -> AccountBalance:
        """Save account balance snapshot"""
        session = self.get_session()
        try:
            balance = AccountBalance(
                account_type=balance_data.get('account_type', 'futures'),
                total_balance=balance_data.get('total_balance'),
                available_balance=balance_data.get('available_balance'),
                locked_balance=balance_data.get('locked_balance', 0.0),
                unrealized_pnl=balance_data.get('unrealized_pnl', 0.0),
                total_wallet_balance=balance_data.get('total_wallet_balance'),
                assets=json.dumps(balance_data.get('assets', []))
            )
            session.add(balance)
            session.commit()
            session.refresh(balance)
            return balance
        finally:
            session.close()
    
    def get_latest_balance(self, account_type: str = 'futures') -> Optional[AccountBalance]:
        """Get latest account balance"""
        session = self.get_session()
        try:
            balance = session.query(AccountBalance).filter(
                AccountBalance.account_type == account_type
            ).order_by(desc(AccountBalance.timestamp)).first()
            return balance
        finally:
            session.close()
    
    # System Metrics Operations
    def add_system_metric(self, metric_data: Dict[str, Any]) -> SystemMetrics:
        """Add system metric"""
        session = self.get_session()
        try:
            metric = SystemMetrics(
                metric_type=metric_data.get('metric_type'),
                metric_name=metric_data.get('metric_name'),
                value=metric_data.get('value'),
                unit=metric_data.get('unit'),
                endpoint=metric_data.get('endpoint'),
                error_code=metric_data.get('error_code'),
                details=json.dumps(metric_data.get('details')) if metric_data.get('details') else None
            )
            session.add(metric)
            session.commit()
            session.refresh(metric)
            return metric
        finally:
            session.close()
    
    # Trading Session Operations
    def create_trading_session(self, session_data: Dict[str, Any]) -> TradingSession:
        """Create new trading session"""
        session = self.get_session()
        try:
            trading_session = TradingSession(
                session_id=session_data.get('session_id', str(uuid.uuid4())),
                symbol=session_data.get('symbol'),
                trading_mode=session_data.get('trading_mode'),
                position_size=session_data.get('position_size'),
                stop_loss=session_data.get('stop_loss'),
                take_profit=session_data.get('take_profit'),
                model_training_id=session_data.get('model_training_id'),
                model_version=session_data.get('model_version')
            )
            session.add(trading_session)
            session.commit()
            session.refresh(trading_session)
            return trading_session
        finally:
            session.close()
    
    def update_trading_session(self, session_id: str, updates: Dict[str, Any]) -> Optional[TradingSession]:
        """Update trading session"""
        session = self.get_session()
        try:
            trading_session = session.query(TradingSession).filter(
                TradingSession.session_id == session_id
            ).first()
            
            if trading_session:
                for key, value in updates.items():
                    if hasattr(trading_session, key):
                        setattr(trading_session, key, value)
                
                # Calculate duration if ending session
                if updates.get('status') in ['STOPPED', 'COMPLETED'] and not trading_session.ended_at:
                    trading_session.ended_at = datetime.utcnow()
                    if trading_session.started_at:
                        trading_session.duration = (trading_session.ended_at - trading_session.started_at).total_seconds()
                
                session.commit()
                session.refresh(trading_session)
                return trading_session
            return None
        finally:
            session.close()
    
    # Analytics and Reporting
    def get_performance_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """Get overall performance summary"""
        session = self.get_session()
        try:
            start_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Trading performance
            trade_stats = self.get_trade_statistics(days_back=days_back)
            
            # Training sessions
            training_count = session.query(ModelTraining).filter(
                ModelTraining.created_at >= start_date
            ).count()
            
            completed_trainings = session.query(ModelTraining).filter(
                and_(
                    ModelTraining.created_at >= start_date,
                    ModelTraining.status == 'COMPLETED'
                )
            ).count()
            
            # System metrics
            error_count = session.query(SystemMetrics).filter(
                and_(
                    SystemMetrics.timestamp >= start_date,
                    SystemMetrics.metric_type == 'error_rate'
                )
            ).count()
            
            return {
                'period_days': days_back,
                'trading': trade_stats,
                'training': {
                    'total_sessions': training_count,
                    'completed_sessions': completed_trainings,
                    'success_rate': round(completed_trainings / training_count * 100, 2) if training_count > 0 else 0
                },
                'system': {
                    'error_count': error_count,
                },
                'generated_at': datetime.utcnow().isoformat()
            }
        finally:
            session.close()

# Global database service instance
db_service = DatabaseService()