"""
Advanced Backtesting Service
Provides comprehensive historical trading simulation and performance analytics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class BacktestTrade:
    """Individual backtest trade record"""
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    commission: float
    regime: str
    confidence: float
    model_used: str

@dataclass
class BacktestResults:
    """Comprehensive backtest performance results"""
    # Basic metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # P&L metrics
    total_pnl: float
    total_pnl_percentage: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    
    # Risk metrics
    max_drawdown: float
    max_drawdown_percentage: float
    calmar_ratio: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Performance metrics
    total_return: float
    annualized_return: float
    volatility: float
    best_trade: float
    worst_trade: float
    avg_trade: float
    
    # Trade analysis
    avg_winning_trade: float
    avg_losing_trade: float
    largest_win_streak: int
    largest_loss_streak: int
    
    # Time metrics
    start_date: str
    end_date: str
    duration_days: int
    
    # Additional metrics
    trades_per_day: float
    commission_paid: float
    regime_performance: Dict[str, Dict[str, float]]
    
    # Trade history
    trades: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]

class BacktestingService:
    """
    Advanced backtesting service for trading strategy evaluation
    """
    
    def __init__(self, binance_service, enhanced_ml_service):
        self.binance_service = binance_service
        self.enhanced_ml_service = enhanced_ml_service
        self.commission_rate = 0.001  # 0.1% commission
        self.initial_balance = 10000.0  # $10K starting balance
        
    async def run_backtest(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        model_type: str = "enhanced",
        initial_balance: float = 10000.0,
        commission_rate: float = 0.001,
        position_size_mode: str = "balanced"
    ) -> BacktestResults:
        """
        Run comprehensive backtest simulation
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            model_type: 'enhanced' or 'basic'
            initial_balance: Starting capital
            commission_rate: Commission percentage (0.001 = 0.1%)
            position_size_mode: 'conservative', 'balanced', or 'aggressive'
        """
        logger.info(f"Starting backtest for {symbol} from {start_date} to {end_date}")
        
        # Setup backtest parameters
        self.initial_balance = initial_balance
        self.commission_rate = commission_rate
        
        try:
            # Get historical data
            historical_data = await self._get_historical_data(symbol, start_date, end_date)
            if historical_data.empty:
                raise ValueError("No historical data available for the specified period")
            
            # Run simulation
            trades, equity_curve = await self._simulate_trading(
                historical_data, symbol, model_type, position_size_mode
            )
            
            # Calculate performance metrics
            results = self._calculate_performance_metrics(
                trades, equity_curve, start_date, end_date
            )
            
            logger.info(f"Backtest completed: {results.total_trades} trades, "
                       f"{results.win_rate:.2f}% win rate, {results.total_pnl:.2f} total P&L")
            
            return results
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    async def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical OHLCV data for backtesting"""
        try:
            # Convert dates to timestamps
            start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
            end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
            
            # Get historical klines (1-hour intervals for detailed simulation)
            klines = self.binance_service.client.get_historical_klines(
                symbol, "1h", start_ts, end_ts
            )
            
            if not klines:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            logger.info(f"Retrieved {len(df)} data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return pd.DataFrame()
    
    async def _simulate_trading(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        model_type: str, 
        position_size_mode: str
    ) -> Tuple[List[BacktestTrade], List[Dict[str, Any]]]:
        """Simulate trading over historical data"""
        
        trades = []
        equity_curve = []
        balance = self.initial_balance
        position = 0.0  # Current position size
        position_value = 0.0  # Value of current position
        
        # Position sizing based on mode
        position_sizes = {
            'conservative': 0.1,  # 10% of balance per trade
            'balanced': 0.25,     # 25% of balance per trade
            'aggressive': 0.5     # 50% of balance per trade
        }
        max_position_size = position_sizes.get(position_size_mode, 0.25)
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            try:
                current_price = row['close']
                
                # Get market data for ML prediction
                lookback_data = data.iloc[max(0, i-50):i+1]  # Last 50 periods
                if len(lookback_data) < 20:  # Need minimum data for analysis
                    continue
                
                # Detect market regime
                regime_info = self._detect_regime_for_backtest(lookback_data)
                regime = regime_info['regime']
                confidence = regime_info['confidence']
                
                # Get trading signal from ML model
                signal = await self._get_trading_signal(
                    lookback_data, symbol, model_type, regime
                )
                
                # Execute trades based on signal
                if signal['action'] == 'BUY' and position <= 0:
                    # Close short position if any, then go long
                    if position < 0:
                        # Close short
                        trade_quantity = abs(position)
                        commission = trade_quantity * current_price * self.commission_rate
                        pnl = position_value - (trade_quantity * current_price)
                        balance += pnl - commission
                        
                        trades.append(BacktestTrade(
                            timestamp=timestamp,
                            symbol=symbol,
                            side='BUY_TO_COVER',
                            quantity=trade_quantity,
                            price=current_price,
                            commission=commission,
                            regime=regime,
                            confidence=confidence,
                            model_used=model_type
                        ))
                        
                        position = 0.0
                        position_value = 0.0
                    
                    # Open long position
                    trade_value = balance * max_position_size
                    trade_quantity = trade_value / current_price
                    commission = trade_value * self.commission_rate
                    
                    if trade_value > commission:
                        position = trade_quantity
                        position_value = trade_value
                        balance -= trade_value + commission
                        
                        trades.append(BacktestTrade(
                            timestamp=timestamp,
                            symbol=symbol,
                            side='BUY',
                            quantity=trade_quantity,
                            price=current_price,
                            commission=commission,
                            regime=regime,
                            confidence=confidence,
                            model_used=model_type
                        ))
                
                elif signal['action'] == 'SELL' and position >= 0:
                    # Close long position if any, then go short
                    if position > 0:
                        # Close long
                        trade_quantity = position
                        trade_value = trade_quantity * current_price
                        commission = trade_value * self.commission_rate
                        balance += trade_value - commission
                        
                        trades.append(BacktestTrade(
                            timestamp=timestamp,
                            symbol=symbol,
                            side='SELL',
                            quantity=trade_quantity,
                            price=current_price,
                            commission=commission,
                            regime=regime,
                            confidence=confidence,
                            model_used=model_type
                        ))
                        
                        position = 0.0
                        position_value = 0.0
                    
                    # Open short position
                    trade_value = balance * max_position_size
                    trade_quantity = trade_value / current_price
                    commission = trade_value * self.commission_rate
                    
                    if trade_value > commission:
                        position = -trade_quantity
                        position_value = trade_value
                        balance += trade_value - commission
                        
                        trades.append(BacktestTrade(
                            timestamp=timestamp,
                            symbol=symbol,
                            side='SELL_SHORT',
                            quantity=trade_quantity,
                            price=current_price,
                            commission=commission,
                            regime=regime,
                            confidence=confidence,
                            model_used=model_type
                        ))
                
                # Calculate current equity
                current_equity = balance
                if position != 0:
                    if position > 0:  # Long position
                        current_equity += position * current_price
                    else:  # Short position
                        current_equity += position_value - (abs(position) * current_price)
                
                equity_curve.append({
                    'timestamp': timestamp.isoformat(),
                    'equity': current_equity,
                    'balance': balance,
                    'position_value': abs(position) * current_price if position != 0 else 0,
                    'price': current_price,
                    'regime': regime
                })
                
            except Exception as e:
                logger.warning(f"Error processing data point at {timestamp}: {e}")
                continue
        
        # Close any remaining positions
        if position != 0 and len(data) > 0:
            final_price = data.iloc[-1]['close']
            if position > 0:
                # Close long
                trade_value = position * final_price
                commission = trade_value * self.commission_rate
                balance += trade_value - commission
            else:
                # Close short
                trade_quantity = abs(position)
                commission = trade_quantity * final_price * self.commission_rate
                pnl = position_value - (trade_quantity * final_price)
                balance += pnl - commission
            
            trades.append(BacktestTrade(
                timestamp=data.index[-1],
                symbol=symbol,
                side='CLOSE',
                quantity=abs(position),
                price=final_price,
                commission=commission,
                regime='unknown',
                confidence=0.0,
                model_used=model_type
            ))
        
        return trades, equity_curve
    
    def _detect_regime_for_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect market regime for backtesting"""
        try:
            if len(data) < 14:
                return {'regime': 'ranging', 'confidence': 0.5}
            
            # Calculate technical indicators for regime detection
            
            # Price momentum
            returns = data['close'].pct_change().dropna()
            volatility = returns.std()
            
            # Moving average trend
            ma_short = data['close'].rolling(10).mean().iloc[-1]
            ma_long = data['close'].rolling(20).mean().iloc[-1]
            current_price = data['close'].iloc[-1]
            
            # Regime classification
            if volatility > 0.04:  # High volatility threshold
                regime = 'volatile'
                confidence = min(volatility * 10, 1.0)
            elif abs(ma_short - ma_long) / current_price > 0.02:  # Strong trend
                regime = 'trending'
                confidence = min(abs(ma_short - ma_long) / current_price * 20, 1.0)
            else:
                regime = 'ranging'
                confidence = 0.7
            
            return {
                'regime': regime,
                'confidence': confidence,
                'volatility': volatility,
                'trend_strength': abs(ma_short - ma_long) / current_price
            }
            
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")
            return {'regime': 'ranging', 'confidence': 0.5}
    
    async def _get_trading_signal(
        self, 
        data: pd.DataFrame, 
        symbol: str, 
        model_type: str, 
        regime: str
    ) -> Dict[str, Any]:
        """Get trading signal from ML model"""
        try:
            # Simplified signal generation for backtesting
            # In real implementation, this would use the actual ML models
            
            closes = data['close'].values
            if len(closes) < 5:
                return {'action': 'HOLD', 'confidence': 0.0}
            
            # Simple momentum + mean reversion strategy based on regime
            short_ma = np.mean(closes[-5:])
            long_ma = np.mean(closes[-20:]) if len(closes) >= 20 else np.mean(closes)
            current_price = closes[-1]
            
            # Price position relative to moving averages
            price_vs_short = (current_price - short_ma) / short_ma
            price_vs_long = (current_price - long_ma) / long_ma
            
            # RSI-like calculation
            price_changes = np.diff(closes[-14:]) if len(closes) >= 14 else np.diff(closes)
            gains = np.where(price_changes > 0, price_changes, 0)
            losses = np.where(price_changes < 0, -price_changes, 0)
            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # Signal generation based on regime
            if regime == 'trending':
                # Trend following
                if price_vs_long > 0.02 and rsi < 80:
                    return {'action': 'BUY', 'confidence': 0.8}
                elif price_vs_long < -0.02 and rsi > 20:
                    return {'action': 'SELL', 'confidence': 0.8}
            
            elif regime == 'ranging':
                # Mean reversion
                if rsi < 30 and price_vs_short < -0.01:
                    return {'action': 'BUY', 'confidence': 0.7}
                elif rsi > 70 and price_vs_short > 0.01:
                    return {'action': 'SELL', 'confidence': 0.7}
            
            elif regime == 'volatile':
                # Reduced position sizing, momentum-based
                if price_vs_short > 0.03 and rsi < 70:
                    return {'action': 'BUY', 'confidence': 0.6}
                elif price_vs_short < -0.03 and rsi > 30:
                    return {'action': 'SELL', 'confidence': 0.6}
            
            return {'action': 'HOLD', 'confidence': 0.5}
            
        except Exception as e:
            logger.warning(f"Signal generation failed: {e}")
            return {'action': 'HOLD', 'confidence': 0.0}
    
    def _calculate_performance_metrics(
        self, 
        trades: List[BacktestTrade], 
        equity_curve: List[Dict[str, Any]], 
        start_date: str, 
        end_date: str
    ) -> BacktestResults:
        """Calculate comprehensive performance metrics"""
        
        if not trades or not equity_curve:
            return self._empty_results(start_date, end_date)
        
        # Convert trades to DataFrame for analysis
        trade_data = []
        for trade in trades:
            trade_data.append(asdict(trade))
        
        trades_df = pd.DataFrame(trade_data)
        
        # Calculate trade P&L
        trade_pnls = []
        position_tracker = {}
        
        for _, trade in trades_df.iterrows():
            symbol = trade['symbol']
            side = trade['side']
            quantity = trade['quantity']
            price = trade['price']
            commission = trade['commission']
            
            if symbol not in position_tracker:
                position_tracker[symbol] = {'quantity': 0, 'avg_price': 0}
            
            if side in ['BUY', 'SELL_SHORT']:
                # Opening position
                if side == 'BUY':
                    position_tracker[symbol]['quantity'] += quantity
                else:  # SELL_SHORT
                    position_tracker[symbol]['quantity'] -= quantity
                position_tracker[symbol]['avg_price'] = price
            
            else:  # Closing position
                if position_tracker[symbol]['quantity'] != 0:
                    if side == 'SELL':  # Closing long
                        pnl = (price - position_tracker[symbol]['avg_price']) * quantity - commission
                    else:  # BUY_TO_COVER - closing short
                        pnl = (position_tracker[symbol]['avg_price'] - price) * quantity - commission
                    
                    trade_pnls.append(pnl)
                    position_tracker[symbol]['quantity'] -= quantity if side == 'SELL' else -quantity
        
        # Basic metrics
        total_trades = len(trade_pnls)
        winning_trades = len([p for p in trade_pnls if p > 0])
        losing_trades = len([p for p in trade_pnls if p < 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(trade_pnls)
        total_pnl_percentage = (total_pnl / self.initial_balance * 100) if self.initial_balance > 0 else 0
        gross_profit = sum([p for p in trade_pnls if p > 0])
        gross_loss = abs(sum([p for p in trade_pnls if p < 0]))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        # Risk metrics
        equity_values = [point['equity'] for point in equity_curve]
        equity_returns = pd.Series(equity_values).pct_change().dropna()
        
        max_equity = max(equity_values)
        final_equity = equity_values[-1]
        max_drawdown = max_equity - min(equity_values[equity_values.index(max_equity):])
        max_drawdown_percentage = (max_drawdown / max_equity * 100) if max_equity > 0 else 0
        
        # Risk-adjusted returns
        if len(equity_returns) > 1:
            volatility = equity_returns.std() * np.sqrt(252 * 24)  # Hourly to annual
            sharpe_ratio = (equity_returns.mean() * 252 * 24) / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = equity_returns[equity_returns < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std() * np.sqrt(252 * 24)
                sortino_ratio = (equity_returns.mean() * 252 * 24) / downside_std
            else:
                sortino_ratio = float('inf')
        else:
            volatility = 0
            sharpe_ratio = 0
            sortino_ratio = 0
        
        calmar_ratio = (final_equity - self.initial_balance) / max_drawdown if max_drawdown > 0 else 0
        
        # Performance metrics
        total_return = ((final_equity - self.initial_balance) / self.initial_balance * 100) if self.initial_balance > 0 else 0
        
        # Time analysis
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        duration_days = (end_dt - start_dt).days
        annualized_return = (total_return / duration_days * 365) if duration_days > 0 else 0
        
        # Trade analysis
        best_trade = max(trade_pnls) if trade_pnls else 0
        worst_trade = min(trade_pnls) if trade_pnls else 0
        avg_trade = np.mean(trade_pnls) if trade_pnls else 0
        avg_winning_trade = np.mean([p for p in trade_pnls if p > 0]) if winning_trades > 0 else 0
        avg_losing_trade = np.mean([p for p in trade_pnls if p < 0]) if losing_trades > 0 else 0
        
        # Streak analysis
        win_streaks, loss_streaks = self._calculate_streaks(trade_pnls)
        largest_win_streak = max(win_streaks) if win_streaks else 0
        largest_loss_streak = max(loss_streaks) if loss_streaks else 0
        
        # Additional metrics
        trades_per_day = total_trades / duration_days if duration_days > 0 else 0
        commission_paid = sum([trade['commission'] for trade in trade_data])
        
        # Regime performance analysis
        regime_performance = self._analyze_regime_performance(trades_df, trade_pnls)
        
        return BacktestResults(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_percentage=total_pnl_percentage,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_percentage=max_drawdown_percentage,
            calmar_ratio=calmar_ratio,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_trade=avg_trade,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade,
            largest_win_streak=largest_win_streak,
            largest_loss_streak=largest_loss_streak,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration_days,
            trades_per_day=trades_per_day,
            commission_paid=commission_paid,
            regime_performance=regime_performance,
            trades=[asdict(trade) for trade in trades],
            equity_curve=equity_curve
        )
    
    def _calculate_streaks(self, trade_pnls: List[float]) -> Tuple[List[int], List[int]]:
        """Calculate winning and losing streaks"""
        if not trade_pnls:
            return [], []
        
        win_streaks = []
        loss_streaks = []
        current_win_streak = 0
        current_loss_streak = 0
        
        for pnl in trade_pnls:
            if pnl > 0:
                current_win_streak += 1
                if current_loss_streak > 0:
                    loss_streaks.append(current_loss_streak)
                    current_loss_streak = 0
            else:
                current_loss_streak += 1
                if current_win_streak > 0:
                    win_streaks.append(current_win_streak)
                    current_win_streak = 0
        
        # Add final streaks
        if current_win_streak > 0:
            win_streaks.append(current_win_streak)
        if current_loss_streak > 0:
            loss_streaks.append(current_loss_streak)
        
        return win_streaks, loss_streaks
    
    def _analyze_regime_performance(self, trades_df: pd.DataFrame, trade_pnls: List[float]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by market regime"""
        regime_stats = {}
        
        if trades_df.empty or not trade_pnls:
            return regime_stats
        
        # Group trades by regime
        regime_groups = trades_df.groupby('regime')
        
        for regime, group in regime_groups:
            regime_trades = len(group)
            regime_pnl = sum(trade_pnls[:regime_trades])  # Simplified allocation
            regime_wins = len([p for p in trade_pnls[:regime_trades] if p > 0])
            
            regime_stats[regime] = {
                'total_trades': regime_trades,
                'win_rate': (regime_wins / regime_trades * 100) if regime_trades > 0 else 0,
                'total_pnl': regime_pnl,
                'avg_confidence': group['confidence'].mean()
            }
        
        return regime_stats
    
    def _empty_results(self, start_date: str, end_date: str) -> BacktestResults:
        """Return empty results for failed backtests"""
        return BacktestResults(
            total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
            total_pnl=0, total_pnl_percentage=0, gross_profit=0, gross_loss=0, profit_factor=0,
            max_drawdown=0, max_drawdown_percentage=0, calmar_ratio=0,
            sharpe_ratio=0, sortino_ratio=0, total_return=0, annualized_return=0,
            volatility=0, best_trade=0, worst_trade=0, avg_trade=0,
            avg_winning_trade=0, avg_losing_trade=0, largest_win_streak=0, largest_loss_streak=0,
            start_date=start_date, end_date=end_date, duration_days=0,
            trades_per_day=0, commission_paid=0, regime_performance={},
            trades=[], equity_curve=[]
        )