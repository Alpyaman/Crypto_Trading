"""
Enhanced Futures Trading Environment
Specialized environment for futures trading with leverage, position management, and advanced features
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from binance.client import Client
import ta
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EnhancedFuturesEnv(gym.Env):
    """
    Enhanced Cryptocurrency Futures Trading Environment
    
    Actions:
    - 0: Close Position
    - 1: Long Position
    - 2: Short Position
    - 3: Hold Current Position
    """
    
    def __init__(
        self,
        symbol='BTCUSDT',
        initial_balance=10000.0,
        trading_fee=0.0004,  # Futures trading fee
        funding_fee=0.0001,   # Funding rate
        window_size=50,
        max_leverage=20,
        position_size_factor=0.1  # Maximum 10% of balance per trade
    ):
        super().__init__()
        
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.funding_fee = funding_fee
        self.window_size = window_size
        self.max_leverage = max_leverage
        self.position_size_factor = position_size_factor
        
        # Actions: 0=Close, 1=Long, 2=Short, 3=Hold
        self.action_space = spaces.Discrete(4)
        
        # Enhanced observation space for futures trading
        # [market_data + indicators + position_info + risk_metrics]
        obs_size = 100  # Increased for more features
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Futures-specific state variables
        self.balance = initial_balance
        self.position_size = 0.0  # Positive for long, negative for short
        self.position_entry_price = 0.0
        self.leverage = 10
        self.margin_used = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Trading state
        self.current_step = 0
        self.data = None
        self.trades = []
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance
        
        # Risk management
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.15  # 15% take profit
        
    def load_data(self, api_key: str, api_secret: str, interval: str = '1h', limit: int = 2000) -> pd.DataFrame:
        """Load enhanced historical data from Binance"""
        try:
            client = Client(api_key, api_secret)
            
            # Get kline data
            klines = client.get_klines(
                symbol=self.symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert to numeric
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add enhanced indicators
            df = self._add_enhanced_indicators(df)
            
            # Add futures-specific features
            df = self._add_futures_features(df)
            
            self.data = df.fillna(0)
            logger.info(f"Loaded {len(self.data)} data points with enhanced features")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading enhanced data: {e}")
            raise
    
    def set_data(self, data: pd.DataFrame):
        """Set data directly for testing purposes"""
        # Add enhanced indicators to the data
        enhanced_data = self._add_enhanced_indicators(data.copy())
        enhanced_data = self._add_futures_features(enhanced_data)
        self.data = enhanced_data.fillna(0)
        logger.info(f"Set test data with {len(self.data)} data points")
        return self.data
    
    def _add_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        # Price-based indicators
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['close'].rolling(20).std()
        df['volatility_ratio'] = df['volatility'] / df['close']
        
        # Multiple timeframe moving averages (including MACD components)
        for period in [5, 10, 12, 20, 26, 50, 100]:
            df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
            df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
        
        # Trend indicators
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Momentum indicators
        df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
        df['rsi_21'] = ta.momentum.rsi(df['close'], window=21)
        df['rsi'] = df['rsi_14']  # Standard RSI alias for compatibility
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # Volatility indicators
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Volume indicators
        df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
        df['price_volume'] = df['close'] * df['volume']
        df['volume_momentum'] = df['volume'].pct_change()
        
        # Additional volatility features
        df['price_volatility'] = df['close'].rolling(20).std()
        df['volatility_ratio'] = df['volatility'] / df['close']
        
        # Support and Resistance levels
        df['support_level'] = df['low'].rolling(20).min()
        df['resistance_level'] = df['high'].rolling(20).max()
        df['support_distance'] = (df['close'] - df['support_level']) / df['close']
        df['resistance_distance'] = (df['resistance_level'] - df['close']) / df['close']
        
        # Additional momentum indicators
        df['momentum'] = ta.momentum.roc(df['close'], window=10)
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        
        # Trend strength
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        df['stoch_momentum'] = df['stoch_k'].pct_change()  # Now stoch_k exists
        
        # Pattern recognition features (simplified)
        df['doji'] = ((df['close'] - df['open']).abs() / (df['high'] - df['low'] + 1e-10) < 0.1).astype(int)
        df['hammer'] = ((df['low'] < df[['open', 'close']].min(axis=1)) & 
                       (df['high'] - df[['open', 'close']].max(axis=1) < 
                        (df[['open', 'close']].max(axis=1) - df['low']) * 0.3)).astype(int)
        df['engulfing'] = ((df['close'] > df['open'].shift(1)) & 
                          (df['open'] < df['close'].shift(1))).astype(int)
        
        # Market structure
        df['higher_highs'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_lows'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['trend_strength'] = df['higher_highs'] - df['lower_lows']
        
        # Time-based features
        if hasattr(df, 'timestamp') or 'timestamp' in df.columns:
            try:
                timestamps = pd.to_datetime(df['timestamp'] if 'timestamp' in df.columns else df.index)
                df['hour'] = timestamps.hour
                df['day_of_week'] = timestamps.dayofweek
            except Exception:
                df['hour'] = 0
                df['day_of_week'] = 0
        else:
            df['hour'] = 0
            df['day_of_week'] = 0
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def _add_futures_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add futures-specific features"""
        # Simulated funding rate (in real implementation, fetch from API)
        df['funding_rate'] = np.random.normal(0.0001, 0.0002, len(df))  # Simulated
        
        # Long/Short ratio simulation (would be real data in production)
        df['long_short_ratio'] = 1 + np.random.normal(0, 0.2, len(df))
        
        # Open Interest simulation
        df['open_interest_change'] = np.random.normal(0.01, 0.05, len(df))
        
        # Liquidation levels simulation
        df['liquidation_pressure'] = np.abs(np.random.normal(0, 0.1, len(df)))
        
        return df
    
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset financial state
        self.balance = self.initial_balance
        self.position_size = 0.0
        self.position_entry_price = 0.0
        self.margin_used = 0.0
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        
        # Reset trading state
        self.current_step = self.window_size
        self.trades = []
        self.max_drawdown = 0.0
        self.peak_balance = self.initial_balance
        
        return self._get_observation(), self._get_info()
    
    def _get_observation(self) -> np.ndarray:
        """Generate enhanced observation vector"""
        if self.data is None or self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape[0])
        
        # Current market data
        current_row = self.data.iloc[self.current_step]
        current_price = current_row['close']
        
        # Market features (last 20 values of key indicators)
        features = []
        
        # Price action features
        lookback = min(20, self.current_step)
        recent_data = self.data.iloc[max(0, self.current_step - lookback):self.current_step + 1]
        
        if len(recent_data) > 0:
            # Price movements
            features.extend([
                current_row['price_change'],
                current_row['volatility_ratio'],
                current_row['bb_position'],
                current_row['bb_width'],
            ])
            
            # Trend indicators
            features.extend([
                current_row['sma_20'] / current_price if current_price > 0 else 0,
                current_row['ema_20'] / current_price if current_price > 0 else 0,
                current_row['macd'] / current_price if current_price > 0 else 0,
                current_row['macd_histogram'] / current_price if current_price > 0 else 0,
            ])
            
            # Momentum indicators
            features.extend([
                current_row['rsi_14'] / 100.0,
                current_row['rsi_21'] / 100.0,
                current_row['williams_r'] / 100.0,
                current_row['stoch_k'] / 100.0,
            ])
            
            # Volume indicators
            features.extend([
                current_row['volume_ratio'],
                np.log1p(current_row['volume']) / 20.0,  # Normalized volume
            ])
            
            # Futures-specific features
            features.extend([
                current_row['funding_rate'] * 10000,  # Scale up
                current_row['long_short_ratio'],
                current_row['open_interest_change'],
                current_row['liquidation_pressure'],
            ])
        
        # Position and portfolio features
        total_value = self.balance + self.unrealized_pnl
        features.extend([
            # Portfolio state
            self.balance / self.initial_balance,
            total_value / self.initial_balance,
            self.realized_pnl / self.initial_balance,
            self.unrealized_pnl / self.initial_balance,
            
            # Position state
            self.position_size / self.initial_balance if self.initial_balance > 0 else 0,
            self.leverage / self.max_leverage,
            self.margin_used / self.balance if self.balance > 0 else 0,
            
            # Position details
            1.0 if self.position_size > 0 else 0.0,  # Long position
            1.0 if self.position_size < 0 else 0.0,  # Short position
            abs(self.position_size) / (self.balance / current_price) if self.balance > 0 and current_price > 0 else 0,
            
            # Risk metrics
            self.max_drawdown,
            (total_value - self.peak_balance) / self.peak_balance if self.peak_balance > 0 else 0,
            len(self.trades) / 100.0,  # Normalized trade count
        ])
        
        # Market structure features
        if len(recent_data) >= 10:
            # Support/Resistance
            support = recent_data['low'].min()
            resistance = recent_data['high'].max()
            features.extend([
                (current_price - support) / current_price if current_price > 0 else 0,
                (resistance - current_price) / current_price if current_price > 0 else 0,
            ])
            
            # Trend strength
            price_trend = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            volume_trend = (recent_data['volume'].iloc[-1] - recent_data['volume'].iloc[0]) / recent_data['volume'].iloc[0]
            features.extend([
                price_trend,
                volume_trend,
            ])
        
        # Pad or truncate to exact size
        while len(features) < self.observation_space.shape[0]:
            features.append(0.0)
        
        features = features[:self.observation_space.shape[0]]
        
        return np.array(features, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute trading action"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, False, self._get_info()
        
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        reward = self._execute_action(action, current_price)
        
        # Update position PnL
        self._update_position_pnl(current_price)
        
        # Check for liquidation or margin call
        liquidated = self._check_liquidation(current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate final reward
        total_reward = reward + self._calculate_performance_reward()
        
        # Check if episode is done
        done = (self.current_step >= len(self.data) - 1) or liquidated or (self.balance <= 0)
        
        # Update risk metrics
        self._update_risk_metrics()
        
        return self._get_observation(), total_reward, done, False, self._get_info()
    
    def _execute_action(self, action: int, current_price: float) -> float:
        """Execute the selected action"""
        reward = 0.0
        
        if action == 0:  # Close Position
            if self.position_size != 0:
                reward = self._close_position(current_price)
        
        elif action == 1:  # Long Position
            if self.position_size <= 0:  # Close short or open long
                if self.position_size < 0:
                    reward += self._close_position(current_price)
                reward += self._open_long_position(current_price)
        
        elif action == 2:  # Short Position
            if self.position_size >= 0:  # Close long or open short
                if self.position_size > 0:
                    reward += self._close_position(current_price)
                reward += self._open_short_position(current_price)
        
        # action == 3 is Hold - no action needed
        
        return reward
    
    def _open_long_position(self, price: float) -> float:
        """Open a long position"""
        position_value = self.balance * self.position_size_factor
        position_size = (position_value * self.leverage) / price
        
        # Calculate fees
        fee = position_value * self.trading_fee
        
        # Update state
        self.position_size = position_size
        self.position_entry_price = price
        self.margin_used = position_value
        self.balance -= fee
        
        # Record trade
        self.trades.append({
            'step': self.current_step,
            'action': 'open_long',
            'price': price,
            'size': position_size,
            'leverage': self.leverage
        })
        
        return -fee / self.initial_balance  # Small negative reward for fees
    
    def _open_short_position(self, price: float) -> float:
        """Open a short position"""
        position_value = self.balance * self.position_size_factor
        position_size = -(position_value * self.leverage) / price  # Negative for short
        
        # Calculate fees
        fee = position_value * self.trading_fee
        
        # Update state
        self.position_size = position_size
        self.position_entry_price = price
        self.margin_used = position_value
        self.balance -= fee
        
        # Record trade
        self.trades.append({
            'step': self.current_step,
            'action': 'open_short',
            'price': price,
            'size': abs(position_size),
            'leverage': self.leverage
        })
        
        return -fee / self.initial_balance  # Small negative reward for fees
    
    def _close_position(self, price: float) -> float:
        """Close current position"""
        if self.position_size == 0:
            return 0.0
        
        # Calculate PnL
        if self.position_size > 0:  # Long position
            pnl = (price - self.position_entry_price) * self.position_size
        else:  # Short position
            pnl = (self.position_entry_price - price) * abs(self.position_size)
        
        # Calculate fees
        position_value = abs(self.position_size) * price / self.leverage
        fee = position_value * self.trading_fee
        
        # Update balance
        net_pnl = pnl - fee
        self.balance += self.margin_used + net_pnl
        self.realized_pnl += net_pnl
        
        # Record trade
        self.trades.append({
            'step': self.current_step,
            'action': 'close_position',
            'price': price,
            'pnl': net_pnl,
            'size': abs(self.position_size)
        })
        
        # Reset position
        self.position_size = 0.0
        self.position_entry_price = 0.0
        self.margin_used = 0.0
        self.unrealized_pnl = 0.0
        
        return net_pnl / self.initial_balance  # Reward based on PnL
    
    def _update_position_pnl(self, current_price: float):
        """Update unrealized PnL for current position"""
        if self.position_size == 0:
            self.unrealized_pnl = 0.0
            return
        
        if self.position_size > 0:  # Long position
            self.unrealized_pnl = (current_price - self.position_entry_price) * self.position_size
        else:  # Short position
            self.unrealized_pnl = (self.position_entry_price - current_price) * abs(self.position_size)
    
    def _check_liquidation(self, current_price: float) -> bool:
        """Check if position should be liquidated"""
        if self.position_size == 0:
            return False
        
        # Calculate liquidation price (simplified)
        if self.position_size > 0:  # Long position
            liquidation_price = self.position_entry_price * (1 - 0.8 / self.leverage)
            if current_price <= liquidation_price:
                self._liquidate_position(current_price)
                return True
        else:  # Short position
            liquidation_price = self.position_entry_price * (1 + 0.8 / self.leverage)
            if current_price >= liquidation_price:
                self._liquidate_position(current_price)
                return True
        
        return False
    
    def _liquidate_position(self, current_price: float):
        """Liquidate position due to insufficient margin"""
        logger.warning(f"Position liquidated at price {current_price}")
        
        # Lose all margin
        self.balance -= self.margin_used
        self.realized_pnl -= self.margin_used
        
        # Record liquidation
        self.trades.append({
            'step': self.current_step,
            'action': 'liquidation',
            'price': current_price,
            'loss': self.margin_used
        })
        
        # Reset position
        self.position_size = 0.0
        self.position_entry_price = 0.0
        self.margin_used = 0.0
        self.unrealized_pnl = 0.0
    
    def _calculate_performance_reward(self) -> float:
        """Calculate reward based on overall performance"""
        total_value = self.balance + self.unrealized_pnl
        return_pct = (total_value - self.initial_balance) / self.initial_balance
        
        # Reward for positive returns, penalty for drawdown
        performance_reward = return_pct * 0.1
        drawdown_penalty = self.max_drawdown * 0.05
        
        return performance_reward - drawdown_penalty
    
    def _update_risk_metrics(self):
        """Update risk management metrics"""
        total_value = self.balance + self.unrealized_pnl
        
        # Update peak balance
        if total_value > self.peak_balance:
            self.peak_balance = total_value
        
        # Update max drawdown
        current_drawdown = (self.peak_balance - total_value) / self.peak_balance
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    def _get_info(self) -> Dict:
        """Get current environment info"""
        total_value = self.balance + self.unrealized_pnl
        
        return {
            'balance': self.balance,
            'position_size': self.position_size,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_value': total_value,
            'margin_used': self.margin_used,
            'leverage': self.leverage,
            'trades_count': len(self.trades),
            'max_drawdown': self.max_drawdown,
            'return_pct': (total_value - self.initial_balance) / self.initial_balance * 100,
            'current_price': self.data.iloc[self.current_step]['close'] if self.current_step < len(self.data) else 0
        }
    
    def render(self, mode: str = 'human'):
        """Render current environment state"""
        info = self._get_info()
        
        print(f"Step: {self.current_step}")
        print(f"Price: ${info['current_price']:.2f}")
        print(f"Balance: ${info['balance']:.2f}")
        print(f"Position: {info['position_size']:.6f}")
        print(f"Unrealized PnL: ${info['unrealized_pnl']:.2f}")
        print(f"Total Value: ${info['total_value']:.2f}")
        print(f"Return: {info['return_pct']:.2f}%")
        print(f"Max Drawdown: {info['max_drawdown']:.2f}%")
        print(f"Trades: {info['trades_count']}")
        print("-" * 50)