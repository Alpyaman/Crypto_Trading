import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
from binance.client import Client
import ta  # Technical Analysis library for more indicators
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class CryptoTradingEnv(gym.Env):
    """
    A custom cryptocurrency trading environment for reinforcement learning.
    
    Actions:
    - 0: Hold
    - 1: Buy
    - 2: Sell
    
    State includes:
    - OHLCV data (normalized)
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Portfolio information (balance, holdings, profit/loss)
    """
    
    def __init__(
        self,
        symbols: list = None,
        initial_balance: float = 10000.0,
        trading_fee: float = 0.001,
        window_size: int = 30,
        period: str = "30 days",
        interval: str = "1h",
        binance_api_key: str = None,
        binance_api_secret: str = None
    ):
        super(CryptoTradingEnv, self).__init__()

        # Trading parameters
        if symbols is None:
            symbols = [
                'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT',
                'BNBUSDT', 'SOLUSDT', 'LINKUSDT', 'UNIUSDT',
                'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT', 'SANDUSDT', 'MANAUSDT'
            ]
        self.symbols = symbols
        self.current_symbol_idx = 0
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.window_size = window_size
        self.period = period
        self.interval = interval

        # Binance API client
        self.binance_client = Client(binance_api_key, binance_api_secret)

        # Load and prepare data for all symbols
        self.all_data = self._load_all_data_binance()
        self.processed_data = self._prepare_data()

        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.crypto_held = {symbol: 0.0 for symbol in self.symbols}
        self.total_trades = 0
        self.portfolio_values = []
        self.portfolio_value = initial_balance  # Ensure portfolio_value is initialized
        
        # Action space: 0=Hold, 1=Buy, 2=Sell for each symbol
        # We'll have 3 actions per symbol: 3 * num_symbols
        self.action_space = spaces.Discrete(3 * len(self.symbols))
        
        # Observation space
        # For each symbol: OHLCV (5) + Technical indicators (10) = 15 features
        # Plus portfolio info (4) = 15 * num_symbols * window_size + 4
        obs_dim = 23 * len(self.symbols) * self.window_size + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Metadata
        self.metadata = {'render.modes': ['human']}

        # Drawdown tracking
        self.max_drawdown = 0
        self.peak_portfolio_value = self.initial_balance
        self.volatility_window = 30
        self.portfolio_values = []

        
    def _load_all_data_binance(self) -> Dict[str, pd.DataFrame]:
        """Load minute/hourly bars from Binance API for all symbols."""
        all_data = {}
        for symbol in self.symbols:
            print(f"Loading Binance data for {symbol}...")
            try:
                klines = self.binance_client.get_historical_klines(
                    symbol,
                    self.interval,
                    self.period
                )
                df = pd.DataFrame(klines, columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'num_trades',
                    'taker_buy_base', 'taker_buy_quote', 'ignore'
                ])
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                df['date'] = pd.to_datetime(df['open_time'], unit='ms')
                df.set_index('date', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']]
                all_data[symbol] = df
                print(f"✅ Loaded {len(df)} bars for {symbol}")
            except Exception as e:
                print(f"❌ Error loading {symbol}: {e}")
                all_data[symbol] = self._create_synthetic_data()
        return all_data
    
    def _create_synthetic_data(self) -> pd.DataFrame:
        """Create synthetic price data for testing."""
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        np.random.seed(42)
        base_price = np.random.uniform(0.1, 100)  # Random base price
        prices = base_price + np.cumsum(np.random.randn(365) * 0.02 * base_price)
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.randn(365) * 0.001),
            'high': prices * (1 + np.abs(np.random.randn(365)) * 0.002),
            'low': prices * (1 - np.abs(np.random.randn(365)) * 0.002),
            'close': prices,
            'volume': np.random.uniform(1000000, 10000000, 365)
        }, index=dates)
        
        return data
    
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators using ta library and custom logic."""
        df = data.copy()
        # Add TA indicators
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_30'] = ta.trend.sma_indicator(df['close'], window=30)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
        macd = ta.trend.macd(df['close'])
        macd_signal = ta.trend.macd_signal(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd - macd_signal
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
        df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=10)
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-9)
        return df
    
    def _prepare_data(self) -> Dict[str, pd.DataFrame]:
        """Prepare and normalize the data for all symbols."""
        processed_data = {}
        
        for symbol in self.symbols:
            # Calculate technical indicators for this symbol
            data = self._calculate_technical_indicators(self.all_data[symbol])
            
            # Select features for the state
            feature_columns = [
                # OHLCV
                'open', 'high', 'low', 'close', 'volume',
                # Technical indicators
                'sma_10', 'sma_30', 'ema_12', 'ema_26', 'macd', 'macd_signal', 'macd_histogram',
                'rsi', 'bb_upper', 'bb_lower', 'bb_width', 'atr', 'stoch_k', 'stoch_d', 'adx', 'cci',
                'volume_sma', 'volume_ratio'
            ]
            
            # Fill NaN values
            data = data[feature_columns].fillna(method='bfill').fillna(method='ffill').fillna(0)
            # Use raw features only, no normalization
            processed_data[symbol] = data
        
        return processed_data
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation (state) for all symbols."""
        # Get the minimum data length across all symbols
        min_length = min(len(self.all_data[symbol]) for symbol in self.symbols)
        
        if self.current_step >= min_length:
            self.current_step = min_length - 1
        
        # Get window of market data for all symbols
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        
        all_market_data = []
        
        for symbol in self.symbols:
            # Get market data for this symbol
            symbol_data = self.processed_data[symbol].iloc[start_idx:end_idx].values
            
            # Pad with zeros if we don't have enough history
            if symbol_data.shape[0] < self.window_size:
                padding = np.zeros((self.window_size - symbol_data.shape[0], symbol_data.shape[1]))
                symbol_data = np.vstack([padding, symbol_data])
            
            all_market_data.append(symbol_data.flatten())
        
        # Combine all symbols' market data
        market_features = np.concatenate(all_market_data)
        
        # Portfolio information
        total_crypto_value = 0
        for symbol in self.symbols:
            current_price = self.all_data[symbol]['close'].iloc[self.current_step]
            total_crypto_value += self.crypto_held[symbol] * current_price
        
        portfolio_value = self.balance + total_crypto_value
        portfolio_info = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            total_crypto_value / self.initial_balance,  # Normalized total crypto value
            portfolio_value / self.initial_balance,  # Normalized total portfolio value
            self.total_trades / 100  # Normalized trade count
        ])
        
        # Combine market data and portfolio data
        observation = np.concatenate([market_features, portfolio_info])

        # Check for NaNs and replace with zeros
        if np.isnan(observation).any():
            print("[DEBUG] NaN detected in observation, replacing with zeros.")
            observation = np.nan_to_num(observation, nan=0.0)

        # Debug: print actual and expected shapes
        expected_shape = self.observation_space.shape[0]
        actual_shape = observation.shape[0]
        if actual_shape != expected_shape:
            print(f"[DEBUG] Observation shape mismatch: actual={actual_shape}, expected={expected_shape}")
            print(f"Symbols: {self.symbols}")
            print(f"Window size: {self.window_size}")
            print(f"Features per symbol: {market_features.size // (len(self.symbols) * self.window_size)}")
            raise ValueError(f"Observation shape mismatch: actual={actual_shape}, expected={expected_shape}")
        return observation.astype(np.float32)
    
    def _calculate_reward(self, action: int, prev_portfolio_value: float) -> float:
        """Calculate the reward with enhanced shaping for better performance."""
        # Calculate current portfolio value across all symbols
        total_crypto_value = 0
        for symbol in self.symbols:
            current_price = self.all_data[symbol]['close'].iloc[self.current_step]
            total_crypto_value += self.crypto_held[symbol] * current_price
        
        current_portfolio_value = self.balance + total_crypto_value
        
        # Portfolio value change reward (main component)
        if prev_portfolio_value > 0:
            portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
            reward = portfolio_return * 2000  # Increased scale for clearer signal
        else:
            reward = 0
        
        # Decode action for additional rewards/penalties
        symbol_idx = action // 3
        symbol_action = action % 3  # 0=Hold, 1=Buy, 2=Sell
        
        # Calculate current profit/loss ratio
        profit_ratio = (current_portfolio_value - self.initial_balance) / self.initial_balance
        
        # Momentum-based reward: reward actions that align with market momentum
        if symbol_idx < len(self.symbols):
            symbol = self.symbols[symbol_idx]
            current_price = self.all_data[symbol]['close'].iloc[self.current_step]
            
            # Calculate short-term momentum (5-day price change)
            if self.current_step >= 5:
                past_price = self.all_data[symbol]['close'].iloc[self.current_step - 5]
                momentum = (current_price - past_price) / past_price
                
                # Reward buying in uptrend, selling in downtrend
                if symbol_action == 1 and momentum > 0.02:  # Buy in uptrend
                    reward += momentum * 100
                elif symbol_action == 2 and momentum < -0.02:  # Sell in downtrend
                    reward += abs(momentum) * 100
        
        # Progressive profit bonus - increasing reward for higher profits
        if profit_ratio > 0:
            reward += profit_ratio ** 1.5 * 100  # Non-linear profit bonus
        
        # Portfolio diversification metrics
        non_zero_holdings = sum(1 for holding in self.crypto_held.values() if holding > 0)
        diversification_score = non_zero_holdings / len(self.symbols)
        # Reward diversification at all times
        reward += diversification_score * 0.5

        # Risk management rewards
        if profit_ratio > 0.1:  # If portfolio is up > 10%
            # Extra reward for diversification when profitable
            reward += diversification_score * 0.5
            # Reward taking some profits (having some cash when very profitable)
            cash_ratio = self.balance / current_portfolio_value if current_portfolio_value > 0 else 0
            if 0.1 <= cash_ratio <= 0.3:  # Optimal cash ratio when profitable
                reward += 1.0
        
        # Sharp penalty for significant losses
        if profit_ratio < -0.1:  # If losing more than 10%
            reward -= abs(profit_ratio) * 200  # Stronger loss penalty
        
        # Time-based performance incentive (beat buy-and-hold baseline)
        # Rough estimate: buy-and-hold should give ~45% over full period
        # So we expect ~0.1% per day on average, adjust based on current step
        if self.current_step > 50:  # After some time has passed
            expected_bnh_ratio = 0.45 * (self.current_step / 365)  # Proportional to time
            if profit_ratio > expected_bnh_ratio:
                reward += (profit_ratio - expected_bnh_ratio) * 50  # Bonus for beating baseline
        
        # Reduced trading penalty to encourage exploration
        if symbol_action != 0:  # If not holding
            reward -= 0.02  # Minimal trading penalty
        
        return reward
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        min_length = min(len(self.all_data[symbol]) for symbol in self.symbols)
        if self.current_step >= min_length - 1:
            return self._get_observation(), 0, True, False, {}
        
        # Calculate previous portfolio value
        prev_total_crypto_value = 0
        for symbol in self.symbols:
            current_price = self.all_data[symbol]['close'].iloc[self.current_step]
            prev_total_crypto_value += self.crypto_held[symbol] * current_price
        prev_portfolio_value = self.balance + prev_total_crypto_value
        
        # Decode action: which symbol and what action
        symbol_idx = action // 3
        symbol_action = action % 3  # 0=Hold, 1=Buy, 2=Sell
        
        if symbol_idx < len(self.symbols):
            symbol = self.symbols[symbol_idx]
            current_price = self.all_data[symbol]['close'].iloc[self.current_step]
            
            # Execute action with improved trading logic
            if symbol_action == 1:  # Buy
                if self.balance > current_price * 0.001:
                    # Dynamic buy amount based on portfolio size and confidence
                    max_buy_ratio = 0.3
                    min_buy_amount = max(current_price * 0.01, 10)
                    buy_amount = min(
                        self.balance * max_buy_ratio,
                        max(min_buy_amount, self.balance * 0.1)
                    )
                    # Simulate slippage: price moves 0.05% against agent
                    slippage_pct = np.random.uniform(0.0002, 0.001)
                    effective_price = current_price * (1 + slippage_pct)
                    # Dynamic fee: 0.08% to 0.15% based on volume
                    dynamic_fee = self.trading_fee + np.random.uniform(0.0001, 0.0007)
                    if buy_amount <= self.balance:
                        crypto_to_buy = (buy_amount * (1 - dynamic_fee)) / effective_price
                        self.crypto_held[symbol] += crypto_to_buy
                        self.balance -= buy_amount
                        self.total_trades += 1
                    
            elif symbol_action == 2:  # Sell
                if self.crypto_held[symbol] > 0:
                    min_sell_amount = max(current_price * 0.01, 10) / current_price
                    sell_amount = max(self.crypto_held[symbol] * 0.5, min_sell_amount)
                    sell_amount = min(sell_amount, self.crypto_held[symbol])
                    # Simulate slippage: price moves 0.05% against agent
                    slippage_pct = np.random.uniform(0.0002, 0.001)
                    effective_price = current_price * (1 - slippage_pct)
                    # Dynamic fee: 0.08% to 0.15% based on volume
                    dynamic_fee = self.trading_fee + np.random.uniform(0.0001, 0.0007)
                    cash_from_sale = sell_amount * effective_price * (1 - dynamic_fee)
                    self.balance += cash_from_sale
                    self.crypto_held[symbol] -= sell_amount
                    if self.crypto_held[symbol] < 1e-8:
                        self.crypto_held[symbol] = 0
                    self.total_trades += 1
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(action, prev_portfolio_value)
        
        # Update portfolio value history
        self.portfolio_values.append(self.portfolio_value)
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value


        # Calculate current portfolio value for early termination and info
        current_portfolio_value = self.balance + sum(
            self.crypto_held[symbol] * self.all_data[symbol]['close'].iloc[self.current_step - 1]
            if self.current_step - 1 < len(self.all_data[symbol]) else 0
            for symbol in self.symbols
        )

        # Calculate drawdown
        drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

        # Calculate volatility (standard deviation of returns over window)
        if len(self.portfolio_values) > self.volatility_window:
            # Ensure both arrays have the same length (window-1)
            returns = np.diff(self.portfolio_values[-self.volatility_window:]) / np.array(self.portfolio_values[-self.volatility_window:-1])
            volatility = np.std(returns)
        else:
            volatility = 0

        # Penalize drawdown and volatility
        reward -= drawdown * 2  # weight can be tuned
        reward -= volatility * 2  # weight can be tuned

        # Early episode termination conditions
        max_drawdown_limit = 0.35  # 35% max drawdown
        max_loss_limit = -0.25     # -25% max loss
        truncated = False
        done = self.current_step >= min_length - 1
        # End episode if max drawdown or loss threshold breached
        if drawdown > max_drawdown_limit or (current_portfolio_value - self.initial_balance) / self.initial_balance < max_loss_limit:
            truncated = True
            done = True
        
        # Calculate current portfolio value for info
        total_crypto_value = 0
        for symbol in self.symbols:
            if self.current_step < len(self.all_data[symbol]):
                current_price = self.all_data[symbol]['close'].iloc[self.current_step]
                total_crypto_value += self.crypto_held[symbol] * current_price
        
        current_portfolio_value = self.balance + total_crypto_value
        self.portfolio_values.append(current_portfolio_value)
        
        # Diversification metrics for info
        non_zero_holdings = sum(1 for holding in self.crypto_held.values() if holding > 0)
        diversification_score = non_zero_holdings / len(self.symbols)
        # Additional info
        info = {
            'portfolio_value': current_portfolio_value,
            'balance': self.balance,
            'crypto_held': dict(self.crypto_held),
            'total_trades': self.total_trades,
            'profit_loss': current_portfolio_value - self.initial_balance,
            'profit_loss_pct': (current_portfolio_value - self.initial_balance) / self.initial_balance * 100,
            'action_symbol': self.symbols[symbol_idx] if symbol_idx < len(self.symbols) else 'NONE',
            'action_type': ['Hold', 'Buy', 'Sell'][symbol_action],
            "drawdown": drawdown,
            "max_drawdown": self.max_drawdown,
            "volatility": volatility,
            "diversification_score": diversification_score,
            "non_zero_holdings": non_zero_holdings,
        }
        
        return self._get_observation(), reward, done, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = self.window_size - 1  # Start after we have enough history
        self.balance = self.initial_balance
        self.crypto_held = {symbol: 0.0 for symbol in self.symbols}  # Reset all holdings
        self.total_trades = 0
        self.portfolio_values = []
        self.portfolio_value = self.initial_balance  # Ensure portfolio_value is reset
        self.max_drawdown = 0
        self.peak_portfolio_value = self.initial_balance
        
        info = {
            'portfolio_value': self.initial_balance,
            'balance': self.balance,
            'crypto_held': dict(self.crypto_held),
            'total_trades': self.total_trades
        }
        
        return self._get_observation(), info
    
    def render(self, mode: str = 'human') -> None:
        """Render the environment state."""
        # Calculate total portfolio value
        total_crypto_value = 0
        for symbol in self.symbols:
            if self.current_step < len(self.all_data[symbol]):
                current_price = self.all_data[symbol]['close'].iloc[self.current_step]
                symbol_value = self.crypto_held[symbol] * current_price
                total_crypto_value += symbol_value
                if symbol_value > 0:
                    print(f"  {symbol}: {self.crypto_held[symbol]:.6f} (${symbol_value:.2f})")
        
        portfolio_value = self.balance + total_crypto_value
        profit_loss = portfolio_value - self.initial_balance
        profit_loss_pct = profit_loss / self.initial_balance * 100
        
        print(f"Step: {self.current_step}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Crypto Holdings:")
        print(f"Total Crypto Value: ${total_crypto_value:.2f}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Profit/Loss: ${profit_loss:.2f} ({profit_loss_pct:.2f}%)")
        print(f"Total Trades: {self.total_trades}")
        print(f"Max Drawdown: {self.max_drawdown:.2f}%")
        print(f"Volatility (30d): {np.std(np.diff(self.portfolio_values[-30:])):.2f}%")
        print("-" * 50)
    
    def close(self):
        """Clean up the environment."""
        pass


# Example usage and testing
if __name__ == "__main__":
    # Create environment with multiple symbols (same as live trading system)
    symbols = [
        'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT',
        'BNBUSDT', 'SOLUSDT', 'LINKUSDT', 'UNIUSDT',
        'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT', 'SANDUSDT', 'MANAUSDT'
    ]
    
    env = CryptoTradingEnv(symbols=symbols, initial_balance=10000)
    
    print("Environment created successfully!")
    print(f"Trading {len(env.symbols)} symbols: {env.symbols}")
    print(f"Action space: {env.action_space} (3 actions × {len(env.symbols)} symbols)")
    print(f"Observation space shape: {env.observation_space.shape}")
    
    # Test the environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Take a few random actions
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        print(f"\nStep {i+1}:")
        print(f"Action: {action} -> {info['action_type']} {info['action_symbol']}")
        print(f"Reward: {reward:.4f}")
        print(f"Portfolio Value: ${info['portfolio_value']:.2f}")
        print(f"Profit/Loss: {info['profit_loss_pct']:.2f}%")
        
        if done:
            break
    
    env.close()
