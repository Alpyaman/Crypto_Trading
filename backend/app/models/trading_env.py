"""
Custom Trading Environment for Reinforcement Learning
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from binance.client import Client
import ta

class CryptoTradingEnv(gym.Env):
    """
    Cryptocurrency trading environment for RL agents
    
    Actions:
    - 0: Hold
    - 1: Buy
    - 2: Sell
    """
    
    def __init__(
        self,
        symbol='BTCUSDT',
        initial_balance=1000.0,
        trading_fee=0.001,
        window_size=30
    ):
        super().__init__()
        
        self.symbol = symbol
        self.initial_balance = initial_balance
        self.trading_fee = trading_fee
        self.window_size = window_size
        
        # Action: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation: OHLCV + indicators + portfolio state
        # [price_data(5) + indicators(15) + portfolio(3)] * window_size
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(23 * window_size,), 
            dtype=np.float32
        )
        
        # State variables
        self.balance = initial_balance
        self.crypto_held = 0.0
        self.current_step = 0
        self.data = None
        self.trades = []
        
    def load_data(self, api_key, api_secret, interval='1h', limit=1000):
        """Load historical data from Binance"""
        client = Client(api_key, api_secret)
        
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
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate technical indicators
        df = self._add_indicators(df)
        
        self.data = df.fillna(0)
        return self.data
    
    def _add_indicators(self, df):
        """Add technical indicators"""
        # Moving averages
        df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
        df['sma_30'] = ta.trend.sma_indicator(df['close'], window=30)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # RSI
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_mid'] = bb.bollinger_mavg()
        
        # Volume indicators
        df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
        
        # ATR (volatility)
        df['atr'] = ta.volatility.average_true_range(
            df['high'], df['low'], df['close'], window=14
        )
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close']
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        return df
    
    def reset(self, seed=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.crypto_held = 0.0
        self.current_step = self.window_size
        self.trades = []
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation"""
        if self.data is None or self.current_step >= len(self.data):
            return np.zeros(self.observation_space.shape)
        
        # Get window of data
        start_idx = max(0, self.current_step - self.window_size)
        end_idx = self.current_step
        
        window_data = self.data.iloc[start_idx:end_idx]
        
        # Market features
        features = []
        for _, row in window_data.iterrows():
            row_features = [
                row['open'], row['high'], row['low'], row['close'], row['volume'],
                row['sma_10'], row['sma_30'], row['ema_12'],
                row['macd'], row['macd_signal'],
                row['rsi'],
                row['bb_high'], row['bb_low'], row['bb_mid'],
                row['volume_ratio'], row['atr'],
                row['stoch_k'], row['stoch_d']
            ]
            features.extend(row_features)
        
        # Pad if needed
        while len(features) < 18 * self.window_size:
            features.insert(0, 0)
        
        # Portfolio state (repeated for each timestep)
        current_price = self.data.iloc[self.current_step]['close']
        portfolio_value = self.balance + self.crypto_held * current_price
        
        portfolio_features = [
            self.balance / self.initial_balance,
            self.crypto_held * current_price / self.initial_balance,
            portfolio_value / self.initial_balance,
            len(self.trades) / 100.0,
            (self.crypto_held > 0) * 1.0
        ]
        
        # Add portfolio state to features
        features.extend(portfolio_features * self.window_size)
        
        obs = np.array(features[:self.observation_space.shape[0]], dtype=np.float32)
        return obs
    
    def step(self, action):
        """Execute one trading step"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, False, {}
        
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        reward = 0
        if action == 1:  # Buy
            if self.balance > current_price * 0.01:  # Min trade size
                buy_amount = self.balance * 0.3  # Use 30% of balance
                buy_amount = min(buy_amount, self.balance)
                
                crypto_bought = (buy_amount * (1 - self.trading_fee)) / current_price
                self.crypto_held += crypto_bought
                self.balance -= buy_amount
                
                self.trades.append({
                    'step': self.current_step,
                    'action': 'buy',
                    'price': current_price,
                    'amount': crypto_bought
                })
                
        elif action == 2:  # Sell
            if self.crypto_held > 0:
                sell_amount = self.crypto_held * 0.5  # Sell 50%
                cash_received = sell_amount * current_price * (1 - self.trading_fee)
                
                self.balance += cash_received
                self.crypto_held -= sell_amount
                
                self.trades.append({
                    'step': self.current_step,
                    'action': 'sell',
                    'price': current_price,
                    'amount': sell_amount
                })
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        next_price = self.data.iloc[self.current_step]['close']
        portfolio_value = self.balance + self.crypto_held * next_price
        reward = (portfolio_value - self.initial_balance) / self.initial_balance * 100
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        # Info
        info = {
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'crypto_held': self.crypto_held,
            'trades': len(self.trades),
            'current_price': next_price
        }
        
        return self._get_observation(), reward, done, False, info
    
    def render(self):
        """Render current state"""
        current_price = self.data.iloc[self.current_step]['close']
        portfolio_value = self.balance + self.crypto_held * current_price
        
        print(f"Step: {self.current_step}")
        print(f"Price: ${current_price:.2f}")
        print(f"Balance: ${self.balance:.2f}")
        print(f"Crypto: {self.crypto_held:.6f}")
        print(f"Portfolio: ${portfolio_value:.2f}")
        print(f"Trades: {len(self.trades)}")
        print("-" * 40)