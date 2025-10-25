"""
Enhanced ML Service for Futures Trading
Advanced machine learning with futures-specific features and multi-timeframe analysis
"""
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from typing import Tuple, Dict
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import ta
from datetime import datetime

from app.models.trading_env import CryptoTradingEnv

logger = logging.getLogger(__name__)


class EnhancedMLService:
    """Enhanced ML Service with futures-specific features and advanced analytics"""
    
    def __init__(self, model_path: str = "models/enhanced_futures_trader.zip"):
        self.model_path = model_path
        self.model = None
        self.env = None
        self.scaler = StandardScaler()
        self.regime_detector = RandomForestClassifier(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Enhanced features for futures trading
        self.feature_names = []
        self.model_performance = {}
        self.market_regime = "unknown"
        
    def extract_enhanced_features(self, df: pd.DataFrame, symbol: str = 'BTCUSDT') -> pd.DataFrame:
        """Extract comprehensive features for futures trading"""
        try:
            logger.info("Extracting enhanced features for futures trading...")
            
            # Basic OHLCV
            features_df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Price-based features
            features_df['price_change'] = df['close'].pct_change()
            features_df['high_low_ratio'] = df['high'] / df['low']
            features_df['open_close_ratio'] = df['open'] / df['close']
            features_df['volume_change'] = df['volume'].pct_change()
            
            # Advanced Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                features_df[f'sma_{period}'] = ta.trend.sma_indicator(df['close'], window=period)
                features_df[f'ema_{period}'] = ta.trend.ema_indicator(df['close'], window=period)
                features_df[f'price_to_sma_{period}'] = df['close'] / features_df[f'sma_{period}']
            
            # MACD family
            macd = ta.trend.MACD(df['close'])
            features_df['macd'] = macd.macd()
            features_df['macd_signal'] = macd.macd_signal()
            features_df['macd_histogram'] = macd.macd_diff()
            features_df['macd_momentum'] = features_df['macd'].pct_change()
            
            # RSI variations
            for period in [14, 21, 30]:
                features_df[f'rsi_{period}'] = ta.momentum.rsi(df['close'], window=period)
            
            features_df['rsi_slope'] = features_df['rsi_14'].diff()
            features_df['rsi_divergence'] = self._calculate_divergence(df['close'], features_df['rsi_14'])
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20)
            features_df['bb_upper'] = bb.bollinger_hband()
            features_df['bb_lower'] = bb.bollinger_lband()
            features_df['bb_middle'] = bb.bollinger_mavg()
            features_df['bb_width'] = (features_df['bb_upper'] - features_df['bb_lower']) / features_df['bb_middle']
            features_df['bb_position'] = (df['close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
            
            # Volatility indicators
            features_df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            features_df['volatility_ratio'] = features_df['atr'] / df['close']
            features_df['price_volatility'] = df['close'].rolling(20).std()
            
            # Volume analysis
            features_df['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
            features_df['volume_ratio'] = df['volume'] / features_df['volume_sma']
            features_df['price_volume'] = df['close'] * df['volume']
            features_df['volume_momentum'] = df['volume'].pct_change()
            
            # Support and Resistance levels
            features_df['support_level'] = df['low'].rolling(20).min()
            features_df['resistance_level'] = df['high'].rolling(20).max()
            features_df['support_distance'] = (df['close'] - features_df['support_level']) / df['close']
            features_df['resistance_distance'] = (features_df['resistance_level'] - df['close']) / df['close']
            
            # Momentum indicators
            features_df['momentum'] = ta.momentum.roc(df['close'], window=10)
            features_df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
            
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            features_df['stoch_k'] = stoch.stoch()
            features_df['stoch_d'] = stoch.stoch_signal()
            features_df['stoch_momentum'] = features_df['stoch_k'].pct_change()
            
            # Trend strength
            features_df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
            features_df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
            
            # Pattern recognition features
            features_df['doji'] = self._detect_doji(df)
            features_df['hammer'] = self._detect_hammer(df)
            features_df['engulfing'] = self._detect_engulfing(df)
            
            # Multi-timeframe features
            features_df = self._add_multi_timeframe_features(features_df, df)
            
            # Market structure
            features_df['higher_highs'] = self._detect_higher_highs(df['high'])
            features_df['lower_lows'] = self._detect_lower_lows(df['low'])
            features_df['trend_strength'] = features_df['higher_highs'] - features_df['lower_lows']
            
            # Time-based features
            features_df['hour'] = pd.to_datetime(df.index).hour if hasattr(df.index, 'hour') else 0
            features_df['day_of_week'] = pd.to_datetime(df.index).dayofweek if hasattr(df.index, 'dayofweek') else 0
            features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
            
            # Fill NaN values
            features_df = features_df.fillna(method='ffill').fillna(0)
            
            # Store feature names
            self.feature_names = features_df.columns.tolist()
            
            logger.info(f"Extracted {len(self.feature_names)} enhanced features")
            return features_df
            
        except Exception as e:
            logger.error(f"Error extracting enhanced features: {e}")
            return df
    
    def _calculate_divergence(self, price: pd.Series, indicator: pd.Series, window: int = 14) -> pd.Series:
        """Calculate price-indicator divergence"""
        try:
            price_slope = price.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            indicator_slope = indicator.rolling(window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
            
            # Divergence occurs when price and indicator have opposite slopes
            divergence = (price_slope * indicator_slope < 0).astype(int)
            return divergence
        except Exception:
            return pd.Series(0, index=price.index)
    
    def _detect_doji(self, df: pd.DataFrame, threshold: float = 0.001) -> pd.Series:
        """Detect Doji candlestick patterns"""
        body_size = abs(df['close'] - df['open']) / df['close']
        return (body_size < threshold).astype(int)
    
    def _detect_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Detect Hammer candlestick patterns"""
        body_size = abs(df['close'] - df['open'])
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        
        hammer = (lower_shadow > 2 * body_size) & (upper_shadow < body_size)
        return hammer.astype(int)
    
    def _detect_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """Detect Engulfing patterns"""
        bullish_engulfing = (
            (df['close'] > df['open']) &  # Current candle is bullish
            (df['close'].shift(1) < df['open'].shift(1)) &  # Previous candle is bearish
            (df['open'] < df['close'].shift(1)) &  # Current open below previous close
            (df['close'] > df['open'].shift(1))   # Current close above previous open
        )
        
        bearish_engulfing = (
            (df['close'] < df['open']) &  # Current candle is bearish
            (df['close'].shift(1) > df['open'].shift(1)) &  # Previous candle is bullish
            (df['open'] > df['close'].shift(1)) &  # Current open above previous close
            (df['close'] < df['open'].shift(1))   # Current close below previous open
        )
        
        return (bullish_engulfing.astype(int) - bearish_engulfing.astype(int))
    
    def _add_multi_timeframe_features(self, features_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """Add multi-timeframe analysis features"""
        try:
            # Simulate higher timeframe by resampling
            # 4H features from 1H data
            df_4h = df.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).fillna(method='ffill')
            
            if len(df_4h) > 0:
                features_df['close_4h'] = df_4h['close'].reindex(df.index, method='ffill')
                features_df['sma_20_4h'] = ta.trend.sma_indicator(df_4h['close'], window=20).reindex(df.index, method='ffill')
                features_df['rsi_4h'] = ta.momentum.rsi(df_4h['close'], window=14).reindex(df.index, method='ffill')
                
                # Trend alignment
                features_df['trend_alignment'] = (
                    (df['close'] > features_df['sma_20_4h']).astype(int)
                )
            
            return features_df.fillna(0)
            
        except Exception as e:
            logger.warning(f"Could not add multi-timeframe features: {e}")
            return features_df
    
    def _detect_higher_highs(self, high_series: pd.Series, window: int = 20) -> pd.Series:
        """Detect higher highs pattern"""
        rolling_max = high_series.rolling(window).max()
        higher_highs = (high_series > rolling_max.shift(1)).astype(int)
        return higher_highs
    
    def _detect_lower_lows(self, low_series: pd.Series, window: int = 20) -> pd.Series:
        """Detect lower lows pattern"""
        rolling_min = low_series.rolling(window).min()
        lower_lows = (low_series < rolling_min.shift(1)).astype(int)
        return lower_lows
    
    def detect_market_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime (trending/ranging/volatile)"""
        try:
            # Calculate trend strength
            sma_20 = ta.trend.sma_indicator(df['close'], window=20)
            sma_50 = ta.trend.sma_indicator(df['close'], window=50)
            
            # Volatility measure
            volatility = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
            
            # Trend direction
            trend_up = (sma_20.iloc[-1] > sma_50.iloc[-1]) and (df['close'].iloc[-1] > sma_20.iloc[-1])
            trend_down = (sma_20.iloc[-1] < sma_50.iloc[-1]) and (df['close'].iloc[-1] < sma_20.iloc[-1])
            
            # High volatility threshold
            high_vol = volatility.iloc[-1] > volatility.quantile(0.8)
            
            if high_vol:
                regime = "volatile"
            elif trend_up or trend_down:
                regime = "trending"
            else:
                regime = "ranging"
            
            self.market_regime = regime
            logger.info(f"Detected market regime: {regime}")
            return regime
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return "unknown"
    
    def calculate_position_size(self, 
                              confidence: float, 
                              account_balance: float, 
                              current_price: float,
                              volatility: float,
                              regime: str = "balanced") -> float:
        """Calculate optimal position size based on confidence and risk"""
        try:
            # Base risk per trade (% of account)
            base_risk = {
                "conservative": 0.01,  # 1%
                "balanced": 0.02,      # 2%
                "aggressive": 0.03     # 3%
            }.get(regime, 0.02)
            
            # Adjust based on confidence
            confidence_multiplier = min(confidence * 2, 1.5)  # Max 1.5x
            
            # Adjust based on volatility (reduce size in high volatility)
            volatility_adjustment = max(0.5, 1 - volatility)
            
            # Adjust based on market regime
            regime_multiplier = {
                "trending": 1.2,
                "ranging": 0.8,
                "volatile": 0.6,
                "unknown": 0.7
            }.get(self.market_regime, 1.0)
            
            # Calculate position size
            risk_amount = account_balance * base_risk * confidence_multiplier * volatility_adjustment * regime_multiplier
            position_size = risk_amount / current_price
            
            # Ensure minimum and maximum position sizes
            min_position = account_balance * 0.005 / current_price  # 0.5% minimum
            max_position = account_balance * 0.1 / current_price    # 10% maximum
            
            position_size = max(min_position, min(position_size, max_position))
            
            logger.info(f"Calculated position size: {position_size:.6f} (confidence: {confidence:.2f}, regime: {self.market_regime})")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return account_balance * 0.01 / current_price  # Fallback to 1%
    
    def train_enhanced_model(self,
                           api_key: str,
                           api_secret: str,
                           symbol: str = 'BTCUSDT',
                           total_timesteps: int = 200000,
                           algorithm: str = 'PPO') -> bool:
        """Train enhanced model with improved features"""
        try:
            logger.info(f"Training enhanced {algorithm} model for {symbol}")
            
            # Create enhanced environment
            env = CryptoTradingEnv(
                symbol=symbol,
                initial_balance=10000.0,
                trading_fee=0.0004,  # Futures fee
                window_size=50  # Larger window for more context
            )
            
            # Load and enhance data
            raw_data = env.load_data(api_key, api_secret, interval='1h', limit=2000)
            enhanced_data = self.extract_enhanced_features(raw_data, symbol)
            
            # Replace environment data with enhanced features
            env.data = enhanced_data
            
            # Detect market regime for training data
            self.detect_market_regime(enhanced_data)
            
            # Wrap environment
            vec_env = DummyVecEnv([lambda: env])
            
            # Choose algorithm
            if algorithm == 'PPO':
                self.model = PPO(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=3e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    ent_coef=0.01,
                    verbose=1,
                    policy_kwargs=dict(net_arch=[512, 512, 256])
                )
            elif algorithm == 'A2C':
                self.model = A2C(
                    "MlpPolicy",
                    vec_env,
                    learning_rate=7e-4,
                    n_steps=5,
                    gamma=0.99,
                    gae_lambda=1.0,
                    ent_coef=0.01,
                    verbose=1
                )
            else:
                raise ValueError(f"Unsupported algorithm: {algorithm}")
            
            # Set up callbacks
            checkpoint_callback = CheckpointCallback(
                save_freq=10000,
                save_path='./models/checkpoints/',
                name_prefix=f'{algorithm.lower()}_futures'
            )
            
            # Train
            logger.info("Starting enhanced model training...")
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=[checkpoint_callback],
                progress_bar=True
            )
            
            # Save model and metadata
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.model.save(self.model_path)
            
            # Save additional components
            joblib.dump(self.scaler, self.model_path.replace('.zip', '_scaler.pkl'))
            
            # Save training metadata
            metadata = {
                'algorithm': algorithm,
                'symbol': symbol,
                'features_count': len(self.feature_names),
                'feature_names': self.feature_names,
                'training_timesteps': total_timesteps,
                'market_regime': self.market_regime,
                'training_date': datetime.now().isoformat()
            }
            
            joblib.dump(metadata, self.model_path.replace('.zip', '_metadata.pkl'))
            
            logger.info(f"Enhanced model training completed and saved to {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error training enhanced model: {e}")
            return False
    
    def load_enhanced_model(self) -> bool:
        """Load enhanced model with all components"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Enhanced model not found at {self.model_path}")
                return False
            
            # Load main model
            self.model = PPO.load(self.model_path)
            
            # Load scaler if available
            scaler_path = self.model_path.replace('.zip', '_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            # Load metadata if available
            metadata_path = self.model_path.replace('.zip', '_metadata.pkl')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.feature_names = metadata.get('feature_names', [])
                self.market_regime = metadata.get('market_regime', 'unknown')
                logger.info(f"Loaded model metadata: {metadata}")
            
            logger.info("Enhanced model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading enhanced model: {e}")
            return False
    
    def predict_enhanced(self,
                        observation: np.ndarray,
                        market_data: pd.DataFrame,
                        account_balance: float = 10000.0,
                        deterministic: bool = True) -> Tuple[int, float, float, Dict]:
        """Enhanced prediction with confidence, position sizing, and market analysis"""
        if self.model is None:
            logger.warning("Enhanced model not loaded")
            return 0, 0.0, 0.0, {}
        
        try:
            # Make prediction
            action, _states = self.model.predict(observation, deterministic=deterministic)
            
            if hasattr(action, 'item'):
                action = action.item()
            action = int(action)
            
            # Enhanced confidence calculation
            confidence = self._calculate_enhanced_confidence(observation, market_data)
            
            # Detect current market regime
            current_regime = self.detect_market_regime(market_data)
            
            # Calculate optimal position size
            current_price = market_data['close'].iloc[-1]
            volatility = market_data['close'].pct_change().rolling(20).std().iloc[-1]
            
            position_size = self.calculate_position_size(
                confidence, account_balance, current_price, volatility, current_regime
            )
            
            # Additional analysis
            analysis = {
                'market_regime': current_regime,
                'volatility': volatility,
                'trend_strength': self._calculate_trend_strength(market_data),
                'support_resistance': self._get_support_resistance(market_data),
                'risk_score': self._calculate_risk_score(market_data),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Enhanced prediction: action={action}, confidence={confidence:.3f}, size={position_size:.6f}")
            
            return action, confidence, position_size, analysis
            
        except Exception as e:
            logger.error(f"Error in enhanced prediction: {e}")
            return 0, 0.0, 0.0, {}
    
    def _calculate_enhanced_confidence(self, observation: np.ndarray, market_data: pd.DataFrame) -> float:
        """Calculate prediction confidence using multiple methods"""
        try:
            # Base confidence (simplified)
            base_confidence = 0.6
            
            # Trend consistency boost
            sma_20 = ta.trend.sma_indicator(market_data['close'], window=20)
            sma_50 = ta.trend.sma_indicator(market_data['close'], window=50)
            
            trend_consistency = 0.0
            if len(sma_20) > 0 and len(sma_50) > 0:
                if sma_20.iloc[-1] > sma_50.iloc[-1]:  # Uptrend
                    trend_consistency = 0.1
                elif sma_20.iloc[-1] < sma_50.iloc[-1]:  # Downtrend
                    trend_consistency = 0.1
            
            # Volume confirmation
            volume_avg = market_data['volume'].rolling(20).mean().iloc[-1]
            current_volume = market_data['volume'].iloc[-1]
            volume_boost = min(0.1, (current_volume / volume_avg - 1) * 0.1)
            
            # Market regime boost
            regime_boost = {
                "trending": 0.1,
                "ranging": -0.05,
                "volatile": -0.1,
                "unknown": 0.0
            }.get(self.market_regime, 0.0)
            
            confidence = base_confidence + trend_consistency + volume_boost + regime_boost
            confidence = max(0.1, min(1.0, confidence))  # Clamp between 0.1 and 1.0
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error calculating enhanced confidence: {e}")
            return 0.5
    
    def _calculate_trend_strength(self, market_data: pd.DataFrame) -> float:
        """Calculate trend strength indicator"""
        try:
            adx = ta.trend.adx(market_data['high'], market_data['low'], market_data['close']).iloc[-1]
            return min(1.0, adx / 50.0) if not np.isnan(adx) else 0.0
        except Exception:
            return 0.0
    
    def _get_support_resistance(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Get current support and resistance levels"""
        try:
            support = market_data['low'].rolling(20).min().iloc[-1]
            resistance = market_data['high'].rolling(20).max().iloc[-1]
            current_price = market_data['close'].iloc[-1]
            
            return {
                'support': support,
                'resistance': resistance,
                'support_distance': (current_price - support) / current_price,
                'resistance_distance': (resistance - current_price) / current_price
            }
        except Exception:
            return {'support': 0, 'resistance': 0, 'support_distance': 0, 'resistance_distance': 0}
    
    def _calculate_risk_score(self, market_data: pd.DataFrame) -> float:
        """Calculate overall risk score for current market conditions"""
        try:
            # Volatility component
            volatility = market_data['close'].pct_change().rolling(20).std().iloc[-1]
            vol_score = min(1.0, volatility * 10)  # Normalize
            
            # Volume spike component
            volume_avg = market_data['volume'].rolling(20).mean().iloc[-1]
            current_volume = market_data['volume'].iloc[-1]
            volume_score = min(1.0, max(0, (current_volume / volume_avg - 1)))
            
            # Trend uncertainty component
            sma_20 = ta.trend.sma_indicator(market_data['close'], window=20)
            sma_50 = ta.trend.sma_indicator(market_data['close'], window=50)
            
            trend_uncertainty = 0.5
            if len(sma_20) > 1 and len(sma_50) > 1:
                trend_uncertainty = abs(sma_20.iloc[-1] - sma_50.iloc[-1]) / market_data['close'].iloc[-1]
            
            # Combine scores
            risk_score = (vol_score * 0.4 + volume_score * 0.3 + trend_uncertainty * 0.3)
            return min(1.0, risk_score)
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 0.5
    
    def get_enhanced_model_info(self) -> Dict:
        """Get comprehensive model information"""
        if self.model is None:
            return {'loaded': False}
        
        info = {
            'loaded': True,
            'model_type': 'Enhanced PPO',
            'feature_count': len(self.feature_names),
            'market_regime': self.market_regime,
            'model_path': self.model_path
        }
        
        # Add metadata if available
        metadata_path = self.model_path.replace('.zip', '_metadata.pkl')
        if os.path.exists(metadata_path):
            try:
                metadata = joblib.load(metadata_path)
                info.update(metadata)
            except Exception:
                pass
        
        return info