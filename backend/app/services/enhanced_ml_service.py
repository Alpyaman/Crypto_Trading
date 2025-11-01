"""
Enhanced ML Service for Futures Trading
Advanced machine learning with futures-specific features and multi-timeframe analysis
"""
import os
import numpy as np
import pandas as pd
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from typing import Tuple, Dict
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
import ta
from datetime import datetime

# Removed unused import: from app.models.trading_env import CryptoTradingEnv

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
        
        # Regime-specific models
        self.regime_models = {
            'trending': None,
            'ranging': None,
            'volatile': None
        }
        
        # Model paths for different regimes
        self.regime_model_paths = {
            'trending': "models/ppo_trending.zip",
            'ranging': "models/ppo_ranging.zip", 
            'volatile': "models/ppo_volatile.zip"
        }
        
        # Regime-specific training data and performance
        self.regime_training_data = {
            'trending': [],
            'ranging': [],
            'volatile': []
        }
        
        self.regime_performance = {
            'trending': {'episodes': 0, 'avg_reward': 0, 'win_rate': 0},
            'ranging': {'episodes': 0, 'avg_reward': 0, 'win_rate': 0},
            'volatile': {'episodes': 0, 'avg_reward': 0, 'win_rate': 0}
        }
        
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
            
            # Advanced Moving Averages (including MACD components)
            for period in [5, 10, 12, 20, 26, 30, 50, 100, 200]:
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
            
            # Add standard RSI alias for compatibility
            features_df['rsi'] = features_df['rsi_14']  # Standard RSI alias
            
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
            
            # Futures-specific features (simulated for training)
            features_df['funding_rate'] = np.random.normal(0.0001, 0.0002, len(df))  # Simulated funding rate
            features_df['long_short_ratio'] = 1 + np.random.normal(0, 0.2, len(df))  # Long/Short ratio
            features_df['open_interest_change'] = np.random.normal(0.01, 0.05, len(df))  # Open Interest change
            features_df['liquidation_pressure'] = np.abs(np.random.normal(0, 0.1, len(df)))  # Liquidation pressure
            
            # Fill NaN values
            features_df = features_df.ffill().fillna(0)
            
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
            # Check if we have timestamp column for proper resampling
            if 'timestamp' in df.columns:
                # Convert timestamp to datetime if it's not already
                df_temp = df.copy()
                df_temp['timestamp'] = pd.to_datetime(df_temp['timestamp'], unit='ms')
                df_temp.set_index('timestamp', inplace=True)
                
                # Simulate higher timeframe by resampling
                # 4H features from 1H data
                df_4h = df_temp.resample('4H').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).ffill()
                
                if len(df_4h) > 20:  # Need sufficient data
                    # Calculate 4H indicators
                    close_4h = ta.trend.sma_indicator(df_4h['close'], window=20)
                    rsi_4h = ta.momentum.rsi(df_4h['close'], window=14)
                    
                    # Reindex to match original dataframe
                    try:
                        features_df['close_4h'] = close_4h.reindex(df.index, method='ffill').fillna(0)
                        features_df['sma_20_4h'] = close_4h.reindex(df.index, method='ffill').fillna(0)
                        features_df['rsi_4h'] = rsi_4h.reindex(df.index, method='ffill').fillna(0)
                    except Exception:
                        # Fallback for RangeIndex or other index issues
                        features_df['close_4h'] = df['close'].rolling(4).mean().fillna(0)
                        features_df['sma_20_4h'] = df['close'].rolling(20).mean().fillna(0)
                        features_df['rsi_4h'] = ta.momentum.rsi(df['close'], window=14).fillna(50)
                    
                    # Trend alignment
                    features_df['trend_alignment'] = (
                        (df['close'] > features_df['sma_20_4h']).astype(int)
                    )
                else:
                    # Fallback: use simple rolling averages
                    features_df['close_4h'] = df['close'].rolling(4).mean().fillna(0)
                    features_df['sma_20_4h'] = df['close'].rolling(20).mean().fillna(0)
                    features_df['rsi_4h'] = ta.momentum.rsi(df['close'], window=14).fillna(50)
                    features_df['trend_alignment'] = (
                        (df['close'] > features_df['sma_20_4h']).astype(int)
                    )
            else:
                # Fallback approach without timestamp
                features_df['close_4h'] = df['close'].rolling(4).mean().fillna(0)
                features_df['sma_20_4h'] = df['close'].rolling(20).mean().fillna(0)
                features_df['rsi_4h'] = ta.momentum.rsi(df['close'], window=14).fillna(50)
                features_df['trend_alignment'] = (
                    (df['close'] > features_df['sma_20_4h']).astype(int)
                )
            
            return features_df.fillna(0)
            
        except Exception as e:
            logger.warning(f"Could not add multi-timeframe features: {e}")
            # Add default values to maintain feature count consistency
            features_df['close_4h'] = df['close'].fillna(0)
            features_df['sma_20_4h'] = df['close'].rolling(20).mean().fillna(0)
            features_df['rsi_4h'] = 50.0  # Neutral RSI
            features_df['trend_alignment'] = 0
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
    
    def detect_market_regime(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Enhanced market regime detection with quantitative metrics
        Returns comprehensive regime analysis including confidence scores
        """
        try:
            logger.info("🔍 Analyzing market regime with quantitative metrics...")
            
            # 1. ADX for trend strength
            adx = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
            adx_current = adx.iloc[-1] if not adx.empty else 25
            
            # 2. Bollinger Band Width for volatility
            bb_upper = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)
            bb_lower = ta.volatility.bollinger_lband(df['close'], window=20, window_dev=2)
            bb_width = ((bb_upper - bb_lower) / df['close']) * 100
            bb_width_current = bb_width.iloc[-1] if not bb_width.empty else 4
            
            # 3. Additional quantitative indicators
            # RSI for momentum
            rsi = ta.momentum.rsi(df['close'], window=14)
            rsi_current = rsi.iloc[-1] if not rsi.empty else 50
            
            # MACD for trend confirmation
            macd_line = ta.trend.macd(df['close'])
            macd_signal = ta.trend.macd_signal(df['close'])
            macd_histogram = macd_line - macd_signal
            macd_hist_current = macd_histogram.iloc[-1] if not macd_histogram.empty else 0
            
            # 4. Price action analysis
            sma_20 = ta.trend.sma_indicator(df['close'], window=20)
            
            price_vs_sma20 = (df['close'].iloc[-1] / sma_20.iloc[-1] - 1) * 100
            sma_slope = (sma_20.iloc[-1] / sma_20.iloc[-5] - 1) * 100
            
            # 5. Volume analysis
            volume_sma = df['volume'].rolling(20).mean()
            volume_ratio = df['volume'].iloc[-1] / volume_sma.iloc[-1] if volume_sma.iloc[-1] > 0 else 1
            
            # 6. Quantitative thresholds for regime classification (OPTIMIZED)
            regime_scores = {
                'trending': 0,
                'ranging': 0,
                'volatile': 0
            }
            
            # Enhanced ADX-based trend strength scoring (OPTIMIZED)
            if adx_current > 35:  # Very strong trend (higher threshold)
                regime_scores['trending'] += 50
                if adx_current > 50:  # Extremely strong trend
                    regime_scores['trending'] += 30
            elif adx_current > 25:  # Strong trend
                regime_scores['trending'] += 35
            elif adx_current > 20:  # Medium trend
                regime_scores['trending'] += 20
            elif adx_current < 18:  # Weak trend = ranging (adjusted threshold)
                regime_scores['ranging'] += 35
            
            # Enhanced Bollinger Band Width for volatility scoring (OPTIMIZED)
            bb_width_percentile = self._calculate_percentile(bb_width, bb_width_current)
            if bb_width_current > 8.0:  # Extremely high absolute volatility
                regime_scores['volatile'] += 60
            elif bb_width_current > 5.0:  # Very high absolute volatility
                regime_scores['volatile'] += 45
            elif bb_width_current > 3.0:  # High absolute volatility
                regime_scores['volatile'] += 30
            elif bb_width_percentile > 90:  # Very high relative volatility
                regime_scores['volatile'] += 50
            elif bb_width_percentile > 80:  # High relative volatility
                regime_scores['volatile'] += 35
            elif bb_width_percentile < 15:  # Low volatility = ranging
                regime_scores['ranging'] += 30
            elif bb_width_percentile < 30:  # Medium-low volatility = trending
                regime_scores['trending'] += 20
            
            # Enhanced price momentum scoring (OPTIMIZED)
            if abs(price_vs_sma20) > 8:  # Very strong deviation from SMA
                regime_scores['trending'] += 35
                if abs(price_vs_sma20) > 15:  # Extremely strong deviation
                    regime_scores['trending'] += 25
            elif abs(price_vs_sma20) > 4:  # Strong deviation
                regime_scores['trending'] += 25
            elif abs(price_vs_sma20) < 1.5:  # Very close to SMA = ranging
                regime_scores['ranging'] += 25
            
            # Enhanced MACD histogram for trend confirmation (OPTIMIZED)
            if abs(macd_hist_current) > 0.03:  # Strong momentum
                regime_scores['trending'] += 25
            elif abs(macd_hist_current) > 0.015:  # Medium momentum
                regime_scores['trending'] += 15
            elif abs(macd_hist_current) < 0.008:  # Weak momentum = ranging
                regime_scores['ranging'] += 20
            
            # Enhanced volume analysis (OPTIMIZED)
            if volume_ratio > 3.0:  # Extremely high volume = volatile
                regime_scores['volatile'] += 35
            elif volume_ratio > 2.0:  # Very high volume = volatile
                regime_scores['volatile'] += 25
            elif volume_ratio > 1.5:  # High volume = volatile
                regime_scores['volatile'] += 15
            elif volume_ratio < 0.6:  # Low volume = ranging
                regime_scores['ranging'] += 15
            
            # Enhanced RSI extremes for volatility and ranging (OPTIMIZED)
            if rsi_current > 85 or rsi_current < 15:  # Extreme RSI = very volatile
                regime_scores['volatile'] += 40
            elif rsi_current > 80 or rsi_current < 20:  # Very extreme RSI = volatile
                regime_scores['volatile'] += 30
            elif rsi_current > 70 or rsi_current < 30:  # High RSI = volatile
                regime_scores['volatile'] += 15
            elif 40 <= rsi_current <= 60:  # Neutral RSI = ranging
                regime_scores['ranging'] += 20
            
            # Additional volatility boost for extreme BB width
            if bb_width_current > 6.0:  # Very extreme BB width
                regime_scores['volatile'] += 40  # Strong boost for volatility
            elif bb_width_current > 4.5:  # Extreme BB width
                regime_scores['volatile'] += 25
            
            # SMA slope analysis for trend confirmation (ENHANCED)
            if abs(sma_slope) > 3:  # Very strong slope = trending
                regime_scores['trending'] += 25
            elif abs(sma_slope) > 1.5:  # Strong slope = trending
                regime_scores['trending'] += 15
            elif abs(sma_slope) < 0.3:  # Very flat slope = ranging
                regime_scores['ranging'] += 15
            
            # Additional trend persistence check
            if len(df) >= 10:
                recent_closes = df['close'].tail(10).values
                trend_direction_consistency = 0
                for i in range(1, len(recent_closes)):
                    if recent_closes[i] > recent_closes[i-1]:
                        trend_direction_consistency += 1
                    elif recent_closes[i] < recent_closes[i-1]:
                        trend_direction_consistency -= 1
                
                # Strong directional consistency = trending
                if abs(trend_direction_consistency) >= 6:  # 60%+ directional consistency
                    regime_scores['trending'] += 20
                elif abs(trend_direction_consistency) <= 2:  # Low consistency = ranging
                    regime_scores['ranging'] += 10
            
            # 7. Determine primary regime with enhanced confidence scoring
            max_score = max(regime_scores.values())
            min_score = min(regime_scores.values())
            score_range = max_score - min_score
            
            primary_regime = max(regime_scores, key=regime_scores.get)
            
            # Enhanced confidence calculation
            if max_score == 0:  # No clear indicators
                confidence = 0.0
            elif score_range < 15:  # Scores too close together
                confidence = 0.25  # Low confidence
            elif max_score >= 80:  # Very high score
                confidence = min(0.95, 0.7 + (max_score - 80) / 100)
            elif max_score >= 60:  # Good score
                confidence = 0.5 + (max_score - 60) / 60
            else:  # Lower scores
                confidence = max(0.2, max_score / 120)
            
            # Boost confidence if there's a clear winner
            second_highest = sorted(regime_scores.values())[-2] if len(regime_scores) > 1 else 0
            if max_score > second_highest * 1.4:  # Clear dominance
                confidence = min(0.95, confidence * 1.15)
            
            # 8. Calculate sub-regime characteristics
            trend_direction = "neutral"
            if regime_scores['trending'] > 40:
                if price_vs_sma20 > 2 and macd_hist_current > 0:
                    trend_direction = "bullish"
                elif price_vs_sma20 < -2 and macd_hist_current < 0:
                    trend_direction = "bearish"
            
            volatility_level = "medium"
            if bb_width_percentile > 80:
                volatility_level = "high"
            elif bb_width_percentile < 20:
                volatility_level = "low"
            
            # 9. Comprehensive regime analysis
            regime_analysis = {
                'primary_regime': primary_regime,
                'confidence': min(confidence, 1.0),
                'scores': regime_scores,
                'trend_direction': trend_direction,
                'volatility_level': volatility_level,
                'metrics': {
                    'adx': round(adx_current, 2),
                    'bb_width': round(bb_width_current, 2),
                    'bb_width_percentile': round(bb_width_percentile, 1),
                    'rsi': round(rsi_current, 2),
                    'macd_histogram': round(macd_hist_current, 4),
                    'price_vs_sma20': round(price_vs_sma20, 2),
                    'volume_ratio': round(volume_ratio, 2),
                    'sma_slope': round(sma_slope, 2)
                },
                'timestamp': datetime.now().isoformat(),
                'market_state': self._classify_market_state(regime_scores, confidence)
            }
            
            # Update instance variables
            self.market_regime = primary_regime
            
            logger.info(f"📊 Market Regime Analysis: {primary_regime.upper()} "
                       f"(confidence: {confidence:.2f}, ADX: {adx_current:.1f}, "
                       f"BB Width: {bb_width_current:.2f})")
            
            return regime_analysis
            
        except Exception as e:
            logger.error(f"❌ Error detecting market regime: {e}")
            return {
                'primary_regime': 'unknown',
                'confidence': 0.0,
                'scores': {'trending': 0, 'ranging': 0, 'volatile': 0},
                'trend_direction': 'neutral',
                'volatility_level': 'medium',
                'metrics': {},
                'timestamp': datetime.now().isoformat(),
                'market_state': 'uncertain'
            }
    
    def _calculate_percentile(self, series: pd.Series, current_value: float) -> float:
        """Calculate percentile rank of current value in series"""
        try:
            if len(series) == 0:
                return 50.0
            return (series < current_value).mean() * 100
        except Exception:
            return 50.0
    
    def _classify_market_state(self, scores: Dict, confidence: float) -> str:
        """Classify overall market state for strategy selection"""
        max_score = max(scores.values())
        
        if confidence > 0.7:
            if scores['trending'] == max_score:
                return "strong_trend"
            elif scores['volatile'] == max_score:
                return "high_volatility"
            else:
                return "consolidation"
        elif confidence > 0.5:
            return "moderate_" + max(scores, key=scores.get)
        else:
            return "uncertain"
    
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
            regime_value = self.market_regime
            if isinstance(regime_value, dict):
                regime_value = regime_value.get('primary_regime', 'unknown')
            
            regime_multiplier = {
                "trending": 1.2,
                "ranging": 0.8,
                "volatile": 0.6,
                "unknown": 0.7
            }.get(regime_value, 1.0)
            
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
            
            # Create enhanced futures environment
            from app.models.enhanced_futures_env import EnhancedFuturesEnv
            env = EnhancedFuturesEnv(
                symbol=symbol,
                initial_balance=10000.0,
                window_size=50  # Larger window for more context
            )
            
            # Load and enhance data
            raw_data = env.load_data(api_key, api_secret, interval='1h', limit=2000)
            enhanced_data = self.extract_enhanced_features(raw_data, symbol)
            
            # Replace environment data with enhanced features
            env.data = enhanced_data
            
            # Detect market regime for training data
            self.detect_market_regime(enhanced_data)
            
            # Wrap environment with Monitor and normalize observations/rewards
            # Monitor records episode reward info; VecNormalize normalizes obs and rewards
            env = Monitor(env)
            vec_env = DummyVecEnv([lambda: env])
            vec_norm = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            
            # Choose algorithm
            if algorithm == 'PPO':
                self.model = PPO(
                    "MlpPolicy",
                    vec_norm,
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
            # Save trained model
            self.model.save(self.model_path)
            # Save VecNormalize statistics so we can normalize observations/rewards at inference
            try:
                vec_norm.save(self.model_path.replace('.zip', '_vecnormalize.pkl'))
            except Exception:
                logger.warning("Could not save VecNormalize stats")
            
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
        """Load enhanced model with all components including VecNormalize"""
        try:
            if not os.path.exists(self.model_path):
                logger.warning(f"Enhanced model not found at {self.model_path}")
                return False
            
            # Load main model
            self.model = PPO.load(self.model_path)
            
            # Load VecNormalize stats if available
            vecnormalize_path = self.model_path.replace('.zip', '_vecnormalize.pkl')
            if os.path.exists(vecnormalize_path):
                try:
                    # Create a dummy env for VecNormalize loading
                    from app.models.enhanced_futures_env import EnhancedFuturesEnv
                    dummy_env = EnhancedFuturesEnv(symbol='BTCUSDT', initial_balance=10000.0, window_size=50)
                    dummy_vec_env = DummyVecEnv([lambda: dummy_env])
                    self.vec_normalize = VecNormalize.load(vecnormalize_path, venv=dummy_vec_env)
                    # Set training=False for inference
                    self.vec_normalize.training = False
                    logger.info("VecNormalize stats loaded for inference")
                except Exception as e:
                    logger.warning(f"Could not load VecNormalize stats: {e}")
                    self.vec_normalize = None
            else:
                self.vec_normalize = None
            
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
            # Normalize observation if VecNormalize is available
            if hasattr(self, 'vec_normalize') and self.vec_normalize is not None:
                # VecNormalize expects observations in the shape (n_envs, obs_dim)
                obs_normalized = self.vec_normalize.normalize_obs(observation.reshape(1, -1))[0]
            else:
                obs_normalized = observation
            
            # Make prediction
            action, _states = self.model.predict(obs_normalized, deterministic=deterministic)
            
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
    
    # ===== REGIME-SPECIFIC MODEL TRAINING METHODS =====
    
    def train_regime_specific_models(self,
                                   api_key: str,
                                   api_secret: str,
                                   symbol: str = 'BTCUSDT',
                                   total_timesteps: int = 100000,
                                   algorithm: str = 'PPO') -> Dict[str, bool]:
        """
        Train separate models for each market regime (trending, ranging, volatile)
        Returns training success status for each regime
        """
        try:
            logger.info("🎯 Starting regime-specific model training...")
            
            # Step 1: Load and analyze historical data
            from app.models.enhanced_futures_env import EnhancedFuturesEnv
            temp_env = EnhancedFuturesEnv(symbol=symbol, initial_balance=10000.0, window_size=50)
            raw_data = temp_env.load_data(api_key, api_secret, interval='1h', limit=5000)
            enhanced_data = self.extract_enhanced_features(raw_data, symbol)
            
            # Step 2: Segment data by market regimes
            regime_segments = self._segment_data_by_regime(enhanced_data)
            
            training_results = {}
            
            # Step 3: Train model for each regime
            for regime in ['trending', 'ranging', 'volatile']:
                if len(regime_segments[regime]) < 500:  # Minimum data requirement
                    logger.warning(f"Insufficient data for {regime} regime training ({len(regime_segments[regime])} samples)")
                    training_results[regime] = False
                    continue
                
                logger.info(f"🔄 Training {regime} model with {len(regime_segments[regime])} samples...")
                success = self._train_single_regime_model(
                    regime=regime,
                    regime_data=regime_segments[regime],
                    symbol=symbol,
                    total_timesteps=total_timesteps,
                    algorithm=algorithm
                )
                training_results[regime] = success
                
                if success:
                    logger.info(f"✅ {regime.capitalize()} model training completed")
                else:
                    logger.error(f"❌ {regime.capitalize()} model training failed")
            
            # Step 4: Update regime performance tracking
            self._update_regime_performance(training_results)
            
            logger.info(f"🎯 Regime-specific training completed: {training_results}")
            return training_results
            
        except Exception as e:
            logger.error(f"❌ Error in regime-specific training: {e}")
            return {'trending': False, 'ranging': False, 'volatile': False}
    
    def _segment_data_by_regime(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Segment historical data by market regimes"""
        try:
            regime_segments = {
                'trending': [],
                'ranging': [],
                'volatile': []
            }
            
            # Analyze data in sliding windows
            window_size = 100  # Analyze 100 candles at a time
            step_size = 20     # Move window by 20 candles
            
            for start_idx in range(0, len(data) - window_size, step_size):
                end_idx = start_idx + window_size
                window_data = data.iloc[start_idx:end_idx].copy()
                
                if len(window_data) < window_size:
                    continue
                
                # Analyze regime for this window
                regime_analysis = self.detect_market_regime(window_data)
                primary_regime = regime_analysis['primary_regime']
                confidence = regime_analysis['confidence']
                
                # Only include high-confidence regime classifications
                if confidence > 0.6 and primary_regime in regime_segments:
                    # Add metadata for training
                    window_data['regime'] = primary_regime
                    window_data['regime_confidence'] = confidence
                    window_data['window_start'] = start_idx
                    
                    regime_segments[primary_regime].append(window_data)
            
            # Combine segments for each regime
            combined_segments = {}
            for regime in regime_segments:
                if regime_segments[regime]:
                    combined_segments[regime] = pd.concat(regime_segments[regime], ignore_index=True)
                else:
                    combined_segments[regime] = pd.DataFrame()
            
            # Log regime distribution
            for regime, segment in combined_segments.items():
                logger.info(f"📊 {regime.capitalize()} regime: {len(segment)} samples")
            
            return combined_segments
            
        except Exception as e:
            logger.error(f"❌ Error segmenting data by regime: {e}")
            return {'trending': pd.DataFrame(), 'ranging': pd.DataFrame(), 'volatile': pd.DataFrame()}
    
    def _train_single_regime_model(self,
                                  regime: str,
                                  regime_data: pd.DataFrame,
                                  symbol: str,
                                  total_timesteps: int,
                                  algorithm: str) -> bool:
        """Train a single model for a specific market regime"""
        try:
            # Create regime-specific environment
            from app.models.enhanced_futures_env import EnhancedFuturesEnv
            env = EnhancedFuturesEnv(
                symbol=symbol,
                initial_balance=10000.0,
                window_size=50,
                regime_focus=regime  # Focus training on this regime
            )
            
            # Set regime-specific data
            env.data = regime_data
            
            # Configure regime-specific parameters
            regime_config = self._get_regime_training_config(regime)
            
            # Setup environment
            env = Monitor(env)
            vec_env = DummyVecEnv([lambda: env])
            vec_norm = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            
            # Create model with regime-specific configuration
            if algorithm == 'PPO':
                model = PPO(
                    "MlpPolicy",
                    vec_norm,
                    learning_rate=regime_config['learning_rate'],
                    n_steps=regime_config['n_steps'],
                    batch_size=regime_config['batch_size'],
                    n_epochs=regime_config['n_epochs'],
                    gamma=regime_config['gamma'],
                    gae_lambda=regime_config['gae_lambda'],
                    clip_range=regime_config['clip_range'],
                    ent_coef=regime_config['ent_coef'],
                    verbose=1,
                    policy_kwargs=dict(net_arch=regime_config['net_arch'])
                )
            else:
                raise ValueError(f"Algorithm {algorithm} not supported for regime training")
            
            # Setup callbacks
            model_path = self.regime_model_paths[regime]
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            checkpoint_callback = CheckpointCallback(
                save_freq=max(10000, total_timesteps // 10),
                save_path=f'./models/checkpoints/{regime}/',
                name_prefix=f'{algorithm.lower()}_{regime}'
            )
            
            # Train the model
            logger.info(f"🏃 Training {regime} model...")
            model.learn(
                total_timesteps=total_timesteps,
                callback=[checkpoint_callback],
                progress_bar=True
            )
            
            # Save regime-specific model
            model.save(model_path)
            self.regime_models[regime] = model
            
            # Save regime-specific VecNormalize and metadata
            vec_norm_path = model_path.replace('.zip', '_vecnormalize.pkl')
            try:
                vec_norm.save(vec_norm_path)
            except Exception as e:
                logger.warning(f"Could not save VecNormalize for {regime}: {e}")
            
            # Save regime-specific metadata
            metadata = {
                'regime': regime,
                'algorithm': algorithm,
                'symbol': symbol,
                'training_samples': len(regime_data),
                'training_timesteps': total_timesteps,
                'training_date': datetime.now().isoformat(),
                'regime_config': regime_config
            }
            
            metadata_path = model_path.replace('.zip', '_metadata.pkl')
            joblib.dump(metadata, metadata_path)
            
            logger.info(f"✅ {regime.capitalize()} model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error training {regime} model: {e}")
            return False
    
    def _get_regime_training_config(self, regime: str) -> Dict:
        """Get regime-specific training hyperparameters"""
        base_config = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01,
            'net_arch': [512, 512, 256]
        }
        
        # Regime-specific optimizations
        if regime == 'trending':
            # For trending markets: focus on momentum, higher learning rate
            base_config.update({
                'learning_rate': 5e-4,  # Higher learning rate for trend following
                'gamma': 0.995,         # Longer horizon for trends
                'gae_lambda': 0.98,     # Better advantage estimation for trends
                'net_arch': [512, 512, 512, 256]  # Deeper network for trend patterns
            })
        elif regime == 'ranging':
            # For ranging markets: focus on mean reversion, conservative parameters
            base_config.update({
                'learning_rate': 2e-4,  # Lower learning rate for stability
                'gamma': 0.98,          # Shorter horizon for mean reversion
                'clip_range': 0.15,     # Tighter clipping for stability
                'n_epochs': 15,         # More epochs for better convergence
                'net_arch': [256, 256, 128]  # Smaller network for simpler patterns
            })
        elif regime == 'volatile':
            # For volatile markets: focus on risk management, robust parameters
            base_config.update({
                'learning_rate': 1e-4,  # Much lower learning rate for stability
                'gamma': 0.95,          # Shorter horizon for quick adaptation
                'ent_coef': 0.02,       # Higher entropy for exploration
                'clip_range': 0.1,      # Very tight clipping for stability
                'batch_size': 32,       # Smaller batch size for faster updates
                'net_arch': [256, 256, 256, 128]  # Balanced network
            })
        
        return base_config
    
    def load_regime_specific_model(self, regime: str) -> bool:
        """Load a specific regime model"""
        try:
            if regime not in self.regime_model_paths:
                logger.error(f"Invalid regime: {regime}")
                return False
            
            model_path = self.regime_model_paths[regime]
            
            if not os.path.exists(model_path):
                logger.warning(f"Regime model not found: {model_path}")
                return False
            
            # Load the regime model
            self.regime_models[regime] = PPO.load(model_path)
            
            # Load regime-specific VecNormalize if available
            vec_norm_path = model_path.replace('.zip', '_vecnormalize.pkl')
            if os.path.exists(vec_norm_path):
                try:
                    from app.models.enhanced_futures_env import EnhancedFuturesEnv
                    dummy_env = EnhancedFuturesEnv(symbol='BTCUSDT', initial_balance=10000.0, window_size=50)
                    dummy_vec_env = DummyVecEnv([lambda: dummy_env])
                    regime_vec_norm = VecNormalize.load(vec_norm_path, venv=dummy_vec_env)
                    regime_vec_norm.training = False
                    setattr(self, f'vec_normalize_{regime}', regime_vec_norm)
                except Exception as e:
                    logger.warning(f"Could not load VecNormalize for {regime}: {e}")
            
            logger.info(f"✅ {regime.capitalize()} model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading {regime} model: {e}")
            return False
    
    def load_all_regime_models(self) -> Dict[str, bool]:
        """Load all regime-specific models"""
        results = {}
        for regime in ['trending', 'ranging', 'volatile']:
            results[regime] = self.load_regime_specific_model(regime)
        
        loaded_count = sum(results.values())
        logger.info(f"📦 Loaded {loaded_count}/3 regime-specific models")
        return results
    
    def _update_regime_performance(self, training_results: Dict[str, bool]):
        """Update regime performance tracking after training"""
        for regime, success in training_results.items():
            if success:
                self.regime_performance[regime]['episodes'] += 1
                # Performance metrics will be updated during actual trading
        
        logger.info(f"📊 Updated regime performance tracking: {self.regime_performance}")
    
    def get_regime_model_info(self) -> Dict:
        """Get information about all regime-specific models"""
        info = {
            'regime_models_available': {},
            'current_regime': self.market_regime,
            'regime_performance': self.regime_performance
        }
        
        for regime in ['trending', 'ranging', 'volatile']:
            model_path = self.regime_model_paths[regime]
            model_loaded = self.regime_models[regime] is not None
            model_exists = os.path.exists(model_path)
            
            info['regime_models_available'][regime] = {
                'model_exists': model_exists,
                'model_loaded': model_loaded,
                'model_path': model_path
            }
            
            # Add metadata if available
            metadata_path = model_path.replace('.zip', '_metadata.pkl')
            if os.path.exists(metadata_path):
                try:
                    metadata = joblib.load(metadata_path)
                    info['regime_models_available'][regime]['metadata'] = metadata
                except Exception:
                    pass
        
        return info