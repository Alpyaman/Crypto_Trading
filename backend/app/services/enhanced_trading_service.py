"""
Enhanced Trading Service for Futures Trading
Advanced trading service with improved ML integration and risk management
"""
import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime
import uuid

from app.services.binance_service import BinanceService
from app.services.enhanced_ml_service import EnhancedMLService
try:
    from app.services.database_service import db_service
except ImportError:
    # Create a mock db_service if not available
    class MockDBService:
        def create_trade(self, data): 
            class MockTrade:
                def __init__(self): self.id = 1
            return MockTrade()
        def create_trading_session(self, data): 
            class MockSession:
                def to_dict(self): return data
            return MockSession()
        def update_trading_session(self, sid, updates): 
            class MockSession:
                def to_dict(self): return updates
            return MockSession()
    db_service = MockDBService()

logger = logging.getLogger(__name__)


class EnhancedTradingService:
    """Enhanced trading service with advanced ML and risk management"""
    
    def __init__(self, binance_service: BinanceService, enhanced_ml_service: EnhancedMLService, db_service=None):
        self.binance_service = binance_service
        self.ml_service = enhanced_ml_service
        self.db_service = db_service or globals().get('db_service')
        self.is_trading = False
        self.trading_task = None
        self.trading_history = []
        
        # ML observation configuration
        self.window_size = 50  # Must match EnhancedFuturesEnv window_size
        
        # Enhanced trading configuration
        self.current_position = {
            'symbol': None,
            'side': None,  # 'long' or 'short'
            'size': 0.0,
            'entry_price': 0.0,
            'leverage': 10,
            'margin_used': 0.0,
            'unrealized_pnl': 0.0,
            'entry_time': None
        }
        
        # Database integration
        self.trading_session_id = None
        
        # Risk management settings
        self.risk_config = {
            'max_position_size': 0.1,  # 10% of balance
            'stop_loss_pct': 0.05,     # 5% stop loss
            'take_profit_pct': 0.15,   # 15% take profit
            'max_leverage': 20,
            'min_confidence': 0.6,     # Minimum prediction confidence
            'max_daily_trades': 10,
            'max_drawdown': 0.2        # 20% max drawdown
        }
        
        # Performance tracking
        self.daily_trades = 0
        self.last_trade_date = None
        self.peak_balance = None
        self.max_drawdown = 0.0
        self.realized_pnl = 0.0
        self.trades_history = []
        
    async def start_enhanced_trading(self, 
                                   symbol: str = "BTCUSDT", 
                                   mode: str = "balanced", 
                                   leverage: int = 10) -> bool:
        """Start enhanced automated futures trading"""
        try:
            if self.is_trading:
                logger.warning("Enhanced trading is already active")
                return False
            
            if not self.ml_service.model:
                logger.error("Enhanced ML model not loaded")
                return False
            
            # Initialize trading session
            await self._initialize_trading_session(symbol, leverage)
            
            self.is_trading = True
            self.trading_task = asyncio.create_task(
                self._enhanced_trading_loop(symbol, mode)
            )
            
            logger.info(f"Started enhanced futures trading {symbol} in {mode} mode with {leverage}x leverage")
            return True
            
        except Exception as e:
            logger.error(f"Error starting enhanced trading: {e}")
            return False
    
    async def _initialize_trading_session(self, symbol: str, leverage: int):
        """Initialize trading session with enhanced setup"""
        try:
            # Set up futures trading
            self.binance_service.set_margin_type(symbol, 'CROSSED')
            self.binance_service.set_leverage(symbol, leverage)
            
            # Initialize position tracking
            self.current_position['symbol'] = symbol
            self.current_position['leverage'] = leverage
            
            # Get account balance for risk calculations
            account = self.binance_service.client.futures_account()
            self.peak_balance = float(account['totalWalletBalance'])
            
            # Reset daily counters
            self.daily_trades = 0
            self.last_trade_date = datetime.now().date()
            
            logger.info(f"Enhanced trading session initialized for {symbol}")
            
        except Exception as e:
            logger.error(f"Error initializing trading session: {e}")
            raise
    
    async def _enhanced_trading_loop(self, symbol: str, mode: str):
        """Enhanced trading loop with comprehensive market analysis"""
        loop_iteration = 0
        try:
            logger.info(f"🔄 Starting enhanced trading loop for {symbol} in {mode} mode")
            
            while self.is_trading:
                loop_iteration += 1
                logger.info(f"📊 Trading Loop #{loop_iteration} - Analyzing market for {symbol}")
                
                # Check daily trade limits
                current_date = datetime.now().date()
                if current_date != self.last_trade_date:
                    self.daily_trades = 0
                    self.last_trade_date = current_date
                    logger.info("🗓️ New trading day - Reset daily trade count")
                
                if self.daily_trades >= self.risk_config['max_daily_trades']:
                    logger.warning(f"⚠️ Daily trade limit reached ({self.daily_trades}/{self.risk_config['max_daily_trades']}) - Waiting 1 hour")
                    await asyncio.sleep(3600)  # Wait 1 hour
                    continue
                
                # Get current market data
                logger.info(f"💰 Fetching current price for {symbol}...")
                current_price = self.binance_service.get_current_price(symbol)
                if not current_price:
                    logger.error(f"❌ Failed to get current price for {symbol} - Retrying in 1 minute")
                    await asyncio.sleep(60)
                    continue
                
                logger.info(f"💵 Current {symbol} price: ${current_price:,.2f}")
                
                # Get enhanced market data for analysis
                logger.info("📈 Fetching enhanced market data...")
                market_data = await self._get_enhanced_market_data(symbol)
                if market_data is None:
                    logger.error("❌ Failed to get market data - Retrying in 1 minute")
                    await asyncio.sleep(60)
                    continue
                
                logger.info(f"✅ Market data loaded: {len(market_data)} data points")
                
                # Update position PnL
                await self._update_position_status(current_price)
                
                # Check risk management conditions
                logger.info("🛡️ Checking risk management conditions...")
                risk_check = await self._check_risk_conditions()
                if not risk_check:
                    logger.warning("⚠️ Risk conditions not met - Waiting 5 minutes")
                    await asyncio.sleep(300)  # Wait 5 minutes if risk conditions not met
                    continue

                logger.info("✅ Risk conditions passed")

                # Get ML prediction with enhanced features
                logger.info("🤖 Getting ML prediction...")
                prediction_result = await self._get_enhanced_prediction(market_data)
                if prediction_result is None:
                    logger.error("❌ Failed to get ML prediction - Waiting 5 minutes")
                    await asyncio.sleep(300)
                    continue
                
                action, confidence, position_size, analysis = prediction_result
                action_names = {0: 'CLOSE', 1: 'LONG', 2: 'SHORT', 3: 'HOLD'}
                action_name = action_names.get(action, 'UNKNOWN')
                
                # Add current price to analysis for position sizing
                analysis['current_price'] = current_price
                
                logger.info(f"🎯 ML Prediction: {action_name} (confidence: {confidence:.1%})")
                logger.info(f"📊 Market regime: {analysis.get('market_regime', 'Unknown')}")
                logger.info(f"📈 Risk score: {analysis.get('risk_score', 0):.3f}")
                
                # Execute trading decision
                if confidence >= self.risk_config['min_confidence']:
                    logger.info(f"✅ Confidence threshold met ({confidence:.1%} >= {self.risk_config['min_confidence']:.1%})")
                    await self._execute_enhanced_trading_decision(
                        symbol, action, confidence, position_size, current_price, analysis, mode
                    )
                else:
                    logger.info(f"⏸️ Skipping trade - Low confidence: {confidence:.1%} < {self.risk_config['min_confidence']:.1%}")
                
                # Log current status
                await self._log_trading_status(symbol, current_price, analysis)
                
                # Wait before next iteration
                logger.info("⏰ Waiting 5 minutes before next analysis...")
                await asyncio.sleep(300)  # 5 minutes
                
        except asyncio.CancelledError:
            logger.info("🛑 Enhanced trading loop cancelled")
        except Exception as e:
            logger.error(f"💥 Error in enhanced trading loop: {e}")
            self.is_trading = False
    
    async def _get_enhanced_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get comprehensive market data for analysis"""
        try:
            # Get historical klines
            klines = self.binance_service.get_historical_klines(
                symbol=symbol,
                interval='1h',
                limit=200  # More data for enhanced analysis
            )
            
            if not klines or len(klines) < 100:
                logger.warning("Insufficient market data")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(klines)
            
            # Extract enhanced features using ML service
            enhanced_df = self.ml_service.extract_enhanced_features(df, symbol)
            
            return enhanced_df
            
        except Exception as e:
            logger.error(f"Error getting enhanced market data: {e}")
            return None
    
    async def _get_enhanced_prediction(self, market_data: pd.DataFrame) -> Optional[Tuple]:
        """Get enhanced ML prediction with analysis"""
        try:
            # Prepare observation (last 50 data points)
            observation_data = market_data.tail(50)
            
            # Create observation vector (simplified for this example)
            observation = self._create_observation_vector(observation_data)
            
            if observation is None:
                return None
            
            # Get account balance for position sizing
            account = self.binance_service.client.futures_account()
            account_balance = float(account['totalWalletBalance'])
            
            # Get enhanced prediction
            action, confidence, position_size, analysis = self.ml_service.predict_enhanced(
                observation=observation,
                market_data=market_data,
                account_balance=account_balance,
                deterministic=True
            )
            
            return action, confidence, position_size, analysis
            
        except Exception as e:
            logger.error(f"Error getting enhanced prediction: {e}")
            return None
    
    def _safe_get_value(self, data_row, key: str, default: float = 0.0) -> float:
        """Safely get value from data row, handling None and missing keys"""
        try:
            value = data_row.get(key, default)
            if value is None or pd.isna(value):
                return default
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _create_observation_vector(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Create observation vector matching EXACT training format from EnhancedFuturesEnv
        
        The observation must match the training environment structure:
        - Market features (price action, trends, momentum, volume, futures-specific)
        - Position and portfolio features
        - Risk metrics
        - Market structure features
        
        Total size: 100 features (matching observation_space)
        """
        try:
            if len(data) < self.window_size:
                logger.warning(f"Insufficient data: need {self.window_size}, got {len(data)}")
                return None
            
            # Use the last row for current market state
            current_row = data.iloc[-1]
            current_price = self._safe_get_value(current_row, 'close', 1.0)
            
            features = []
            
            # === MARKET FEATURES (Price Action) ===
            # Price movements (4 features)
            features.extend([
                self._safe_get_value(current_row, 'price_change', 0.0),
                self._safe_get_value(current_row, 'volatility_ratio', 0.0),  
                self._safe_get_value(current_row, 'bb_position', 0.0),
                self._safe_get_value(current_row, 'bb_width', 0.0),
            ])
            
            # === TREND INDICATORS (4 features) ===
            features.extend([
                self._safe_get_value(current_row, 'sma_20', current_price) / current_price if current_price > 0 else 0.0,
                self._safe_get_value(current_row, 'ema_20', current_price) / current_price if current_price > 0 else 0.0,
                self._safe_get_value(current_row, 'macd', 0.0) / current_price if current_price > 0 else 0.0,
                self._safe_get_value(current_row, 'macd_histogram', 0.0) / current_price if current_price > 0 else 0.0,
            ])
            
            # === MOMENTUM INDICATORS (4 features) ===
            features.extend([
                self._safe_get_value(current_row, 'rsi_14', 50.0) / 100.0,
                self._safe_get_value(current_row, 'rsi_21', 50.0) / 100.0,
                self._safe_get_value(current_row, 'williams_r', -50.0) / 100.0,
                self._safe_get_value(current_row, 'stoch_k', 50.0) / 100.0,
            ])
            
            # === VOLUME INDICATORS (2 features) ===
            features.extend([
                self._safe_get_value(current_row, 'volume_ratio', 1.0),
                np.log1p(self._safe_get_value(current_row, 'volume', 1.0)) / 20.0,  # Normalized volume
            ])
            
            # === FUTURES-SPECIFIC FEATURES (4 features) ===
            features.extend([
                self._safe_get_value(current_row, 'funding_rate', 0.0001) * 10000,  # Scale up
                self._safe_get_value(current_row, 'long_short_ratio', 1.0),
                self._safe_get_value(current_row, 'open_interest_change', 0.0),
                self._safe_get_value(current_row, 'liquidation_pressure', 0.0),
            ])
            
            # === POSITION AND PORTFOLIO FEATURES (13 features) ===
            # Get current balance and position info
            balance = getattr(self, 'current_balance', 10000.0) or 10000.0
            initial_balance = 10000.0  # Match training environment
            position_size = self.current_position.get('size', 0.0) or 0.0
            margin_used = self.current_position.get('margin_used', 0.0) or 0.0
            unrealized_pnl = self.current_position.get('unrealized_pnl', 0.0) or 0.0
            realized_pnl = getattr(self, 'realized_pnl', 0.0) or 0.0
            
            total_value = balance + unrealized_pnl
            
            features.extend([
                # Portfolio state (4 features)
                balance / initial_balance,
                total_value / initial_balance,
                realized_pnl / initial_balance,
                unrealized_pnl / initial_balance,
                
                # Position state (3 features)
                position_size / initial_balance if initial_balance > 0 else 0.0,
                self.current_position.get('leverage', 10) / 20.0,  # Normalized by max leverage
                margin_used / balance if balance > 0 else 0.0,
                
                # Position details (3 features)
                1.0 if self.current_position.get('side') == 'long' else 0.0,
                1.0 if self.current_position.get('side') == 'short' else 0.0,
                abs(position_size) / (balance / current_price) if balance > 0 and current_price > 0 else 0.0,
                
                # Risk metrics (3 features)  
                getattr(self, 'max_drawdown', 0.0) or 0.0,
                ((total_value - (getattr(self, 'peak_balance', None) or initial_balance)) / 
                 (getattr(self, 'peak_balance', None) or initial_balance)) if (getattr(self, 'peak_balance', None) or initial_balance) > 0 else 0.0,
                len(getattr(self, 'trades_history', [])) / 100.0,  # Normalized trade count
            ])
            
            # === MARKET STRUCTURE FEATURES (4 features) ===
            # Support/Resistance and Trend analysis
            lookback = min(20, len(data))
            recent_data = data.tail(lookback)
            
            if len(recent_data) >= 10:
                support = recent_data['low'].min()
                resistance = recent_data['high'].max()
                
                # Support/Resistance (2 features)
                features.extend([
                    (current_price - support) / current_price if current_price > 0 else 0.0,
                    (resistance - current_price) / current_price if current_price > 0 else 0.0,
                ])
                
                # Trend strength (2 features)
                price_trend = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
                volume_trend = (recent_data['volume'].iloc[-1] - recent_data['volume'].iloc[0]) / recent_data['volume'].iloc[0]
                
                features.extend([
                    price_trend,
                    volume_trend,
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
            
            # === ADDITIONAL TECHNICAL FEATURES ===
            # Fill remaining features to reach exactly 100
            additional_features = []
            
            # Additional price features
            additional_features.extend([
                self._safe_get_value(current_row, 'high_low_ratio', 1.0) - 1.0,
                self._safe_get_value(current_row, 'open_close_ratio', 1.0) - 1.0,
                self._safe_get_value(current_row, 'volume_change', 0.0),
            ])
            
            # Additional moving averages
            for period in [10, 50]:
                sma_col = f'sma_{period}'
                ema_col = f'ema_{period}'
                additional_features.extend([
                    self._safe_get_value(current_row, sma_col, current_price) / current_price if current_price > 0 else 1.0,
                    self._safe_get_value(current_row, ema_col, current_price) / current_price if current_price > 0 else 1.0,
                ])
            
            # Additional momentum indicators
            additional_features.extend([
                self._safe_get_value(current_row, 'macd_signal', 0.0) / current_price if current_price > 0 else 0.0,
                self._safe_get_value(current_row, 'macd_momentum', 0.0),
                self._safe_get_value(current_row, 'atr_14', 0.0) / current_price if current_price > 0 else 0.0,
            ])
            
            # Time-based features
            additional_features.extend([
                self._safe_get_value(current_row, 'hour', 0) / 24.0,
                self._safe_get_value(current_row, 'day_of_week', 0) / 7.0,
                self._safe_get_value(current_row, 'is_weekend', 0),
            ])
            
            # Market regime indicators
            additional_features.extend([
                self._safe_get_value(current_row, 'higher_highs', 0),
                self._safe_get_value(current_row, 'lower_lows', 0),
                self._safe_get_value(current_row, 'trend_strength', 0),
            ])
            
            features.extend(additional_features)
            
            # === ENSURE EXACT SIZE ===
            expected_size = 100  # Must match EnhancedFuturesEnv observation_space
            
            if len(features) < expected_size:
                # Pad with zeros if needed
                features.extend([0.0] * (expected_size - len(features)))
            elif len(features) > expected_size:
                # Truncate if too many features
                features = features[:expected_size]
            
            # Final validation
            if len(features) != expected_size:
                raise ValueError(f"Feature mismatch: {len(features)} != {expected_size}")
            
            # Convert to numpy array with exact dtype
            observation = np.array(features, dtype=np.float32)
            
            # Validate no NaN or infinite values
            if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
                logger.warning("Observation contains NaN or infinite values, replacing with zeros")
                observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
            
            logger.debug(f"Created observation vector: shape={observation.shape}, dtype={observation.dtype}")
            return observation
            
        except Exception as e:
            logger.error(f"Error creating observation vector: {e}")
            # Return zero observation as fallback
            return np.zeros(100, dtype=np.float32)
    
    async def _execute_enhanced_trading_decision(self,
                                               symbol: str,
                                               action: int,
                                               confidence: float,
                                               position_size: float,
                                               current_price: float,
                                               analysis: Dict,
                                               mode: str):
        """Execute enhanced trading decision with comprehensive logging and risk management"""
        try:
            action_names = {0: 'CLOSE', 1: 'LONG', 2: 'SHORT', 3: 'HOLD'}
            action_name = action_names.get(action, 'UNKNOWN')
            
            logger.info("🎯 EXECUTING TRADING DECISION")
            logger.info(f"   Action: {action_name} | Confidence: {confidence:.1%}")
            logger.info(f"   Position Size: {position_size:.6f} | Price: ${current_price:,.2f}")
            logger.info(f"   Market Regime: {analysis.get('market_regime', 'Unknown')}")
            logger.info(f"   Risk Score: {analysis.get('risk_score', 0):.3f}")
            
            # Action mapping: 0=Close, 1=Long, 2=Short, 3=Hold
            if action == 0:  # Close Position
                if self.current_position['side']:
                    logger.info(f"🔄 Closing {self.current_position['side']} position")
                    await self._close_current_position(symbol, "ML_SIGNAL")
                else:
                    logger.info("ℹ️ No position to close")
                
            elif action == 1:  # Long Position
                if self.current_position['side'] == 'long':
                    logger.info("ℹ️ Already in LONG position - no action needed")
                else:
                    # Close any short position first
                    if self.current_position['side'] == 'short':
                        logger.info("🔄 Switching from SHORT to LONG - closing short first")
                        await self._close_current_position(symbol, "POSITION_SWITCH")
                    
                    # Open long position
                    logger.info("📈 Opening LONG position")
                    await self._open_long_position(symbol, position_size, current_price, confidence, analysis)
                    
            elif action == 2:  # Short Position
                if self.current_position['side'] == 'short':
                    logger.info("ℹ️ Already in SHORT position - no action needed")
                else:
                    # Close any long position first
                    if self.current_position['side'] == 'long':
                        logger.info("🔄 Switching from LONG to SHORT - closing long first")
                        await self._close_current_position(symbol, "POSITION_SWITCH")
                    
                    # Open short position
                    logger.info("📉 Opening SHORT position")
                    await self._open_short_position(symbol, position_size, current_price, confidence, analysis)
            
            elif action == 3:  # Hold
                logger.info("⏸️ HOLD signal - no trading action")
            
        except Exception as e:
            logger.error(f"💥 Error executing enhanced trading decision: {e}")
    
    async def _open_long_position(self, symbol: str, position_size: float, current_price: float, 
                                confidence: float, analysis: Dict):
        """Open enhanced long position with risk management"""
        try:
            # Adjust position size based on risk management
            adjusted_size = self._adjust_position_size(position_size, confidence, analysis)
            
            if adjusted_size <= 0:
                logger.info("Position size too small after risk adjustment")
                return
            
            # Place futures order
            order_result = self.binance_service.place_futures_order(
                symbol=symbol,
                side='BUY',
                order_type='MARKET',
                quantity=adjusted_size
            )
            
            if order_result:
                # Update position tracking
                self.current_position.update({
                    'side': 'long',
                    'size': adjusted_size,
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'confidence': confidence,
                    'analysis': analysis
                })
                
                # Record trade to database
                trade_data = {
                    'symbol': symbol,
                    'side': 'BUY',
                    'quantity': adjusted_size,
                    'price': current_price,
                    'order_id': order_result.get('orderId'),
                    'position_size': position_size,
                    'commission': order_result.get('commission', 0.0)
                }
                self._record_trade_to_database(trade_data)
                
                # Set stop loss and take profit
                await self._set_position_risk_management(symbol, current_price, 'long')
                
                self.daily_trades += 1
                
                logger.info(f"Opened long position: {adjusted_size} at ${current_price:.2f}")
                
        except Exception as e:
            logger.error(f"Error opening long position: {e}")
    
    async def _open_short_position(self, symbol: str, position_size: float, current_price: float,
                                 confidence: float, analysis: Dict):
        """Open enhanced short position with risk management"""
        try:
            # Adjust position size based on risk management
            adjusted_size = self._adjust_position_size(position_size, confidence, analysis)
            
            if adjusted_size <= 0:
                logger.info("Position size too small after risk adjustment")
                return
            
            # Place futures order
            order_result = self.binance_service.place_futures_order(
                symbol=symbol,
                side='SELL',
                order_type='MARKET',
                quantity=adjusted_size
            )
            
            if order_result:
                # Update position tracking
                self.current_position.update({
                    'side': 'short',
                    'size': adjusted_size,
                    'entry_price': current_price,
                    'entry_time': datetime.now(),
                    'confidence': confidence,
                    'analysis': analysis
                })
                
                # Set stop loss and take profit
                await self._set_position_risk_management(symbol, current_price, 'short')
                
                self.daily_trades += 1
                
                logger.info(f"Opened short position: {adjusted_size} at ${current_price:.2f}")
                
        except Exception as e:
            logger.error(f"Error opening short position: {e}")
    
    def _adjust_position_size(self, base_size: float, confidence: float, analysis: Dict) -> float:
        """
        Position sizing with comprehensive risk management
        Implements multi-layer risk controls to prevent catastrophic losses
        """
        try:
            current_price = analysis.get('current_price', 115000)  # Fallback price
            
            # 1. ACCOUNT-LEVEL RISK LIMITS
            account_balance = self._get_account_balance()
            if account_balance <= 0:
                logger.error("Invalid account balance for position sizing")
                return 0.0
            
            max_portfolio_risk = account_balance * self.risk_config.get('max_portfolio_risk', 0.20)  # 20% max
            current_exposure = self._calculate_current_exposure()
            
            if current_exposure >= max_portfolio_risk:
                logger.warning(f"Maximum portfolio risk reached: ${current_exposure:.2f} >= ${max_portfolio_risk:.2f}")
                return 0.0
            
            # Calculate remaining risk budget
            remaining_risk_budget = max_portfolio_risk - current_exposure
            
            # 2. DRAWDOWN-BASED POSITION REDUCTION
            current_drawdown = self._calculate_current_drawdown()
            drawdown_reduction = 1.0
            
            if current_drawdown > 0.05:  # 5% drawdown threshold
                if current_drawdown > 0.20:  # Severe drawdown
                    logger.warning(f"Severe drawdown detected: {current_drawdown:.1%} - Halting trading")
                    return 0.0
                elif current_drawdown > 0.10:  # Moderate drawdown
                    drawdown_reduction = max(0.3, 1.0 - (current_drawdown * 3))  # Reduce up to 70%
                    logger.warning(f"Moderate drawdown: {current_drawdown:.1%} - Reducing position size by {(1-drawdown_reduction):.1%}")
                else:  # Minor drawdown
                    drawdown_reduction = max(0.5, 1.0 - (current_drawdown * 2))  # Reduce up to 50%
                    logger.info(f"Minor drawdown: {current_drawdown:.1%} - Reducing position size by {(1-drawdown_reduction):.1%}")
            
            # 3. KELLY CRITERION WITH HALF-KELLY SAFETY
            kelly_size = self._calculate_kelly_position_size(analysis, confidence)
            kelly_size *= 0.5  # Use half-kelly for risk management
            
            # 4. LEVERAGE-ADJUSTED RISK
            leverage = analysis.get('leverage', self.current_position.get('leverage', 10))
            leverage_risk_adjustment = min(1.0, 20.0 / leverage)  # Reduce size for higher leverage
            
            # 5. VOLATILITY-BASED ADJUSTMENT
            volatility_adjustment = self._get_volatility_adjustment(analysis)
            
            # 6. MARKET REGIME ADJUSTMENT
            regime_multiplier = self._get_regime_multiplier(analysis)
            
            # 7. APPLY ALL ADJUSTMENTS
            adjusted_size = min(base_size, kelly_size)
            adjusted_size *= drawdown_reduction
            adjusted_size *= leverage_risk_adjustment
            adjusted_size *= volatility_adjustment
            adjusted_size *= regime_multiplier
            
            # Confidence adjustment with diminishing returns
            confidence_multiplier = min(confidence * 1.2 + 0.3, 1.0)
            adjusted_size *= confidence_multiplier
            
            # 8. TRANSACTION COST AND SLIPPAGE ADJUSTMENTS
            transaction_costs = self._calculate_total_transaction_costs(adjusted_size, current_price, analysis)
            expected_slippage = self._estimate_slippage(adjusted_size, current_price, analysis)
            
            cost_adjustment = max(0.7, 1.0 - (transaction_costs * 100))
            slippage_adjustment = max(0.8, 1.0 - (expected_slippage * 50))
            
            adjusted_size *= cost_adjustment
            adjusted_size *= slippage_adjustment
            
            # 9. HARD POSITION LIMITS
            min_size = self._get_minimum_position_size()
            max_size = self._get_maximum_position_size(account_balance, current_price)
            
            final_size = np.clip(adjusted_size, min_size, max_size)
            
            # Apply portfolio risk budget constraint
            max_position_value_by_budget = remaining_risk_budget
            max_size_by_budget = max_position_value_by_budget / current_price
            
            if final_size * current_price > remaining_risk_budget:
                logger.warning(f"Position would exceed remaining risk budget: ${final_size * current_price:.2f} > ${remaining_risk_budget:.2f}")
                final_size = max(0, max_size_by_budget)
                
                # If even minimum position exceeds budget, reject
                if final_size < min_size:
                    logger.warning("Even minimum position would exceed portfolio risk budget")
                    return 0.0
            
            # 10. LIQUIDATION SAFETY VALIDATION
            if not self._validate_liquidation_safety(final_size, current_price, leverage):
                logger.warning("Position would exceed liquidation safety threshold")
                return 0.0
            
            # 11. CONCENTRATION RISK CHECK
            if not self._validate_concentration_limits(final_size, current_price):
                logger.warning("Position would exceed concentration limits")
                return 0.0
            
            # 12. CORRELATION RISK ASSESSMENT
            if not self._validate_correlation_risk(final_size):
                logger.warning("Position would increase correlation risk beyond limits")
                final_size *= 0.5  # Reduce instead of blocking
            
            # Comprehensive logging
            self._log_position_sizing_breakdown({
                'account_balance': account_balance,
                'base_size': base_size,
                'kelly_size': kelly_size,
                'final_size': final_size,
                'current_drawdown': current_drawdown,
                'drawdown_reduction': drawdown_reduction,
                'leverage_adjustment': leverage_risk_adjustment,
                'volatility_adjustment': volatility_adjustment,
                'regime_multiplier': regime_multiplier,
                'confidence_multiplier': confidence_multiplier,
                'transaction_costs': transaction_costs,
                'expected_slippage': expected_slippage,
                'current_exposure': current_exposure,
                'max_portfolio_risk': max_portfolio_risk
            })
            
            return final_size
            
        except Exception as e:
            logger.error(f"Critical error in position sizing: {e}")
            # Emergency fallback - minimal position
            return max(0.001, base_size * 0.1)
    
    def _calculate_kelly_position_size(self, analysis: Dict, confidence: float) -> float:
        """Calculate optimal position size using Kelly Criterion based on current Binance futures balance"""
        try:
            # Get real-time account balance from Binance futures
            # Check if we're in a test environment or have a mocked service
            if hasattr(self.binance_service, 'get_futures_account_balance'):
                # Use the service method (works with mocks)
                balance_info = self.binance_service.get_futures_account_balance()
                current_balance = float(balance_info['totalWalletBalance'])
                available_balance = float(balance_info['availableBalance'])
            else:
                # Fallback to direct client call (production)
                account = self.binance_service.client.futures_account()
                current_balance = float(account['totalWalletBalance'])
                available_balance = float(account['availableBalance'])
            
            logger.info(f"💰 Current Futures Account Balance: ${current_balance:.2f}")
            logger.info(f"💰 Available Balance: ${available_balance:.2f}")
            
            # Historical performance data adjusted by ML prediction and confidence
            base_win_rate = analysis.get('historical_win_rate', 0.52)  # Base historical win rate
            ml_prediction = analysis.get('prediction', 0.5)  # ML prediction (0.5 = neutral)
            
            # Adjust win rate based on ML prediction and confidence
            # Higher confidence in bullish prediction increases effective win rate
            # Lower confidence or bearish prediction decreases it
            prediction_adjustment = (ml_prediction - 0.5) * confidence  # -0.5 to +0.5 range
            adjusted_win_rate = base_win_rate + prediction_adjustment
            adjusted_win_rate = max(0.1, min(0.9, adjusted_win_rate))  # Keep within reasonable bounds
            
            avg_win = analysis.get('avg_win_ratio', 0.08)  # 8% average win
            avg_loss = analysis.get('avg_loss_ratio', 0.06)  # 6% average loss
            
            # Kelly formula: f = (bp - q) / b
            # where: b = odds (avg_win/avg_loss), p = win probability, q = loss probability
            if avg_loss > 0:
                b = avg_win / avg_loss  # Odds ratio
                p = adjusted_win_rate  # Adjusted win probability based on ML prediction
                q = 1 - p     # Loss probability
                
                kelly_fraction = (b * p - q) / b
                
                # Apply confidence as a multiplier with more dynamic range
                # Lower confidence should significantly reduce position size
                if confidence >= 0.90:
                    confidence_factor = 0.8 + (confidence - 0.90) * 2.0  # 80% to 100% of Kelly
                elif confidence >= 0.80:
                    confidence_factor = 0.6 + (confidence - 0.80) * 2.0  # 60% to 80% of Kelly
                elif confidence >= 0.70:
                    confidence_factor = 0.4 + (confidence - 0.70) * 2.0  # 40% to 60% of Kelly
                else:
                    confidence_factor = 0.2 + (confidence - 0.60) * 2.0  # 20% to 40% of Kelly
                
                kelly_fraction *= confidence_factor
                
                # Cap at reasonable limits (never more than 25% of balance)
                kelly_fraction = max(0.001, min(0.25, kelly_fraction))
                
                # Convert to BTC amount using available balance (more conservative)
                current_price = analysis.get('current_price', 115000)
                leverage = analysis.get('leverage', 10)  # Default 10x leverage
                
                # Use available balance for position sizing (safer than total balance)
                balance_for_sizing = min(current_balance, available_balance)
                
                # Kelly size calculation (this is the margin requirement, not the position size)
                kelly_margin_value = kelly_fraction * balance_for_sizing
                kelly_size = kelly_margin_value / current_price  # BTC margin requirement
                
                # Apply Binance minimum trading size (0.001 BTC for BTCUSDT) - this is position size, not margin
                binance_min_position_size = 0.001
                
                # For small accounts, ensure minimum viable position but respect Kelly limits
                # Adjust minimum position for very small accounts
                if balance_for_sizing < 100:  # Very small accounts
                    min_position_value = max(5.0, balance_for_sizing * 0.15)  # 15% of balance, min $5
                else:
                    min_position_value = 10.0  # Standard $10 minimum
                
                # This is the margin requirement for the minimum position
                min_margin_value = min_position_value / leverage  # Margin needed
                min_kelly_size = min_margin_value / current_price  # BTC margin requirement
                
                # Apply Binance minimum trading size (0.001 BTC for BTCUSDT) - this is position size, not margin
                binance_min_position_size = 0.001
                
                # Calculate maximum position size (adjust for small accounts)
                if balance_for_sizing < 100:
                    max_position_percentage = 0.25  # Allow up to 25% for small accounts
                else:
                    max_position_percentage = 0.20  # Standard 20%
                
                # Maximum margin we can use
                max_margin_value = balance_for_sizing * max_position_percentage
                max_margin_size = max_margin_value / current_price  # BTC margin limit
                
                # Position sizing logic with leverage consideration
                if balance_for_sizing < 100:
                    # For very small accounts, prioritize Kelly calculation but ensure minimum viability
                    if kelly_size >= (binance_min_position_size / leverage):  # Kelly suggests viable position
                        final_margin_size = kelly_size  # Use Kelly calculation
                    else:
                        # Kelly suggests too small position, use minimum viable
                        final_margin_size = max(kelly_size, min_kelly_size)
                    
                    # Ensure we meet Binance minimum position size (considering leverage)
                    min_required_margin = binance_min_position_size / leverage
                    final_margin_size = max(final_margin_size, min_required_margin)
                    
                    # Cap at maximum margin
                    final_margin_size = min(final_margin_size, max_margin_size)
                else:
                    # Standard calculation for larger accounts
                    min_required_margin = binance_min_position_size / leverage
                    final_margin_size = max(kelly_size, min_kelly_size, min_required_margin)
                    final_margin_size = min(final_margin_size, max_margin_size)
                
                # The final_margin_size is what we return (BTC margin requirement)
                # The actual position size will be final_margin_size * leverage
                
                # Calculate actual position size and values for logging
                actual_position_size = final_margin_size * leverage
                margin_value_used = final_margin_size * current_price
                actual_position_value = actual_position_size * current_price
                
                logger.info("📊 Kelly Criterion Calculation:")
                logger.info(f"   Account Balance: ${current_balance:.2f}")
                logger.info(f"   Available Balance: ${available_balance:.2f}")
                logger.info(f"   Balance for Sizing: ${balance_for_sizing:.2f}")
                logger.info(f"   Leverage: {leverage}x")
                logger.info(f"   Base Win Rate: {base_win_rate:.3f}")
                logger.info(f"   ML Prediction: {ml_prediction:.3f}")
                logger.info(f"   Confidence: {confidence:.3f}")
                logger.info(f"   Prediction Adjustment: {prediction_adjustment:+.3f}")
                logger.info(f"   Adjusted Win Rate: {adjusted_win_rate:.3f}")
                logger.info(f"   Odds Ratio (b): {b:.3f}")
                logger.info(f"   Confidence Factor: {confidence_factor:.3f}")
                logger.info(f"   Kelly Fraction: {kelly_fraction:.4f} ({kelly_fraction*100:.2f}%)")
                logger.info(f"   Raw Kelly Margin: {kelly_size:.6f} BTC (${kelly_margin_value:.2f})")
                logger.info(f"   Minimum Margin: {min_kelly_size:.6f} BTC (${min_margin_value:.2f})")
                logger.info(f"   Binance Min Position: {binance_min_position_size:.6f} BTC")
                logger.info(f"   Max Margin ({max_position_percentage*100:.0f}%): {max_margin_size:.6f} BTC (${max_margin_value:.2f})")
                logger.info(f"   Final Margin Size: {final_margin_size:.6f} BTC (${margin_value_used:.2f})")
                logger.info(f"   Actual Position Size: {actual_position_size:.6f} BTC (${actual_position_value:.2f})")
                
                return final_margin_size
            
            return 0.001  # Minimum fallback
            
        except Exception as e:
            logger.error(f"Error calculating Kelly size: {e}")
            return 0.001
    
    def _calculate_total_transaction_costs(self, position_size: float, price: float, analysis: Dict) -> float:
        """Calculate comprehensive transaction costs including fees, spread, and market impact"""
        try:
            # Base trading fee (Binance futures: 0.02% maker, 0.04% taker)
            trading_fee = 0.0004  # Assume taker fee for market orders
            
            # Bid-ask spread cost (typically 0.01-0.03% for BTCUSDT)
            spread_cost = analysis.get('spread_percentage', 0.0002)  # 0.02% default
            
            # Market impact cost (function of order size relative to order book depth)
            volume_24h = analysis.get('volume_24h', 100000)  # 24h volume in BTC
            position_value = position_size * price
            
            # Market impact increases with order size relative to volume
            volume_ratio = position_value / (volume_24h * price) if volume_24h > 0 else 0.001
            market_impact = min(0.001, volume_ratio * 0.1)  # Cap at 0.1%
            
            # Funding rate impact (for futures positions held overnight)
            funding_rate = analysis.get('funding_rate', 0.0001)  # 0.01% typical
            expected_hold_time = analysis.get('expected_hold_hours', 4)  # 4 hours default
            funding_cost = abs(funding_rate) * (expected_hold_time / 8)  # 8-hour funding periods
            
            # Slippage tolerance (additional buffer)
            slippage_buffer = 0.0001  # 0.01% buffer
            
            total_cost = trading_fee + spread_cost + market_impact + funding_cost + slippage_buffer
            
            logger.info("💸 Transaction Cost Breakdown:")
            logger.info(f"   Trading Fee: {trading_fee:.4f} ({trading_fee*100:.2f}%)")
            logger.info(f"   Spread Cost: {spread_cost:.4f} ({spread_cost*100:.2f}%)")
            logger.info(f"   Market Impact: {market_impact:.4f} ({market_impact*100:.2f}%)")
            logger.info(f"   Funding Cost: {funding_cost:.4f} ({funding_cost*100:.2f}%)")
            logger.info(f"   Total Cost: {total_cost:.4f} ({total_cost*100:.2f}%)")
            
            return total_cost
            
        except Exception as e:
            logger.error(f"Error calculating transaction costs: {e}")
            return 0.001  # 0.1% fallback
    
    def _estimate_slippage(self, position_size: float, price: float, analysis: Dict) -> float:
        """Estimate expected slippage based on market conditions and order size"""
        try:
            # Order book depth analysis
            order_book_depth = analysis.get('order_book_depth', 1000000)  # USD depth
            position_value = position_size * price
            
            # Slippage increases with order size relative to book depth
            depth_ratio = position_value / order_book_depth if order_book_depth > 0 else 0.01
            
            # Base slippage (minimum expected slippage)
            base_slippage = 0.0001  # 0.01% minimum
            
            # Size-based slippage (square root function for diminishing impact)
            size_slippage = min(0.002, depth_ratio ** 0.5 * 0.01)  # Cap at 0.2%
            
            # Volatility-based slippage (higher volatility = more slippage)
            volatility = analysis.get('volatility', 0.02)
            volatility_slippage = min(0.001, (volatility - 0.01) * 0.05) if volatility > 0.01 else 0
            
            # Time of day adjustment (higher slippage during low liquidity periods)
            # This would ideally use real-time liquidity data
            time_adjustment = 1.0  # Placeholder for time-based liquidity
            
            total_slippage = (base_slippage + size_slippage + volatility_slippage) * time_adjustment
            
            logger.info("📈 Slippage Estimation:")
            logger.info(f"   Position Value: ${position_value:,.2f}")
            logger.info(f"   Book Depth: ${order_book_depth:,.2f}")
            logger.info(f"   Depth Ratio: {depth_ratio:.6f}")
            logger.info(f"   Expected Slippage: {total_slippage:.4f} ({total_slippage*100:.2f}%)")
            
            return total_slippage
            
        except Exception as e:
            logger.error(f"Error estimating slippage: {e}")
            return 0.0002  # 0.02% fallback
    
    def _get_account_balance(self) -> float:
        """Get current account balance from Binance futures"""
        try:
            if hasattr(self.binance_service, 'get_futures_account_balance'):
                balance_info = self.binance_service.get_futures_account_balance()
                return float(balance_info['totalWalletBalance'])
            else:
                account = self.binance_service.client.futures_account()
                return float(account['totalWalletBalance'])
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return 0.0
    
    def _calculate_current_exposure(self) -> float:
        """Calculate total current exposure across all positions"""
        try:
            if hasattr(self.binance_service, 'get_futures_positions'):
                positions = self.binance_service.get_futures_positions()
            else:
                positions = self.binance_service.client.futures_position_information()
            
            total_exposure = 0.0
            for position in positions:
                if float(position['positionAmt']) != 0:
                    notional = abs(float(position['notional']))
                    total_exposure += notional
            
            return total_exposure
        except Exception as e:
            logger.error(f"Error calculating current exposure: {e}")
            return 0.0
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak equity"""
        try:
            current_balance = self._get_account_balance()
            
            # Initialize or update peak equity
            if not hasattr(self, '_peak_equity'):
                self._peak_equity = current_balance
            
            self._peak_equity = max(self._peak_equity, current_balance)
            
            if self._peak_equity <= 0:
                return 0.0
            
            drawdown = (self._peak_equity - current_balance) / self._peak_equity
            return max(0.0, drawdown)
            
        except Exception as e:
            logger.error(f"Error calculating drawdown: {e}")
            return 0.0
    
    def _get_volatility_adjustment(self, analysis: Dict) -> float:
        """Calculate position adjustment based on market volatility"""
        try:
            volatility = analysis.get('volatility', 0.02)
            base_vol = 0.02  # 2% daily volatility baseline
            
            if volatility <= base_vol:
                return 1.0
            
            # Reduce position size for higher volatility
            vol_ratio = volatility / base_vol
            adjustment = max(0.3, 1.0 / (1.0 + (vol_ratio - 1.0) * 2))
            
            return adjustment
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {e}")
            return 0.5
    
    def _get_regime_multiplier(self, analysis: Dict) -> float:
        """Get position size multiplier based on market regime"""
        try:
            regime = analysis.get('market_regime', 'unknown')
            
            regime_multipliers = {
                'trending': 1.0,     # Full size in clear trends
                'ranging': 0.6,      # Reduced size in sideways markets
                'volatile': 0.4,     # Much smaller in volatile/uncertain markets
                'bear_market': 0.3,  # Very conservative in bear markets
                'unknown': 0.5       # Conservative for unknown regimes
            }
            
            return regime_multipliers.get(regime, 0.5)
            
        except Exception as e:
            logger.error(f"Error getting regime multiplier: {e}")
            return 0.5
    
    def _get_minimum_position_size(self) -> float:
        """Get minimum viable position size"""
        return 0.001  # Binance minimum for BTCUSDT futures
    
    def _get_maximum_position_size(self, account_balance: float, current_price: float = None) -> float:
        """Get maximum allowed position size based on risk limits"""
        try:
            max_position_percentage = self.risk_config.get('max_position_size', 0.20)
            price = current_price or 115000  # Use provided price or fallback
            
            max_position_value = account_balance * max_position_percentage
            return max_position_value / price
            
        except Exception as e:
            logger.error(f"Error calculating maximum position size: {e}")
            return 0.001
    
    def _validate_liquidation_safety(self, size: float, price: float, leverage: float) -> bool:
        """Validate that position won't risk liquidation"""
        try:
            account_balance = self._get_account_balance()
            position_value = size * price
            required_margin = position_value / leverage
            
            # Ensure we have 3x the required margin as safety buffer
            safety_margin = required_margin * 3
            
            return account_balance >= safety_margin
            
        except Exception as e:
            logger.error(f"Error validating liquidation safety: {e}")
            return False
    
    def _validate_concentration_limits(self, size: float, price: float) -> bool:
        """Validate position doesn't exceed concentration limits"""
        try:
            account_balance = self._get_account_balance()
            position_value = size * price
            
            # No single position should exceed 30% of account
            max_concentration = account_balance * 0.30
            
            return position_value <= max_concentration
            
        except Exception as e:
            logger.error(f"Error validating concentration limits: {e}")
            return False
    
    def _validate_correlation_risk(self, size: float) -> bool:
        """Validate that position doesn't increase correlation risk excessively"""
        try:
            current_exposure = self._calculate_current_exposure()
            account_balance = self._get_account_balance()
            
            # If total exposure would exceed 50% of account, flag correlation risk
            max_correlation_exposure = account_balance * 0.50
            
            return current_exposure <= max_correlation_exposure
            
        except Exception as e:
            logger.error(f"Error validating correlation risk: {e}")
            return True
    
    def _log_position_sizing_breakdown(self, metrics: Dict) -> None:
        """Log detailed position sizing breakdown for analysis"""
        try:
            logger.info("🎯 COMPREHENSIVE POSITION SIZING ANALYSIS:")
            logger.info(f"   💰 Account Balance: ${metrics['account_balance']:.2f}")
            logger.info(f"   📊 Base → Kelly → Final: {metrics['base_size']:.6f} → {metrics['kelly_size']:.6f} → {metrics['final_size']:.6f} BTC")
            
            # Risk adjustments
            logger.info(f"   🔻 Drawdown: {metrics['current_drawdown']:.1%} (reduction: {1-metrics['drawdown_reduction']:.1%})")
            logger.info(f"   ⚖️  Leverage Adj: {metrics['leverage_adjustment']:.3f}")
            logger.info(f"   📈 Volatility Adj: {metrics['volatility_adjustment']:.3f}")
            logger.info(f"   🌊 Regime Mult: {metrics['regime_multiplier']:.3f}")
            logger.info(f"   🎯 Confidence Mult: {metrics['confidence_multiplier']:.3f}")
            
            # Costs
            logger.info(f"   💸 Transaction Costs: {metrics['transaction_costs']:.4f}")
            logger.info(f"   🌊 Expected Slippage: {metrics['expected_slippage']:.4f}")
            
            # Risk limits
            logger.info(f"   🔥 Current Exposure: ${metrics['current_exposure']:.2f} / ${metrics['max_portfolio_risk']:.2f} max")
            
            # Final validation
            position_value = metrics['final_size'] * 115000
            risk_percentage = (position_value / metrics['account_balance']) * 100
            logger.info(f"   ✅ Final Position Risk: {risk_percentage:.1f}% of account")
            
        except Exception as e:
            logger.error(f"Error logging position sizing breakdown: {e}")
    
    async def _set_position_risk_management(self, symbol: str, entry_price: float, side: str):
        """Set stop loss and take profit orders"""
        try:
            if side == 'long':
                stop_price = entry_price * (1 - self.risk_config['stop_loss_pct'])
                take_profit_price = entry_price * (1 + self.risk_config['take_profit_pct'])
            else:  # short
                stop_price = entry_price * (1 + self.risk_config['stop_loss_pct'])
                take_profit_price = entry_price * (1 - self.risk_config['take_profit_pct'])
            
            # Note: In production, you would set actual stop loss and take profit orders
            # For now, we'll track them and check manually
            self.current_position.update({
                'stop_loss': stop_price,
                'take_profit': take_profit_price
            })
            
            logger.info(f"Risk management set: SL=${stop_price:.2f}, TP=${take_profit_price:.2f}")
            
        except Exception as e:
            logger.error(f"Error setting risk management: {e}")
    
    async def _close_current_position(self, symbol: str, reason: str = "ML_SIGNAL"):
        """Close current position"""
        try:
            if self.current_position['side'] is None:
                return
            
            # Determine order side
            order_side = 'SELL' if self.current_position['side'] == 'long' else 'BUY'
            
            # Place close order
            order_result = self.binance_service.place_futures_order(
                symbol=symbol,
                side=order_side,
                order_type='MARKET',
                quantity=self.current_position['size']
            )
            
            if order_result:
                # Calculate PnL
                current_price = self.binance_service.get_current_price(symbol)
                pnl = self._calculate_position_pnl(current_price)
                
                # Record trade
                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': symbol,
                    'side': self.current_position['side'],
                    'size': self.current_position['size'],
                    'entry_price': self.current_position['entry_price'],
                    'exit_price': current_price,
                    'pnl': pnl,
                    'confidence': self.current_position.get('confidence', 0),
                    'reason': reason
                }
                
                self.trading_history.append(trade_record)
                
                # Reset position
                self.current_position.update({
                    'side': None,
                    'size': 0.0,
                    'entry_price': 0.0,
                    'margin_used': 0.0,
                    'unrealized_pnl': 0.0,
                    'entry_time': None
                })
                
                logger.info(f"Closed {trade_record['side']} position: PnL=${pnl:.2f}, Reason={reason}")
                
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    async def _update_position_status(self, current_price: float):
        """Update current position status and check risk conditions"""
        try:
            if self.current_position['side'] is None:
                return
            
            # Update unrealized PnL
            pnl = self._calculate_position_pnl(current_price)
            self.current_position['unrealized_pnl'] = pnl
            
            # Check stop loss and take profit
            stop_loss = self.current_position.get('stop_loss')
            take_profit = self.current_position.get('take_profit')
            
            if stop_loss and take_profit:
                if self.current_position['side'] == 'long':
                    if current_price <= stop_loss:
                        await self._close_current_position(self.current_position['symbol'], "STOP_LOSS")
                    elif current_price >= take_profit:
                        await self._close_current_position(self.current_position['symbol'], "TAKE_PROFIT")
                else:  # short
                    if current_price >= stop_loss:
                        await self._close_current_position(self.current_position['symbol'], "STOP_LOSS")
                    elif current_price <= take_profit:
                        await self._close_current_position(self.current_position['symbol'], "TAKE_PROFIT")
            
        except Exception as e:
            logger.error(f"Error updating position status: {e}")
    
    def _calculate_position_pnl(self, current_price: float) -> float:
        """Calculate current position PnL"""
        if self.current_position['side'] is None or self.current_position['size'] == 0:
            return 0.0
        
        entry_price = self.current_position['entry_price']
        size = self.current_position['size']
        
        if self.current_position['side'] == 'long':
            return (current_price - entry_price) * size
        else:  # short
            return (entry_price - current_price) * size
    
    async def _check_risk_conditions(self) -> bool:
        """Check if trading should continue based on comprehensive risk conditions"""
        try:
            # Get current account balance from Binance futures
            account = self.binance_service.client.futures_account()
            current_balance = float(account['totalWalletBalance'])
            available_balance = float(account['availableBalance'])
            total_margin = float(account.get('totalPositionInitialMargin', 0))

            logger.info("🛡️ RISK MANAGEMENT CHECK")
            logger.info(f"   Current Balance: ${current_balance:.2f}")
            logger.info(f"   Available Balance: ${available_balance:.2f}")
            logger.info(f"   Used Margin: ${total_margin:.2f}")
            
            # Initialize peak balance if not set or update with current balance
            if self.peak_balance is None or current_balance > self.peak_balance:
                old_peak = self.peak_balance
                self.peak_balance = current_balance
                if old_peak is None:
                    logger.info(f"   Peak Balance Initialized: ${self.peak_balance:.2f}")
                else:
                    logger.info(f"   🎉 New Peak Balance: ${old_peak:.2f} → ${self.peak_balance:.2f}")
            
            # Check maximum drawdown
            current_drawdown = 0.0
            if self.peak_balance and current_balance < self.peak_balance:
                current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
                self.max_drawdown = max(self.max_drawdown, current_drawdown)
                
                logger.info(f"   Current Drawdown: {current_drawdown:.1%}")
                logger.info(f"   Max Drawdown: {self.max_drawdown:.1%}")
                logger.info(f"   Drawdown Limit: {self.risk_config['max_drawdown']:.1%}")
                
                if current_drawdown > self.risk_config['max_drawdown']:
                    logger.error(f"❌ RISK CHECK FAILED: Maximum drawdown exceeded ({current_drawdown:.1%} > {self.risk_config['max_drawdown']:.1%})")
                    return False
            
            # Check minimum balance for trading
            min_balance = 10.0  # Minimum $10 to continue trading
            if available_balance < min_balance:
                logger.error(f"❌ RISK CHECK FAILED: Insufficient available balance (${available_balance:.2f} < ${min_balance:.2f})")
                return False
            
            # Check daily trade limit
            if self.daily_trades >= self.risk_config['max_daily_trades']:
                logger.error(f"❌ RISK CHECK FAILED: Daily trade limit reached ({self.daily_trades}/{self.risk_config['max_daily_trades']})")
                return False

            logger.info("   ✅ All risk conditions passed")
            return True
            
        except Exception as e:
            logger.error(f"💥 Error checking risk conditions: {e}")
            return False
    
    async def _log_trading_status(self, symbol: str, current_price: float, analysis: Dict):
        """Log detailed trading status with comprehensive information"""
        try:
            # Position information
            position_status = "No position"
            position_details = ""
            if self.current_position['side']:
                pnl = self.current_position.get('unrealized_pnl', 0)
                entry_price = self.current_position.get('entry_price', 0)
                position_status = f"{self.current_position['side'].upper()} {self.current_position['size']:.6f}"
                position_details = f" | Entry: ${entry_price:.2f} | PnL: ${pnl:.2f}"
            
            # Account status
            try:
                account = self.binance_service.client.futures_account()
                balance = float(account['totalWalletBalance'])
                margin_used = float(account.get('totalPositionInitialMargin', 0))
                available = balance - margin_used
            except Exception:
                balance = available = margin_used = 0.0
            
            # Risk metrics
            volatility = analysis.get('volatility', 0)
            risk_score = analysis.get('risk_score', 0)
            trend_strength = analysis.get('trend_strength', 0)
            
            # Market analysis summary
            market_regime = analysis.get('market_regime', 'Unknown')
            support_resistance = analysis.get('support_resistance', {})
            support = support_resistance.get('support', 0)
            resistance = support_resistance.get('resistance', 0)

            logger.info("📊 TRADING STATUS SUMMARY")
            logger.info(f"   Symbol: {symbol} | Price: ${current_price:,.2f}")
            logger.info(f"   Position: {position_status}{position_details}")
            logger.info(f"   Account: ${balance:.2f} (Available: ${available:.2f}, Used: ${margin_used:.2f})")
            logger.info(f"   Market: {market_regime} | Risk: {risk_score:.3f} | Volatility: {volatility:.3f}")
            logger.info(f"   Support: ${support:.2f} | Resistance: ${resistance:.2f}")
            logger.info(f"   Daily Trades: {self.daily_trades}/{self.risk_config['max_daily_trades']}")
            logger.info(f"   Trading Active: {'YES' if self.is_trading else 'NO'}")
            
            # Log why we might not be trading
            if self.is_trading and not self.current_position['side']:
                reasons = []
                if volatility > 0.05:
                    reasons.append(f"High volatility ({volatility:.3f})")
                if risk_score > 0.7:
                    reasons.append(f"High risk ({risk_score:.3f})")
                if self.daily_trades >= self.risk_config['max_daily_trades']:
                    reasons.append("Daily limit reached")
                if trend_strength < 0.3:
                    reasons.append(f"Weak trend ({trend_strength:.3f})")
                
                if reasons:
                    logger.info(f"   🚫 Not trading due to: {', '.join(reasons)}")
                else:
                    logger.info("   ✅ Ready to trade - Waiting for good signal")
            
        except Exception as e:
            logger.error(f"Error logging trading status: {e}")
    
    def stop_enhanced_trading(self) -> bool:
        """Stop enhanced trading"""
        try:
            if not self.is_trading:
                return True
                
            self.is_trading = False
            
            if self.trading_task and not self.trading_task.done():
                self.trading_task.cancel()
            
            logger.info("Enhanced trading stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping enhanced trading: {e}")
            return False
    
    def get_enhanced_status(self) -> Dict:
        """Get comprehensive trading status"""
        try:
            # Calculate performance metrics
            total_trades = len(self.trading_history)
            profitable_trades = len([t for t in self.trading_history if t['pnl'] > 0])
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
            
            total_pnl = sum(t['pnl'] for t in self.trading_history)
            
            return {
                'is_trading': self.is_trading,
                'current_position': self.current_position,
                'risk_config': self.risk_config,
                'performance': {
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl,
                    'max_drawdown': self.max_drawdown,
                    'daily_trades': self.daily_trades
                },
                'recent_trades': self.trading_history[-5:] if self.trading_history else []
            }
            
        except Exception as e:
            logger.error(f"Error getting enhanced status: {e}")
            return {'error': str(e)}
    
    def _record_trade_to_database(self, trade_data: Dict):
        """Record trade execution to database"""
        try:
            # Prepare trade data for database
            db_trade_data = {
                'symbol': trade_data.get('symbol'),
                'side': trade_data.get('side'),
                'quantity': trade_data.get('quantity'),
                'price': trade_data.get('price'),
                'pnl': trade_data.get('pnl', 0.0),
                'commission': trade_data.get('commission', 0.0),
                'order_id': trade_data.get('order_id'),
                'status': trade_data.get('status', 'FILLED'),
                'trading_mode': getattr(self, 'current_mode', 'enhanced'),
                'position_size': trade_data.get('position_size'),
                'model_training_id': getattr(self, 'current_model_training_id', None)
            }
            
            # Save to database
            if self.db_service:
                trade_record = self.db_service.create_trade(db_trade_data)
                logger.info(f"📊 Trade recorded to database: ID {trade_record.id}")
            else:
                logger.warning("📊 Database service not available - trade not recorded")
            
            # Update trading session statistics if we have an active session
            if self.trading_session_id:
                self._update_trading_session_stats()
                
        except Exception as e:
            logger.error(f"Failed to record trade to database: {e}")
    
    def _update_trading_session_stats(self):
        """Update trading session performance statistics"""
        try:
            if not self.trading_session_id:
                return
                
            # Calculate session statistics
            total_trades = len(self.trading_history)
            winning_trades = len([t for t in self.trading_history if t.get('pnl', 0) > 0])
            losing_trades = len([t for t in self.trading_history if t.get('pnl', 0) < 0])
            total_pnl = sum(t.get('pnl', 0) for t in self.trading_history)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Update session in database
            updates = {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'updated_at': datetime.utcnow()
            }
            
            if self.db_service:
                self.db_service.update_trading_session(self.trading_session_id, updates)
                logger.debug(f"📈 Trading session stats updated: {total_trades} trades, {win_rate:.1f}% win rate")
            else:
                logger.warning("📈 Database service not available - session stats not updated")
            
        except Exception as e:
            logger.error(f"Failed to update trading session stats: {e}")
    
    def start_trading_session(self, symbol: str, mode: str, position_size: float, 
                             stop_loss: Optional[float] = None, take_profit: Optional[float] = None):
        """Start a new trading session and record it in the database"""
        try:
            self.trading_session_id = str(uuid.uuid4())
            
            session_data = {
                'session_id': self.trading_session_id,
                'symbol': symbol,
                'trading_mode': mode,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'model_version': getattr(self.ml_service, 'model_version', 'unknown'),
                'status': 'ACTIVE'
            }
            
            # Save to database
            if self.db_service:
                trading_session = self.db_service.create_trading_session(session_data)
                self.current_mode = mode
                
                logger.info(f"🚀 Trading session started: {self.trading_session_id}")
                return trading_session.to_dict()
            else:
                self.current_mode = mode
                logger.warning(f"🚀 Trading session started (no database): {self.trading_session_id}")
                return session_data
            
        except Exception as e:
            logger.error(f"Failed to start trading session: {e}")
            return None
    
    def stop_trading_session(self):
        """Stop the current trading session"""
        try:
            if not self.trading_session_id:
                return None
                
            # Final statistics update
            self._update_trading_session_stats()
            
            # Mark session as stopped
            updates = {
                'status': 'STOPPED',
                'ended_at': datetime.utcnow()
            }
            
            if self.db_service:
                session = self.db_service.update_trading_session(self.trading_session_id, updates)
                logger.info(f"🛑 Trading session stopped: {self.trading_session_id}")
                
                self.trading_session_id = None
                return session.to_dict() if session else None
            else:
                logger.info(f"🛑 Trading session stopped (no database): {self.trading_session_id}")
                self.trading_session_id = None
                return updates
            
        except Exception as e:
            logger.error(f"Failed to stop trading session: {e}")
            return None