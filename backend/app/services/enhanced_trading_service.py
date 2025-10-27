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

from app.services.binance_service import BinanceService
from app.services.enhanced_ml_service import EnhancedMLService

logger = logging.getLogger(__name__)


class EnhancedTradingService:
    """Enhanced trading service with advanced ML and risk management"""
    
    def __init__(self, binance_service: BinanceService, enhanced_ml_service: EnhancedMLService):
        self.binance_service = binance_service
        self.ml_service = enhanced_ml_service
        self.is_trading = False
        self.trading_task = None
        self.trading_history = []
        
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
    
    def _create_observation_vector(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Create observation vector for ML model"""
        try:
            if len(data) < 20:
                return None
            
            # Use the last row for current market state
            last_row = data.iloc[-1]
            
            # Create a simplified observation vector
            # In production, this would be more sophisticated and match training data
            features = []
            
            # Price features
            if 'close' in last_row:
                features.extend([
                    last_row.get('price_change', 0),
                    last_row.get('volatility_ratio', 0),
                    last_row.get('bb_position', 0),
                    last_row.get('bb_width', 0),
                ])
            
            # Technical indicators
            current_price = last_row.get('close', 1)
            features.extend([
                last_row.get('sma_20', current_price) / current_price,
                last_row.get('ema_20', current_price) / current_price,
                last_row.get('macd', 0) / current_price,
                last_row.get('rsi_14', 50) / 100.0,
            ])
            
            # Volume features
            features.extend([
                last_row.get('volume_ratio', 1),
                min(1.0, last_row.get('volume', 0) / 1000000),  # Normalized volume
            ])
            
            # Position features (current position status)
            features.extend([
                1.0 if self.current_position['side'] == 'long' else 0.0,
                1.0 if self.current_position['side'] == 'short' else 0.0,
                self.current_position['size'] / 1000.0,  # Normalized position size
            ])
            
            # Pad to required size (simplified)
            while len(features) < 100:  # Match enhanced environment observation size
                features.append(0.0)
            
            return np.array(features[:100], dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error creating observation vector: {e}")
            return None
    
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
        """Adjust position size based on risk factors"""
        try:
            # Start with base size
            adjusted_size = base_size
            
            # Adjust based on confidence
            confidence_multiplier = min(confidence * 1.5, 1.0)
            adjusted_size *= confidence_multiplier
            
            # Adjust based on market volatility
            volatility = analysis.get('volatility', 0.02)
            volatility_adjustment = max(0.5, 1.0 - (volatility - 0.02) * 10)
            adjusted_size *= volatility_adjustment
            
            # Adjust based on market regime
            regime = analysis.get('market_regime', 'unknown')
            regime_multiplier = {
                'trending': 1.0,
                'ranging': 0.7,
                'volatile': 0.5,
                'unknown': 0.6
            }.get(regime, 0.6)
            adjusted_size *= regime_multiplier
            
            # Ensure within risk limits
            max_size = self.risk_config['max_position_size'] * self.peak_balance / self.current_position.get('entry_price', 1)
            adjusted_size = min(adjusted_size, max_size)
            
            return max(0.001, adjusted_size)  # Minimum position size
            
        except Exception as e:
            logger.error(f"Error adjusting position size: {e}")
            return base_size
    
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
            # Get current account balance
            account = self.binance_service.client.futures_account()
            current_balance = float(account['totalWalletBalance'])
            total_margin = float(account.get('totalPositionInitialMargin', 0))
            available_balance = current_balance - total_margin

            logger.info("🛡️ RISK MANAGEMENT CHECK")
            logger.info(f"   Current Balance: ${current_balance:.2f}")
            logger.info(f"   Available Balance: ${available_balance:.2f}")
            logger.info(f"   Used Margin: ${total_margin:.2f}")
            
            # Initialize peak balance if not set
            if self.peak_balance is None:
                self.peak_balance = current_balance
                logger.info(f"   Peak Balance Initialized: ${self.peak_balance:.2f}")
            
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
            
            # Update peak balance
            if current_balance > self.peak_balance:
                old_peak = self.peak_balance
                self.peak_balance = current_balance
                logger.info(f"   🎉 New Peak Balance: ${old_peak:.2f} → ${self.peak_balance:.2f}")
            
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