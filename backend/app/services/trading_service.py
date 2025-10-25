"""
Trading Service - Enhanced for Futures Trading
Orchestrates trading operations using ML predictions and Binance Futures API
"""
import logging
import asyncio
from typing import Dict, Optional, List
from datetime import datetime

from app.services.binance_service import BinanceService
from app.services.ml_service import MLService

logger = logging.getLogger(__name__)


class TradingService:
    """Main trading service that coordinates ML predictions and futures order execution"""
    
    def __init__(self, binance_service: BinanceService, ml_service: MLService):
        self.binance_service = binance_service
        self.ml_service = ml_service
        self.is_trading = False
        self.trading_task = None
        self.trading_history = []
        self.current_position = None
        
        # Futures trading configuration
        self.default_leverage = 10
        self.use_leverage = True
        
    def start_trading(self, symbol: str = "BTCUSDT", mode: str = "balanced", leverage: int = 10) -> bool:
        """Start automated futures trading"""
        try:
            if self.is_trading:
                logger.warning("Trading is already active")
                return False
            
            if not self.ml_service.model:
                logger.error("No trained model available")
                return False
            
            # Set up futures trading
            self._setup_futures_trading(symbol, leverage)
                
            self.is_trading = True
            self.trading_task = asyncio.create_task(
                self._trading_loop(symbol, mode)
            )
            
            logger.info(f"Started futures trading {symbol} in {mode} mode with {leverage}x leverage")
            return True
            
        except Exception as e:
            logger.error(f"Error starting futures trading: {e}")
            return False
    
    def _setup_futures_trading(self, symbol: str, leverage: int):
        """Set up futures trading configuration"""
        try:
            # Set margin type to CROSSED for better risk management
            self.binance_service.set_margin_type(symbol, 'CROSSED')
            
            # Set leverage
            if self.use_leverage:
                self.binance_service.set_leverage(symbol, leverage)
                logger.info(f"Futures trading setup complete for {symbol} with {leverage}x leverage")
            
        except Exception as e:
            logger.error(f"Error setting up futures trading: {e}")
            raise
    
    def stop_trading(self) -> bool:
        """Stop automated trading"""
        try:
            if not self.is_trading:
                return True
                
            self.is_trading = False
            
            if self.trading_task and not self.trading_task.done():
                self.trading_task.cancel()
            
            logger.info("Trading stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping trading: {e}")
            return False
    
    async def _trading_loop(self, symbol: str, mode: str):
        """Main trading loop"""
        try:
            while self.is_trading:
                # Get current market data
                current_price = self.binance_service.get_current_price(symbol)
                if not current_price:
                    await asyncio.sleep(60)  # Wait 1 minute before retry
                    continue
                
                # Get historical data for prediction
                klines = self.binance_service.get_historical_klines(
                    symbol=symbol,
                    interval='1h',
                    limit=100
                )
                
                if not klines:
                    await asyncio.sleep(60)
                    continue
                
                # Prepare observation for ML model
                observation = self._prepare_observation(klines)
                
                if observation is not None:
                    # Get ML prediction
                    action, confidence = self.ml_service.predict(observation)
                    
                    # Execute trading decision
                    await self._execute_trading_decision(
                        symbol, action, confidence, current_price, mode
                    )
                
                # Wait before next iteration (5 minutes)
                await asyncio.sleep(300)
                
        except asyncio.CancelledError:
            logger.info("Trading loop cancelled")
        except Exception as e:
            logger.error(f"Error in trading loop: {e}")
            self.is_trading = False
    
    def _prepare_observation(self, klines: List[Dict]) -> Optional[List[float]]:
        """Prepare observation data for ML model"""
        try:
            import pandas as pd
            import ta
            
            if len(klines) < 50:  # Need enough data for indicators
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(klines)
            
            # Calculate technical indicators (matching training environment)
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
            
            # ATR
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=14
            )
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            
            # Additional per-row features to match training observation size
            # 5 extra features per row -> total features per row becomes 23 (18 + 5)
            import numpy as _np

            df['close_pct_change'] = df['close'].pct_change().fillna(0)
            df['log_return'] = _np.log(df['close'].replace(0, _np.nan)).diff().fillna(0)
            df['high_low_spread'] = (df['high'] - df['low']) / (df['low'] + 1e-10)
            df['body_size'] = (df['close'] - df['open']) / (df['open'] + 1e-10)
            df['typical_price_ratio'] = ((df['high'] + df['low'] + df['close']) / 3) / (df['close'] + 1e-10)

            # Fill NaN values
            df = df.fillna(0)

            # Get last 30 rows with all features (matching training window)
            window_data = df.tail(30)
            
            # Extract features in the same order as training
            features = []
            for _, row in window_data.iterrows():
                row_features = [
                    row['open'], row['high'], row['low'], row['close'], row['volume'],
                    row['sma_10'], row['sma_30'], row['ema_12'],
                    row['macd'], row['macd_signal'],
                    row['rsi'],
                    row['bb_high'], row['bb_low'], row['bb_mid'],
                    row['volume_ratio'], row['atr'],
                    row['stoch_k'], row['stoch_d'],
                    # Extra features to match training shape
                    row['close_pct_change'], row['log_return'], row['high_low_spread'],
                    row['body_size'], row['typical_price_ratio']
                ]
                features.extend(row_features)

            # Convert to numpy array (stable-baselines expects numpy inputs)
            import numpy as np
            obs = np.array(features, dtype=float)

            # Sanity check: expected shape is 23 features * 30 window = 690
            if obs.size != 23 * 30:
                logger.warning(f"Prepared observation size {obs.size} != expected {23*30}")

            return obs
            
        except Exception as e:
            logger.error(f"Error preparing observation: {e}")
            return None
    
    async def _execute_trading_decision(
        self, 
        symbol: str, 
        action: int, 
        confidence: float,
        current_price: float,
        mode: str
    ):
        """Execute futures trading decision based on ML prediction"""
        try:
            # Get current positions
            positions = self.binance_service.get_position_info(symbol)
            current_position = positions[0] if positions else None
            
            # Get portfolio information
            portfolio = self.binance_service.get_portfolio_value()
            
            # Define risk parameters based on mode
            risk_params = self._get_risk_parameters(mode)
            
            # Only trade if confidence is above threshold
            if confidence < risk_params['min_confidence']:
                return
            
            # Calculate position size based on available balance
            position_size = self._calculate_futures_position_size(
                portfolio, current_price, risk_params
            )
            
            if position_size <= 0:
                return
            
            # Execute action based on current position and signal
            if action == 1:  # Buy signal
                await self._execute_futures_buy(symbol, position_size, current_price, current_position)
            elif action == 2:  # Sell signal
                await self._execute_futures_sell(symbol, position_size, current_price, current_position)
            # action == 0 is hold, do nothing
            
        except Exception as e:
            logger.error(f"Error executing futures trading decision: {e}")
    
    def _calculate_futures_position_size(
        self, 
        portfolio: Dict, 
        current_price: float, 
        risk_params: Dict
    ) -> float:
        """Calculate position size for futures trading"""
        try:
            available_balance = portfolio.get('available_balance', 0)
            
            if available_balance <= 0:
                return 0
            
            # Use a percentage of available balance for position
            max_position_value = available_balance * risk_params['max_position_size']
            
            # With leverage, we can trade larger positions
            leveraged_position_value = max_position_value * self.default_leverage
            
            # Calculate quantity based on current price
            position_size = leveraged_position_value / current_price
            
            # Round to appropriate precision (typically 3 decimal places for BTC)
            position_size = round(position_size, 3)
            
            logger.info(f"Calculated futures position size: {position_size} (leveraged value: ${leveraged_position_value:.2f})")
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating futures position size: {e}")
            return 0
    
    def _get_risk_parameters(self, mode: str) -> Dict[str, float]:
        """Get risk parameters based on trading mode for futures"""
        params = {
            'conservative': {
                'min_confidence': 0.8,
                'max_position_size': 0.05,  # 5% of available balance per trade
                'stop_loss': 0.02,  # 2%
                'take_profit': 0.04  # 4%
            },
            'balanced': {
                'min_confidence': 0.65,
                'max_position_size': 0.1,  # 10% of available balance per trade
                'stop_loss': 0.03,  # 3%
                'take_profit': 0.06  # 6%
            },
            'aggressive': {
                'min_confidence': 0.55,
                'max_position_size': 0.2,  # 20% of available balance per trade
                'stop_loss': 0.05,  # 5%
                'take_profit': 0.1  # 10%
            }
        }
        
        return params.get(mode, params['balanced'])
    
    def _calculate_position_size(
        self, 
        balance: Dict, 
        price: float, 
        risk_params: Dict[str, float]
    ) -> float:
        """Calculate position size based on available balance and risk parameters"""
        try:
            # Get USDT balance
            usdt_balance = balance.get('USDT', {}).get('free', 0)
            
            # Calculate max position value
            max_position_value = usdt_balance * risk_params['max_position_size']
            
            # Calculate quantity
            quantity = max_position_value / price
            
            # Round to appropriate decimal places (typically 6 for crypto)
            return round(quantity, 6)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    async def _execute_futures_buy(self, symbol: str, quantity: float, price: float, current_position: Optional[Dict]):
        """Execute futures buy order (long position)"""
        try:
            # If we have a short position, close it first
            if current_position and current_position['side'] == 'SHORT':
                logger.info("Closing existing SHORT position before opening LONG")
                close_result = self.binance_service.close_position(symbol)
                if close_result:
                    self._log_trade('CLOSE_SHORT', symbol, current_position['position_amt'], price, close_result.get('orderId'))
            
            # Open long position
            result = self.binance_service.place_futures_order(
                symbol=symbol,
                side='BUY',
                quantity=quantity,
                order_type='MARKET'
            )
            
            if result:
                self.current_position = {
                    'symbol': symbol,
                    'side': 'LONG',
                    'quantity': result['quantity'],
                    'entry_price': result.get('avg_price', price),
                    'timestamp': datetime.now(),
                    'order_id': result['orderId'],
                    'leverage': self.default_leverage
                }
                
                self._log_trade('BUY_LONG', symbol, result['quantity'], result.get('avg_price', price), result['orderId'])
                logger.info(f"Executed LONG position: {result}")
            
        except Exception as e:
            logger.error(f"Error executing futures buy order: {e}")
    
    async def _execute_futures_sell(self, symbol: str, quantity: float, price: float, current_position: Optional[Dict]):
        """Execute futures sell order (short position)"""
        try:
            # If we have a long position, close it first
            if current_position and current_position['side'] == 'LONG':
                logger.info("Closing existing LONG position before opening SHORT")
                close_result = self.binance_service.close_position(symbol)
                if close_result:
                    self._log_trade('CLOSE_LONG', symbol, current_position['position_amt'], price, close_result.get('orderId'))
            
            # Open short position
            result = self.binance_service.place_futures_order(
                symbol=symbol,
                side='SELL',
                quantity=quantity,
                order_type='MARKET'
            )
            
            if result:
                self.current_position = {
                    'symbol': symbol,
                    'side': 'SHORT',
                    'quantity': result['quantity'],
                    'entry_price': result.get('avg_price', price),
                    'timestamp': datetime.now(),
                    'order_id': result['orderId'],
                    'leverage': self.default_leverage
                }
                
                self._log_trade('SELL_SHORT', symbol, result['quantity'], result.get('avg_price', price), result['orderId'])
                logger.info(f"Executed SHORT position: {result}")
            
        except Exception as e:
            logger.error(f"Error executing futures sell order: {e}")
    
    def _log_trade(self, action: str, symbol: str, quantity: float, price: float, order_id: int):
        """Log a trade to trading history"""
        self.trading_history.append({
            'action': action,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now(),
            'order_id': order_id
        })
    
    async def _execute_buy(self, symbol: str, quantity: float, price: float):
        """Execute buy order"""
        try:
            result = self.binance_service.place_market_order(
                symbol=symbol,
                side='BUY',
                quantity=quantity
            )
            
            if result:
                self.current_position = {
                    'symbol': symbol,
                    'side': 'BUY',
                    'quantity': result['quantity'],
                    'entry_price': result['price'],
                    'timestamp': datetime.now(),
                    'order_id': result['orderId']
                }
                
                self.trading_history.append({
                    'action': 'BUY',
                    'symbol': symbol,
                    'quantity': result['quantity'],
                    'price': result['price'],
                    'timestamp': datetime.now(),
                    'order_id': result['orderId']
                })
                
                logger.info(f"Executed BUY order: {result}")
            
        except Exception as e:
            logger.error(f"Error executing buy order: {e}")
    
    async def _execute_sell(self, symbol: str, quantity: float, price: float):
        """Execute sell order"""
        try:
            # Get current crypto holdings
            balance = self.binance_service.get_account_balance()
            base_asset = symbol.replace('USDT', '')
            crypto_balance = balance.get(base_asset, {}).get('free', 0)
            
            # Use actual holdings or requested quantity, whichever is smaller
            sell_quantity = min(quantity, crypto_balance)
            
            if sell_quantity > 0:
                result = self.binance_service.place_market_order(
                    symbol=symbol,
                    side='SELL',
                    quantity=sell_quantity
                )
                
                if result:
                    self.current_position = None
                    
                    self.trading_history.append({
                        'action': 'SELL',
                        'symbol': symbol,
                        'quantity': result['quantity'],
                        'price': result['price'],
                        'timestamp': datetime.now(),
                        'order_id': result['orderId']
                    })
                    
                    logger.info(f"Executed SELL order: {result}")
            
        except Exception as e:
            logger.error(f"Error executing sell order: {e}")
    
    def get_trading_status(self) -> Dict:
        """Get current futures trading status"""
        # Get live position information from Binance
        live_positions = self.binance_service.get_position_info()
        
        return {
            'is_trading': self.is_trading,
            'current_position': self.current_position,
            'live_positions': live_positions,
            'total_trades': len(self.trading_history),
            'recent_trades': self.trading_history[-5:] if self.trading_history else [],
            'leverage': self.default_leverage,
            'trading_mode': 'futures'
        }
    
    def get_trading_history(self) -> List[Dict]:
        """Get complete trading history"""
        return self.trading_history