"""
Trading Service
Orchestrates trading operations using ML predictions and Binance API
"""
import logging
import asyncio
from typing import Dict, Optional, List
from datetime import datetime

from app.services.binance_service import BinanceService
from app.services.ml_service import MLService

logger = logging.getLogger(__name__)


class TradingService:
    """Main trading service that coordinates ML predictions and order execution"""
    
    def __init__(self, binance_service: BinanceService, ml_service: MLService):
        self.binance_service = binance_service
        self.ml_service = ml_service
        self.is_trading = False
        self.trading_task = None
        self.trading_history = []
        self.current_position = None
        
    def start_trading(self, symbol: str = "BTCUSDT", mode: str = "balanced") -> bool:
        """Start automated trading"""
        try:
            if self.is_trading:
                logger.warning("Trading is already active")
                return False
            
            if not self.ml_service.model:
                logger.error("No trained model available")
                return False
                
            self.is_trading = True
            self.trading_task = asyncio.create_task(
                self._trading_loop(symbol, mode)
            )
            
            logger.info(f"Started trading {symbol} in {mode} mode")
            return True
            
        except Exception as e:
            logger.error(f"Error starting trading: {e}")
            return False
    
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
        """Execute trading decision based on ML prediction"""
        try:
            # Get account balance
            balance = self.binance_service.get_account_balance()
            
            # Define risk parameters based on mode
            risk_params = self._get_risk_parameters(mode)
            
            # Only trade if confidence is above threshold
            if confidence < risk_params['min_confidence']:
                return
            
            # Calculate position size
            position_size = self._calculate_position_size(
                balance, current_price, risk_params
            )
            
            if position_size <= 0:
                return
            
            # Execute action
            if action == 1:  # Buy signal
                await self._execute_buy(symbol, position_size, current_price)
            elif action == 2:  # Sell signal
                await self._execute_sell(symbol, position_size, current_price)
            # action == 0 is hold, do nothing
            
        except Exception as e:
            logger.error(f"Error executing trading decision: {e}")
    
    def _get_risk_parameters(self, mode: str) -> Dict[str, float]:
        """Get risk parameters based on trading mode"""
        params = {
            'conservative': {
                'min_confidence': 0.8,
                'max_position_size': 0.1,  # 10% of balance
                'stop_loss': 0.02,  # 2%
                'take_profit': 0.04  # 4%
            },
            'balanced': {
                'min_confidence': 0.65,
                'max_position_size': 0.25,  # 25% of balance
                'stop_loss': 0.03,  # 3%
                'take_profit': 0.06  # 6%
            },
            'aggressive': {
                'min_confidence': 0.55,
                'max_position_size': 0.5,  # 50% of balance
                'stop_loss': 0.05,  # 5%
                'take_profit': 0.1   # 10%
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
        """Get current trading status"""
        return {
            'is_trading': self.is_trading,
            'current_position': self.current_position,
            'total_trades': len(self.trading_history),
            'recent_trades': self.trading_history[-5:] if self.trading_history else []
        }
    
    def get_trading_history(self) -> List[Dict]:
        """Get complete trading history"""
        return self.trading_history