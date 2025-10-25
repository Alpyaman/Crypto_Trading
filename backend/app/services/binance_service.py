"""
Binance API Service
Handles all interactions with Binance exchange - Enhanced for Futures Trading
"""
from binance.client import Client
from binance.exceptions import BinanceAPIException
from typing import Dict, List, Optional
import logging
import time
from requests.exceptions import ReadTimeout, ConnectionError, RequestException

logger = logging.getLogger(__name__)


class BinanceService:
    """Service for interacting with Binance API - Enhanced for Futures Trading"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        # Initialize client with custom timeout settings
        self.client = Client(
            api_key, 
            api_secret, 
            testnet=testnet,
            requests_params={
                'timeout': 30,  # Increase timeout to 30 seconds
                'verify': True
            }
        )
        self.testnet = testnet
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
        # Simple cache for last known good values
        self._price_cache = {}
        self._cache_timestamp = {}
        self.cache_ttl = 60  # Cache valid for 60 seconds
        
        # Futures trading configuration
        self.default_leverage = 10  # Default leverage for futures
        self.margin_type = 'CROSSED'  # CROSSED or ISOLATED
        
    def _retry_api_call(self, func, *args, **kwargs):
        """Wrapper to retry API calls with exponential backoff"""
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except (ReadTimeout, ConnectionError, RequestException) as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"API call failed after {self.max_retries} attempts: {e}")
            except BinanceAPIException as e:
                logger.error(f"Binance API error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in API call: {e}")
                raise
        
        # If we get here, all retries failed
        raise last_exception
        
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached value is still valid"""
        if key not in self._cache_timestamp:
            return False
        return (time.time() - self._cache_timestamp[key]) < self.cache_ttl
    
    def _get_cached_price(self, symbol: str) -> Optional[float]:
        """Get cached price if valid"""
        if self._is_cache_valid(symbol):
            return self._price_cache.get(symbol)
        return None
    
    def _cache_price(self, symbol: str, price: float):
        """Cache a price with timestamp"""
        self._price_cache[symbol] = price
        self._cache_timestamp[symbol] = time.time()
        
    def check_api_connectivity(self) -> bool:
        """Check if the API is accessible"""
        try:
            def _ping():
                return self.client.ping()
            
            self._retry_api_call(_ping)
            return True
        except Exception as e:
            logger.error(f"API connectivity check failed: {e}")
            return False
    
    def get_service_status(self) -> Dict:
        """Get comprehensive service status"""
        status = {
            'api_connected': False,
            'testnet_mode': self.testnet,
            'cache_size': len(self._price_cache),
            'last_errors': []
        }
        
        try:
            status['api_connected'] = self.check_api_connectivity()
        except Exception as e:
            status['last_errors'].append(f"Connectivity check failed: {str(e)}")
        
        return status
        
    def get_account_balance(self) -> Dict[str, float]:
        """Get futures account balances with retry logic"""
        try:
            def _get_balance():
                # Get futures account information
                account = self.client.futures_account()
                balances = {}
                
                # Parse futures account balance
                for balance in account['assets']:
                    asset = balance['asset']
                    wallet_balance = float(balance['walletBalance'])
                    unrealized_pnl = float(balance['unrealizedProfit'])
                    margin_balance = float(balance['marginBalance'])
                    available_balance = float(balance['availableBalance'])
                    
                    if wallet_balance > 0 or unrealized_pnl != 0:
                        balances[asset] = {
                            'wallet_balance': wallet_balance,
                            'unrealized_pnl': unrealized_pnl,
                            'margin_balance': margin_balance,
                            'available_balance': available_balance,
                            'total': margin_balance
                        }
                
                return balances
            
            return self._retry_api_call(_get_balance)
        except (ReadTimeout, ConnectionError, RequestException) as e:
            logger.error(f"Network error getting futures account balance: {e}")
            return {}
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting futures account balance: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting futures account balance: {e}")
            return {}
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current futures price for a symbol with retry logic and caching fallback"""
        try:
            def _get_price():
                # Use futures symbol ticker for better accuracy
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
                price = float(ticker['price'])
                # Cache the successful result
                self._cache_price(symbol, price)
                return price
            
            return self._retry_api_call(_get_price)
        except (ReadTimeout, ConnectionError, RequestException) as e:
            logger.error(f"Network error getting futures price for {symbol}: {e}")
            # Try to return cached value
            cached_price = self._get_cached_price(symbol)
            if cached_price is not None:
                logger.warning(f"Using cached futures price for {symbol}: {cached_price}")
                return cached_price
            return None
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting futures price for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting futures price for {symbol}: {e}")
            return None
    
    def get_24h_ticker(self, symbol: str) -> Optional[Dict]:
        """Get 24h futures ticker statistics with retry logic"""
        try:
            def _get_ticker():
                # Use futures 24hr ticker for comprehensive data
                ticker = self.client.futures_ticker(symbol=symbol)
                return {
                    'symbol': ticker['symbol'],
                    'price': float(ticker['lastPrice']),
                    'change': float(ticker['priceChange']),
                    'change_percent': float(ticker['priceChangePercent']),
                    'volume': float(ticker['volume']),
                    'quote_volume': float(ticker['quoteVolume']),
                    'high': float(ticker['highPrice']),
                    'low': float(ticker['lowPrice']),
                    'open_interest': float(ticker.get('openInterest', 0)),
                    'mark_price': float(ticker.get('markPrice', ticker['lastPrice']))
                }
            
            return self._retry_api_call(_get_ticker)
        except (ReadTimeout, ConnectionError, RequestException) as e:
            logger.error(f"Network error getting 24h futures ticker for {symbol}: {e}")
            return None
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting 24h futures ticker for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting 24h futures ticker for {symbol}: {e}")
            return None
    
    def place_futures_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float,
        order_type: str = 'MARKET',
        price: Optional[float] = None,
        leverage: Optional[int] = None
    ) -> Optional[Dict]:
        """Place a futures order with leverage support"""
        try:
            # Set leverage if specified
            if leverage:
                self.set_leverage(symbol, leverage)
            
            def _place_order():
                order_params = {
                    'symbol': symbol,
                    'side': side,  # 'BUY' or 'SELL'
                    'type': order_type,  # 'MARKET', 'LIMIT', etc.
                    'quantity': quantity,
                }
                
                # Add price for limit orders
                if order_type == 'LIMIT' and price:
                    order_params['price'] = price
                    order_params['timeInForce'] = 'GTC'  # Good Till Cancelled
                
                order = self.client.futures_create_order(**order_params)
                
                return {
                    'orderId': order['orderId'],
                    'symbol': order['symbol'],
                    'side': order['side'],
                    'type': order['type'],
                    'quantity': float(order['origQty']),
                    'executed_qty': float(order.get('executedQty', 0)),
                    'price': float(order.get('price', 0)),
                    'avg_price': float(order.get('avgPrice', 0)),
                    'status': order['status'],
                    'timestamp': order['updateTime']
                }
            
            return self._retry_api_call(_place_order)
        except (ReadTimeout, ConnectionError, RequestException) as e:
            logger.error(f"Network error placing futures order: {e}")
            return None
        except BinanceAPIException as e:
            logger.error(f"Binance API error placing futures order: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error placing futures order: {e}")
            return None
    
    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a futures symbol"""
        try:
            def _set_leverage():
                result = self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
                return result['leverage'] == leverage
            
            success = self._retry_api_call(_set_leverage)
            if success:
                logger.info(f"Successfully set leverage to {leverage}x for {symbol}")
            return success
        except Exception as e:
            logger.error(f"Error setting leverage for {symbol}: {e}")
            return False
    
    def set_margin_type(self, symbol: str, margin_type: str = 'CROSSED') -> bool:
        """Set margin type for a futures symbol (CROSSED or ISOLATED)"""
        try:
            def _set_margin():
                self.client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
                return True
            
            success = self._retry_api_call(_set_margin)
            if success:
                logger.info(f"Successfully set margin type to {margin_type} for {symbol}")
                self.margin_type = margin_type
            return success
        except BinanceAPIException as e:
            # Margin type might already be set correctly
            if "No need to change margin type" in str(e):
                logger.info(f"Margin type already set to {margin_type} for {symbol}")
                return True
            logger.error(f"Error setting margin type for {symbol}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error setting margin type for {symbol}: {e}")
            return False
    
    def get_position_info(self, symbol: str = None) -> List[Dict]:
        """Get current futures positions"""
        try:
            def _get_positions():
                positions = self.client.futures_position_information(symbol=symbol)
                active_positions = []
                
                for pos in positions:
                    position_amt = float(pos['positionAmt'])
                    if abs(position_amt) > 0:  # Only include positions with size > 0
                        active_positions.append({
                            'symbol': pos['symbol'],
                            'position_amt': position_amt,
                            'entry_price': float(pos['entryPrice']),
                            'mark_price': float(pos['markPrice']),
                            'unrealized_pnl': float(pos['unRealizedProfit']),
                            'percentage': float(pos['percentage']),
                            'side': 'LONG' if position_amt > 0 else 'SHORT',
                            'margin_type': pos['marginType'],
                            'leverage': int(pos['leverage'])
                        })
                
                return active_positions
            
            return self._retry_api_call(_get_positions)
        except Exception as e:
            logger.error(f"Error getting position info: {e}")
            return []
    
    def close_position(self, symbol: str) -> Optional[Dict]:
        """Close all positions for a symbol"""
        try:
            positions = self.get_position_info(symbol)
            if not positions:
                logger.info(f"No open positions for {symbol}")
                return None
            
            position = positions[0]  # Should only be one position per symbol
            position_amt = abs(position['position_amt'])
            
            # Determine the opposite side to close the position
            close_side = 'SELL' if position['side'] == 'LONG' else 'BUY'
            
            # Place market order to close position
            return self.place_futures_order(
                symbol=symbol,
                side=close_side,
                quantity=position_amt,
                order_type='MARKET'
            )
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return None
    
    def get_historical_klines(
        self,
        symbol: str,
        interval: str = '1h',
        limit: int = 500
    ) -> List[Dict]:
        """Get historical futures candlestick data with retry logic"""
        try:
            def _get_klines():
                # Use futures klines for better data consistency
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
                
                candles = []
                for kline in klines:
                    candles.append({
                        'timestamp': kline[0],
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5]),
                        'quote_volume': float(kline[7]),
                        'trades': int(kline[8])
                    })
                
                return candles
            
            return self._retry_api_call(_get_klines)
        except (ReadTimeout, ConnectionError, RequestException) as e:
            logger.error(f"Network error getting futures historical data for {symbol}: {e}")
            return []
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting futures historical data for {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting futures historical data for {symbol}: {e}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get futures trading rules for a symbol"""
        try:
            def _get_symbol_info():
                # Get futures exchange info
                exchange_info = self.client.futures_exchange_info()
                
                for symbol_info in exchange_info['symbols']:
                    if symbol_info['symbol'] == symbol:
                        # Extract important filters
                        filters = {}
                        for f in symbol_info['filters']:
                            if f['filterType'] == 'LOT_SIZE':
                                filters['min_qty'] = float(f['minQty'])
                                filters['max_qty'] = float(f['maxQty'])
                                filters['step_size'] = float(f['stepSize'])
                            elif f['filterType'] == 'PRICE_FILTER':
                                filters['min_price'] = float(f['minPrice'])
                                filters['max_price'] = float(f['maxPrice'])
                                filters['tick_size'] = float(f['tickSize'])
                            elif f['filterType'] == 'MIN_NOTIONAL':
                                filters['min_notional'] = float(f['notional'])
                            elif f['filterType'] == 'MARKET_LOT_SIZE':
                                filters['market_min_qty'] = float(f['minQty'])
                                filters['market_max_qty'] = float(f['maxQty'])
                        
                        return {
                            'symbol': symbol_info['symbol'],
                            'status': symbol_info['status'],
                            'base_asset': symbol_info['baseAsset'],
                            'quote_asset': symbol_info['quoteAsset'],
                            'contract_type': symbol_info['contractType'],
                            'delivery_date': symbol_info.get('deliveryDate'),
                            'filters': filters,
                            'precision': {
                                'quantity': symbol_info['quantityPrecision'],
                                'price': symbol_info['pricePrecision'],
                                'base': symbol_info['baseAssetPrecision'],
                                'quote': symbol_info['quotePrecision']
                            }
                        }
                
                return None
            
            return self._retry_api_call(_get_symbol_info)
        except Exception as e:
            logger.error(f"Error getting futures symbol info for {symbol}: {e}")
            return None
    
    def check_order_status(self, symbol: str, order_id: int) -> Optional[Dict]:
        """Check status of a futures order"""
        try:
            def _check_order():
                order = self.client.futures_get_order(symbol=symbol, orderId=order_id)
                return {
                    'orderId': order['orderId'],
                    'status': order['status'],
                    'type': order['type'],
                    'side': order['side'],
                    'origQty': float(order['origQty']),
                    'executedQty': float(order['executedQty']),
                    'cumQuote': float(order['cumQuote']),
                    'avgPrice': float(order.get('avgPrice', 0)),
                    'time': order['time'],
                    'updateTime': order['updateTime']
                }
            
            return self._retry_api_call(_check_order)
        except Exception as e:
            logger.error(f"Error checking futures order status: {e}")
            return None
    
    def get_portfolio_value(self, base_currency: str = 'USDT') -> Dict[str, float]:
        """Calculate total futures portfolio value and statistics"""
        try:
            account_info = self.client.futures_account()
            
            total_wallet_balance = 0.0
            total_unrealized_pnl = 0.0
            total_margin_balance = 0.0
            available_balance = 0.0
            
            for asset in account_info['assets']:
                if asset['asset'] == base_currency:
                    total_wallet_balance = float(asset['walletBalance'])
                    total_unrealized_pnl = float(asset['unrealizedProfit'])
                    total_margin_balance = float(asset['marginBalance'])
                    available_balance = float(asset['availableBalance'])
                    break
            
            return {
                'total_wallet_balance': total_wallet_balance,
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_margin_balance': total_margin_balance,
                'available_balance': available_balance,
                'total_initial_margin': float(account_info.get('totalInitialMargin', 0)),
                'total_maint_margin': float(account_info.get('totalMaintMargin', 0)),
                'max_withdraw_amount': float(account_info.get('maxWithdrawAmount', 0))
            }
        except Exception as e:
            logger.error(f"Error calculating futures portfolio value: {e}")
            return {
                'total_wallet_balance': 0.0,
                'total_unrealized_pnl': 0.0,
                'total_margin_balance': 0.0,
                'available_balance': 0.0,
                'total_initial_margin': 0.0,
                'total_maint_margin': 0.0,
                'max_withdraw_amount': 0.0
            }
    
    def validate_futures_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = 'MARKET'
    ) -> tuple[bool, str]:
        """Validate a futures order before placing"""
        try:
            # Get symbol info
            info = self.get_symbol_info(symbol)
            if not info:
                return False, "Could not get futures symbol info"
            
            filters = info['filters']
            
            # Check quantity constraints
            if quantity < filters.get('min_qty', 0):
                return False, f"Quantity too small. Min: {filters['min_qty']}"
            
            if quantity > filters.get('max_qty', float('inf')):
                return False, f"Quantity too large. Max: {filters['max_qty']}"
            
            # For market orders, use market lot size if available
            if order_type == 'MARKET':
                market_min = filters.get('market_min_qty', filters.get('min_qty', 0))
                market_max = filters.get('market_max_qty', filters.get('max_qty', float('inf')))
                
                if quantity < market_min:
                    return False, f"Market order quantity too small. Min: {market_min}"
                if quantity > market_max:
                    return False, f"Market order quantity too large. Max: {market_max}"
            
            # Check notional value
            price = self.get_current_price(symbol)
            if price:
                notional = quantity * price
                min_notional = filters.get('min_notional', 0)
                if notional < min_notional:
                    return False, f"Order value too small. Min notional: {min_notional}"
            
            # Check available balance for new positions
            portfolio = self.get_portfolio_value()
            available = portfolio.get('available_balance', 0)
            
            if available <= 0:
                return False, "Insufficient available balance for futures trading"
            
            return True, "Futures order valid"
            
        except Exception as e:
            return False, f"Futures validation error: {str(e)}"