"""
Binance API Service
Handles all interactions with Binance exchange
"""
from binance.client import Client
from binance.exceptions import BinanceAPIException
from typing import Dict, List, Optional
import logging
import time
from requests.exceptions import ReadTimeout, ConnectionError, RequestException

logger = logging.getLogger(__name__)


class BinanceService:
    """Service for interacting with Binance API"""
    
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
        """Get account balances with retry logic"""
        try:
            def _get_balance():
                account = self.client.get_account()
                balances = {}
                
                for balance in account['balances']:
                    asset = balance['asset']
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    total = free + locked
                    
                    if total > 0:
                        balances[asset] = {
                            'free': free,
                            'locked': locked,
                            'total': total
                        }
                
                return balances
            
            return self._retry_api_call(_get_balance)
        except (ReadTimeout, ConnectionError, RequestException) as e:
            logger.error(f"Network error getting account balance: {e}")
            return {}
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting account balance: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error getting account balance: {e}")
            return {}
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol with retry logic and caching fallback"""
        try:
            def _get_price():
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                price = float(ticker['price'])
                # Cache the successful result
                self._cache_price(symbol, price)
                return price
            
            return self._retry_api_call(_get_price)
        except (ReadTimeout, ConnectionError, RequestException) as e:
            logger.error(f"Network error getting price for {symbol}: {e}")
            # Try to return cached value
            cached_price = self._get_cached_price(symbol)
            if cached_price is not None:
                logger.warning(f"Using cached price for {symbol}: {cached_price}")
                return cached_price
            return None
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting price for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting price for {symbol}: {e}")
            return None
    
    def get_24h_ticker(self, symbol: str) -> Optional[Dict]:
        """Get 24h ticker statistics with retry logic"""
        try:
            def _get_ticker():
                ticker = self.client.get_ticker(symbol=symbol)
                return {
                    'symbol': ticker['symbol'],
                    'price': float(ticker['lastPrice']),
                    'change': float(ticker['priceChange']),
                    'change_percent': float(ticker['priceChangePercent']),
                    'volume': float(ticker['volume']),
                    'high': float(ticker['highPrice']),
                    'low': float(ticker['lowPrice'])
                }
            
            return self._retry_api_call(_get_ticker)
        except (ReadTimeout, ConnectionError, RequestException) as e:
            logger.error(f"Network error getting 24h ticker for {symbol}: {e}")
            return None
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting 24h ticker for {symbol}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting 24h ticker for {symbol}: {e}")
            return None
    
    def place_market_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float
    ) -> Optional[Dict]:
        """Place a market order"""
        try:
            order = self.client.create_order(
                symbol=symbol,
                side=side,  # 'BUY' or 'SELL'
                type='MARKET',
                quantity=quantity
            )
            
            return {
                'orderId': order['orderId'],
                'symbol': order['symbol'],
                'side': order['side'],
                'quantity': float(order['executedQty']),
                'price': float(order['fills'][0]['price']) if order['fills'] else 0,
                'status': order['status'],
                'timestamp': order['transactTime']
            }
        except BinanceAPIException as e:
            logger.error(f"Error placing order: {e}")
            return None
    
    def get_historical_klines(
        self,
        symbol: str,
        interval: str = '1h',
        limit: int = 500
    ) -> List[Dict]:
        """Get historical candlestick data with retry logic"""
        try:
            def _get_klines():
                klines = self.client.get_klines(
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
                        'volume': float(kline[5])
                    })
                
                return candles
            
            return self._retry_api_call(_get_klines)
        except (ReadTimeout, ConnectionError, RequestException) as e:
            logger.error(f"Network error getting historical data for {symbol}: {e}")
            return []
        except BinanceAPIException as e:
            logger.error(f"Binance API error getting historical data for {symbol}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting historical data for {symbol}: {e}")
            return []
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict]:
        """Get trading rules for a symbol"""
        try:
            info = self.client.get_symbol_info(symbol)
            
            # Extract important filters
            filters = {}
            for f in info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    filters['min_qty'] = float(f['minQty'])
                    filters['max_qty'] = float(f['maxQty'])
                    filters['step_size'] = float(f['stepSize'])
                elif f['filterType'] == 'PRICE_FILTER':
                    filters['min_price'] = float(f['minPrice'])
                    filters['max_price'] = float(f['maxPrice'])
                    filters['tick_size'] = float(f['tickSize'])
                elif f['filterType'] == 'MIN_NOTIONAL':
                    filters['min_notional'] = float(f['minNotional'])
            
            return {
                'symbol': info['symbol'],
                'status': info['status'],
                'base_asset': info['baseAsset'],
                'quote_asset': info['quoteAsset'],
                'filters': filters
            }
        except BinanceAPIException as e:
            logger.error(f"Error getting symbol info: {e}")
            return None
    
    def check_order_status(self, symbol: str, order_id: int) -> Optional[Dict]:
        """Check status of an order"""
        try:
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            return {
                'orderId': order['orderId'],
                'status': order['status'],
                'executedQty': float(order['executedQty']),
                'cummulativeQuoteQty': float(order['cummulativeQuoteQty'])
            }
        except BinanceAPIException as e:
            logger.error(f"Error checking order status: {e}")
            return None
    
    def get_portfolio_value(self, base_currency: str = 'USDT') -> float:
        """Calculate total portfolio value in base currency"""
        try:
            balances = self.get_account_balance()
            total_value = 0.0
            
            for asset, balance in balances.items():
                if balance['total'] > 0:
                    if asset == base_currency:
                        total_value += balance['total']
                    else:
                        # Get price in base currency
                        symbol = f"{asset}{base_currency}"
                        price = self.get_current_price(symbol)
                        if price:
                            total_value += balance['total'] * price
            
            return total_value
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return 0.0
    
    def validate_order(
        self,
        symbol: str,
        side: str,
        quantity: float
    ) -> tuple[bool, str]:
        """Validate an order before placing"""
        try:
            # Get symbol info
            info = self.get_symbol_info(symbol)
            if not info:
                return False, "Could not get symbol info"
            
            filters = info['filters']
            
            # Check quantity
            if quantity < filters['min_qty']:
                return False, f"Quantity too small. Min: {filters['min_qty']}"
            
            if quantity > filters['max_qty']:
                return False, f"Quantity too large. Max: {filters['max_qty']}"
            
            # Check notional value
            price = self.get_current_price(symbol)
            if price:
                notional = quantity * price
                if notional < filters.get('min_notional', 0):
                    return False, f"Order value too small. Min: {filters['min_notional']}"
            
            return True, "Order valid"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"