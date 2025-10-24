"""
Binance API Service
Handles all interactions with Binance exchange
"""
from binance.client import Client
from binance.exceptions import BinanceAPIException
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class BinanceService:
    """Service for interacting with Binance API"""
    
    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.client = Client(api_key, api_secret, testnet=testnet)
        self.testnet = testnet
        
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balances"""
        try:
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
        except BinanceAPIException as e:
            logger.error(f"Error getting balance: {e}")
            return {}
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except BinanceAPIException as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_24h_ticker(self, symbol: str) -> Optional[Dict]:
        """Get 24h ticker statistics"""
        try:
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
        except BinanceAPIException as e:
            logger.error(f"Error getting 24h ticker for {symbol}: {e}")
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
        """Get historical candlestick data"""
        try:
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
        except BinanceAPIException as e:
            logger.error(f"Error getting historical data: {e}")
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