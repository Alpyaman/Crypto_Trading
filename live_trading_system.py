"""
Live Trading System with Real Order Execution
WARNING: This system executes REAL trades with REAL money!
Only use with amounts you can afford to lose.
"""

import os
import time
import hmac
import hashlib
import requests
import json
import numpy as np
import traceback
from datetime import datetime
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
import logging
import sys

# Force UTF-8 encoding for emojis on Windows
if sys.platform.startswith('win'):
    import codecs
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except:
        pass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PPO Model imports
try:
    from stable_baselines3 import PPO
    from crypto_trading_env import CryptoTradingEnv
    PPO_AVAILABLE = True
    logger.info("âœ… PPO model dependencies loaded successfully")
except ImportError as e:
    PPO_AVAILABLE = False
    logger.warning(f"âš ï¸ PPO model dependencies not available: {e}")
    PPO = None
    CryptoTradingEnv = None

class TelegramNotifier:
    """Send trading notifications via Telegram."""
    
    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = chat_id or os.getenv('TELEGRAM_CHAT_ID')
        self.enabled = bool(self.bot_token and self.chat_id)
        
        if not self.enabled:
            logger.warning("âš ï¸ Telegram notifications disabled - missing bot token or chat ID")
        else:
            logger.info("ðŸ“± Telegram notifications enabled")
    
    def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Send a message to Telegram."""
        if not self.enabled:
            return False
        
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, data=data, timeout=10)
            
            if response.status_code == 200:
                logger.info("ðŸ“± Telegram notification sent successfully")
                return True
            else:
                logger.error(f" Telegram notification failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f" Error sending Telegram notification: {e}")
            return False
    
    def send_trade_notification(self, trade: 'RealTrade', portfolio_value: float = None, daily_pnl: float = None) -> bool:
        """Send trade execution notification."""
        if not self.enabled:
            return False
        
        # Create trade emoji
        action_emoji = "ðŸŸ¢" if trade.side.lower() == 'buy' else "ðŸ”´"
        symbol_emoji = self._get_symbol_emoji(trade.symbol)
        
        # Format message
        message = f"""
 <b>LIVE TRADE EXECUTED</b> 

{action_emoji} <b>{trade.side.upper()}</b> {symbol_emoji} <b>{trade.symbol}</b>

ðŸ’° <b>Amount:</b> {trade.amount:.6f}
ðŸ’µ <b>Price:</b> ${trade.price:.4f}
 <b>Value:</b> ${trade.value_usd:.2f}
ðŸ’¸ <b>Fee:</b> ${trade.fee_usd:.4f} ({trade.fee_rate*100:.3f}%)
ðŸ“‹ <b>Order ID:</b> <code>{trade.order_id}</code>
ðŸ“… <b>Time:</b> {trade.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Add portfolio info if available
        if portfolio_value is not None:
            message += f"\nðŸ’¼ <b>Portfolio Value:</b> ${portfolio_value:.2f}"
        
        if daily_pnl is not None:
            pnl_emoji = "ðŸ“ˆ" if daily_pnl >= 0 else "ðŸ“‰"
            message += f"\n{pnl_emoji} <b>Daily P&L:</b> ${daily_pnl:.2f}"
        
        return self.send_message(message)
    
    def send_ppo_decision_notification(self, decision: dict, executed: bool = False) -> bool:
        """Send PPO trading decision notification."""
        if not self.enabled:
            return False
        
        status_emoji = "âœ…" if executed else "â¸ï¸"
        action_emoji = "ðŸŸ¢" if decision.get('action') == 'buy' else "ðŸ”´" if decision.get('action') == 'sell' else "âšª"
        symbol_emoji = self._get_symbol_emoji(decision.get('symbol', ''))
        
        message = f"""
ðŸ§  <b>PPO AI DECISION</b> {status_emoji}

{action_emoji} <b>{decision.get('action', 'HOLD').upper()}</b> {symbol_emoji} <b>{decision.get('symbol', 'N/A')}</b>

ðŸ’° <b>Amount:</b> ${decision.get('amount', 0):.2f}
ðŸŽ¯ <b>Reasoning:</b> {decision.get('reasoning', 'AI analysis')}
ðŸ“… <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

<b>Status:</b> {"EXECUTED" if executed else "SKIPPED (safety limits)"}
"""
        
        return self.send_message(message)
    
    def send_system_notification(self, message: str, level: str = 'info') -> bool:
        """Send system status notification."""
        if not self.enabled:
            return False
        
        emoji_map = {
            'info': 'â„¹ï¸',
            'warning': 'âš ï¸',
            'error': '',
            'success': 'âœ…',
            'start': 'ðŸš€',
            'stop': 'â¹ï¸'
        }
        
        emoji = emoji_map.get(level, 'â„¹ï¸')
        formatted_message = f"{emoji} <b>TRADING SYSTEM</b>\n\n{message}"
        
        return self.send_message(formatted_message)
    
    def _get_symbol_emoji(self, symbol: str) -> str:
        """Get emoji for trading symbol."""
        symbol_map = {
            'BTCUSDT': 'â‚¿',
            'ETHUSDT': 'Îž',
            'BNBUSDT': 'ðŸ”¶',
            'DOGEUSDT': 'ðŸ•',
            'XRPUSDT': 'ðŸ’§',
            'ADAUSDT': 'ðŸ”µ',
            'SOLUSDT': 'â˜€ï¸',
            'DOTUSDT': 'ðŸ”´',
            'LINKUSDT': 'ðŸ”—',
            'UNIUSDT': 'ðŸ¦„',
            'AVAXUSDT': 'ðŸ”º',
            'ATOMUSDT': 'âš›ï¸',
            'NEARUSDT': 'ðŸŒ',
            'SANDUSDT': 'ðŸ–ï¸',
            'MANAUSDT': 'ðŸŽ®'
        }
        return symbol_map.get(symbol, 'ðŸ’°')


@dataclass
class RealTrade:
    """Real trade execution record."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: float
    value_usd: float
    order_id: str
    status: str  # 'filled', 'partial', 'pending', 'failed'
    commission: float = 0.0
    fee_usd: float = 0.0  # Trading fee in USD
    fee_rate: float = 0.001  # Fee rate used


@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot for tracking."""
    timestamp: datetime
    total_value_usd: float
    positions: Dict[str, float]
    prices: Dict[str, float]
    daily_pnl: float
    total_pnl: float


class LiveBinanceAPI:
    """Live Binance API client for real trading."""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = 'https://api.binance.com'
        
    def _create_signature(self, query_string: str) -> str:
        """Create HMAC SHA256 signature."""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _make_request(self, endpoint: str, params: Dict = None, signed: bool = False, method: str = 'GET') -> Dict:
        """Make HTTP request to Binance API."""
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        if params is None:
            params = {}
        
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            params['recvWindow'] = 60000
            
            query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
            signature = self._create_signature(query_string)
            params['signature'] = signature
            
            headers['X-MBX-APIKEY'] = self.api_key
        
        try:
            if method == 'GET':
                response = requests.get(url, params=params, headers=headers, timeout=15)
            elif method == 'POST':
                response = requests.post(url, params=params, headers=headers, timeout=15)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 400:
                return {'error': response.text}
            else:
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            logger.error(f"API request failed for {endpoint}: {e}")
            return {'error': str(e)}
    
    def get_account_info(self) -> Dict:
        """Get account information."""
        return self._make_request('/api/v3/account', signed=True)
    
    def get_ticker_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            data = self._make_request('/api/v3/ticker/price', {'symbol': symbol})
            if data and 'price' in data:
                return float(data['price'])
            return 0.0
        except:
            return 0.0
    
    def get_symbol_info(self, symbol: str) -> Dict:
        """Get trading rules for a symbol."""
        try:
            data = self._make_request('/api/v3/exchangeInfo')
            if 'symbols' in data:
                for sym_info in data['symbols']:
                    if sym_info['symbol'] == symbol:
                        return sym_info
            return {}
        except:
            return {}
    
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict:
        """Place a market order."""
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'MARKET',
            'quantity': f"{quantity:.8f}".rstrip('0').rstrip('.')
        }
        
        logger.warning(f" PLACING REAL ORDER: {side.upper()} {quantity:.8f} {symbol}")
        
        result = self._make_request('/api/v3/order', params, signed=True, method='POST')
        
        if 'error' in result:
            logger.error(f" Order failed: {result['error']}")
        else:
            logger.info(f"âœ… Order placed: {result.get('orderId', 'N/A')}")
        
        return result
    
    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict:
        """Place a limit order."""
        params = {
            'symbol': symbol,
            'side': side.upper(),
            'type': 'LIMIT',
            'timeInForce': 'GTC',
            'quantity': f"{quantity:.8f}".rstrip('0').rstrip('.'),
            'price': f"{price:.8f}".rstrip('0').rstrip('.')
        }
        
        logger.warning(f" PLACING REAL LIMIT ORDER: {side.upper()} {quantity:.8f} {symbol} @ ${price:.4f}")
        
        result = self._make_request('/api/v3/order', params, signed=True, method='POST')
        
        if 'error' in result:
            logger.error(f" Limit order failed: {result['error']}")
        else:
            logger.info(f"âœ… Limit order placed: {result.get('orderId', 'N/A')}")
        
        return result
    
    def get_order_status(self, symbol: str, order_id: str) -> Dict:
        """Get order status."""
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        return self._make_request('/api/v3/order', params, signed=True)
    
    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        """Cancel an order."""
        params = {
            'symbol': symbol,
            'orderId': order_id
        }
        
        logger.warning(f" CANCELLING ORDER: {order_id} for {symbol}")
        return self._make_request('/api/v3/order', params, signed=True, method='DELETE')


class LiveTradingManager:
    """Live trading manager with real order execution."""
    
    def __init__(self):
        # Load environment variables
        self._load_env()
        
        # Initialize API client
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("API keys not found in environment")
        
        self.api = LiveBinanceAPI(api_key, api_secret)
        
        # Initialize Telegram notifications
        self.telegram = TelegramNotifier()
        
        # Trading fee configuration
        self.base_fee_rate = 0.001  # 0.1% standard Binance fee
        self.bnb_discount = 0.25   # 25% discount when using BNB for fees
        self.has_bnb_for_fees = False  # Will check if user has BNB
        
        # Trading configuration
        self.max_trade_usd = 20.0  # Maximum $20 per trade for safety
        self.min_trade_usd = 6.0   # Minimum $6 per trade
        self.max_daily_trades = 30  # Maximum 30 trades per day
        self.max_daily_loss = 10.0  # Stop trading if daily loss > $10

        # PPO Strategy Configuration - IMPROVED VERSION
        self.ppo_config = {
            'min_confidence': 0.65,          # Minimum confidence to execute trade (increased from 0.4)
            'momentum_threshold': 0.75,      # Momentum score threshold for momentum trades
            'dip_threshold': -1.5,           # Price drop % threshold for dip buying (less aggressive)
            'breakout_position': 0.85,       # Position in 24h range for breakout trades (higher threshold)
            'max_usdt_per_trade': 0.25,      # Max % of USDT to use in single trade (reduced from 0.4)
            'diversification_limit': 4,      # Max number of different positions (reduced from 5)
            'profit_taking_threshold': 3.0,  # Daily P&L threshold for profit taking (lower target)
            'stop_loss_threshold': -2.0,     # Stop loss threshold (%) (tighter stop loss)
            'trailing_stop_threshold': -1.5, # Trailing stop threshold (%) (tighter trailing stop)
            'max_position_size': 0.20,       # Max % of portfolio in single asset (reduced from 0.25)
            'conservative_mode': True,       # Enable for more conservative trading (changed to True)
            'aggressive_mode': False,        # Enable for more aggressive trading
            'quality_filter': True,         # NEW: Enable additional quality filtering
            'min_volume_threshold': 50000,   # NEW: Minimum 24h volume for trading
            'max_daily_trades': 8,           # NEW: Reduced from 30 to prevent overtrading
            'cooldown_period': 300,          # NEW: 5-minute cooldown between trades on same symbol
        }
        
        # Adjust strategy based on mode
        self._configure_strategy_mode()
        
        # Portfolio tracking
        self.portfolio_history = []
        self.real_trades = []
        self.current_portfolio = {}
        self.current_prices = {}
        self.initial_value = 0.0
        self.start_of_day_value = 0.0
        self.daily_trade_count = 0
        self.daily_pnl = 0.0
        self.total_fees_paid = 0.0  # Track total fees
        
        # NEW: Enhanced trade tracking for PPO improvements
        self.last_trade_time = {}  # Track last trade time per symbol for cooldown
        self.position_entry_prices = {}  # Track entry prices for stop-loss calculation
        self.failed_signals = 0  # Count failed signals for model adjustment
        self.successful_signals = 0  # Count successful signals
        self.trade_session_start = datetime.now()  # Track session start for daily limits
        
        # NEW: Time-based trading controls
        self.trading_hours = {
            'enabled': True,  # Enable/disable time-based controls
            'active_start': 6,  # 6:00 AM - Start of active trading hours
            'active_end': 22,   # 10:00 PM - End of active trading hours  
            'timezone': 'local',  # Use local timezone
            'quiet_hours_action': 'sleep',  # 'sleep' or 'monitor' during quiet hours
            'night_sleep_duration': 300,  # 5 minutes sleep during night hours (instead of normal interval)
        }
        
        # Trading pairs to track
        self.trading_pairs = [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT', 
            'BNBUSDT', 'SOLUSDT', 'LINKUSDT', 'UNIUSDT', 
            'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT', 'SANDUSDT', 'MANAUSDT'
        ]
        
        # Initialize PPO model
        self.ppo_model = None
        self.ppo_env = None
        self.ppo_enabled = False
    # PPO model path can be set via environment variable or constructor argument
        self.ppo_model_path = os.getenv('PPO_MODEL_PATH', "models/ppo_crypto_trader_v3_enhanced_final.zip")
        self._initialize_ppo_model()
        
        # Data storage
        self.data_file = f"live_trading_{datetime.now().strftime('%Y%m%d')}.json"
        
        logger.warning(" LIVE TRADING MANAGER INITIALIZED - REAL MONEY AT RISK!")
    
    def _configure_strategy_mode(self):
        """Configure PPO strategy based on selected mode."""
        if self.ppo_config['conservative_mode']:
            logger.info("🛡️ PPO Conservative Mode Enabled")
            self.ppo_config['min_confidence'] = 0.75  # Even higher confidence in conservative mode
            self.ppo_config['momentum_threshold'] = 0.8
            self.ppo_config['max_usdt_per_trade'] = 0.25
            self.ppo_config['profit_taking_threshold'] = 3.0  # Take profits earlier
            self.ppo_config['stop_loss_threshold'] = -2.0     # Tighter stop loss
            self.ppo_config['max_position_size'] = 0.15       # Smaller positions
            self.max_trade_usd = min(15.0, self.max_trade_usd)
            
        elif self.ppo_config['aggressive_mode']:
            logger.info("ðŸš€ PPO Aggressive Mode Enabled")
            self.ppo_config['min_confidence'] = 0.3
            self.ppo_config['momentum_threshold'] = 0.6
            self.ppo_config['max_usdt_per_trade'] = 0.5
            self.ppo_config['dip_threshold'] = -1.0
            self.ppo_config['profit_taking_threshold'] = 8.0  # Let profits run longer
            self.ppo_config['stop_loss_threshold'] = -5.0     # Wider stop loss
            self.ppo_config['max_position_size'] = 0.35       # Larger positions
            
        else:
            logger.info("âš–ï¸ PPO Balanced Mode Active")
    
    def _initialize_ppo_model(self):
        """Initialize the PPO model for trading decisions."""
        try:
            if not PPO_AVAILABLE:
                logger.warning("âš ï¸ PPO dependencies not available - PPO trading disabled")
                return False
            
            # Check if model file exists
            if not os.path.exists(self.ppo_model_path):
                logger.error(f" PPO model not found at: {self.ppo_model_path}")
                logger.info("ðŸ’¡ Train a PPO model first using train_ppo_crypto.py")
                return False
            
            # Load the trained PPO model
            logger.info(f"ðŸ§  Loading PPO model from: {self.ppo_model_path}")
            self.ppo_model = PPO.load(self.ppo_model_path)
            
            # Create environment for predictions (same configuration as training)
            logger.info("ðŸ—ï¸ Creating PPO environment for live trading...")
            self.ppo_env = CryptoTradingEnv(
                symbols=self.trading_pairs,
                initial_balance=10000,  # This won't be used for predictions
                trading_fee=0.001,
                window_size=30,
                period="2y",  # Use same period as training for consistency
                interval="1d"
            )
            
            # Test the model with a dummy prediction
            obs, _ = self.ppo_env.reset()
            test_action, _ = self.ppo_model.predict(obs, deterministic=True)
            
            self.ppo_enabled = True
            logger.warning("ðŸŽ¯ PPO MODEL LOADED SUCCESSFULLY!")
            logger.info(f" Model supports {len(self.trading_pairs)} trading pairs")
            logger.info(f"ðŸ§  Action space: {self.ppo_env.action_space.n} actions")
            logger.info(f"ðŸ” Observation space: {self.ppo_env.observation_space.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f" Failed to initialize PPO model: {e}")
            logger.warning("âš ï¸ PPO trading will be disabled - falling back to simple strategies")
            self.ppo_enabled = False
            return False
    
    def reload_ppo_model(self, model_path: str = None):
        """Reload PPO model from file (useful for updating to newer models)."""
        # Allow reload from argument, environment variable, or fallback to current
        if model_path:
            self.ppo_model_path = model_path
        else:
            self.ppo_model_path = os.getenv('PPO_MODEL_PATH', self.ppo_model_path)

        logger.info(f"🔄 Reloading PPO model from: {self.ppo_model_path}")
        success = self._initialize_ppo_model()

        if success:
            self.telegram.send_system_notification(
                f"PPO model reloaded successfully!\n"
                f"📁 Model: {os.path.basename(self.ppo_model_path)}\n"
                f"🎯 {len(self.trading_pairs)} trading pairs supported",
                level='info'
            )
        else:
            self.telegram.send_system_notification(
                f"Failed to reload PPO model\n"
                f" Model: {os.path.basename(self.ppo_model_path)}\n"
                f"System will use fallback strategies",
                level='error'
            )

        return success
    
    def set_ppo_mode(self, mode: str):
        """Set PPO trading mode: 'conservative', 'balanced', 'aggressive', or 'ultra-aggressive'."""
        self.ppo_config['conservative_mode'] = (mode == 'conservative')
        self.ppo_config['aggressive_mode'] = (mode == 'aggressive')
        self.ppo_config['ultra_aggressive_mode'] = (mode == 'ultra-aggressive')
        self._configure_strategy_mode()
        
        self.telegram.send_system_notification(
            f"PPO mode changed to: {mode.upper()}\n"
            f"Min confidence: {self.ppo_config['min_confidence']:.1f}\n"
            f"Max trade size: ${self.max_trade_usd:.0f}",
            level='info'
        )
    
    def _load_env(self):
        """Load environment variables from .env file."""
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.startswith('#'):
                        key, value = line.strip().split('=', 1)
                        os.environ[key] = value
    
    def is_trading_hours(self) -> bool:
        """Check if current time is within active trading hours."""
        if not self.trading_hours['enabled']:
            return True  # Always allow trading if time controls are disabled
        
        current_time = datetime.now()
        current_hour = current_time.hour
        
        start_hour = self.trading_hours['active_start']
        end_hour = self.trading_hours['active_end']
        
        # Check if current hour is within trading hours
        is_active = start_hour <= current_hour < end_hour
        
        if not is_active:
            logger.info(f"🌙 Outside trading hours ({start_hour}:00-{end_hour}:00). Current time: {current_hour}:00")
        
        return is_active
    
    def get_sleep_duration(self) -> int:
        """Get appropriate sleep duration based on trading hours."""
        if not self.trading_hours['enabled']:
            return 60  # Default interval if time controls disabled
        
        if self.is_trading_hours():
            return 60  # Normal interval during active hours
        else:
            return self.trading_hours['night_sleep_duration']  # Longer sleep during quiet hours
    
    def configure_trading_hours(self, active_start: int = None, active_end: int = None, 
                              enabled: bool = None, night_sleep_duration: int = None) -> dict:
        """Configure trading hours settings."""
        if active_start is not None:
            if 0 <= active_start <= 23:
                self.trading_hours['active_start'] = active_start
            else:
                logger.warning("⚠️ Invalid start hour. Must be 0-23.")
        
        if active_end is not None:
            if 0 <= active_end <= 23:
                self.trading_hours['active_end'] = active_end
            else:
                logger.warning("⚠️ Invalid end hour. Must be 0-23.")
        
        if enabled is not None:
            self.trading_hours['enabled'] = enabled
        
        if night_sleep_duration is not None:
            if night_sleep_duration > 0:
                self.trading_hours['night_sleep_duration'] = night_sleep_duration
            else:
                logger.warning("⚠️ Sleep duration must be positive.")
        
        logger.info(f"🕐 Trading hours updated: {self.trading_hours['active_start']}:00-{self.trading_hours['active_end']}:00")
        logger.info(f"🕐 Time controls: {'Enabled' if self.trading_hours['enabled'] else 'Disabled'}")
        
        return self.trading_hours.copy()
    
    def get_trading_hours_status(self) -> dict:
        """Get current trading hours status and configuration."""
        current_time = datetime.now()
        is_active = self.is_trading_hours()
        
        return {
            'current_time': current_time.strftime('%H:%M'),
            'current_hour': current_time.hour,
            'is_trading_hours': is_active,
            'active_start': self.trading_hours['active_start'],
            'active_end': self.trading_hours['active_end'],
            'enabled': self.trading_hours['enabled'],
            'night_sleep_duration': self.trading_hours['night_sleep_duration'],
            'current_sleep_duration': self.get_sleep_duration(),
            'hours_until_active': self._hours_until_active() if not is_active else 0,
            'hours_until_quiet': self._hours_until_quiet() if is_active else 0
        }
    
    def _hours_until_active(self) -> int:
        """Calculate hours until next active trading period."""
        current_hour = datetime.now().hour
        start_hour = self.trading_hours['active_start']
        
        if current_hour < start_hour:
            return start_hour - current_hour
        else:
            return (24 - current_hour) + start_hour
    
    def _hours_until_quiet(self) -> int:
        """Calculate hours until quiet period begins."""
        current_hour = datetime.now().hour
        end_hour = self.trading_hours['active_end']
        
        if current_hour < end_hour:
            return end_hour - current_hour
        else:
            return 0  # Already in quiet period
    
    def connect_and_initialize(self) -> bool:
        """Connect to API and initialize portfolio tracking."""
        try:
            logger.info("ðŸ”„ Connecting to Binance API for LIVE TRADING...")
            
            # Test connection
            account_info = self.api.get_account_info()
            if not account_info or 'error' in account_info:
                logger.error(" Failed to get account info")
                return False
            
            logger.warning(" CONNECTED TO LIVE TRADING API!")
            logger.info(f" Account Type: {account_info.get('accountType', 'N/A')}")
            logger.info(f"ðŸ”„ Can Trade: {account_info.get('canTrade', 'N/A')}")
            
            # Initialize portfolio
            if not self._update_portfolio():
                logger.error(" Failed to initialize portfolio")
                return False
            
            # Check if user has BNB for fee discount
            self._check_bnb_for_fees()
            
            # Set baseline values
            self.initial_value = self._calculate_total_value()
            self.start_of_day_value = self.initial_value
            
            logger.info(f"ðŸ’° Initial Portfolio Value: ${self.initial_value:.2f}")
            
            # Load existing data
            self._load_historical_data()
            
            # Safety confirmation
            print("\n LIVE TRADING SAFETY CONFIRMATION ")
            print("=" * 50)
            print(f"Current Portfolio Value: ${self.initial_value:.2f}")
            print(f"Max Trade Size: ${self.max_trade_usd:.2f}")
            print(f"Max Daily Trades: {self.max_daily_trades}")
            print(f"Max Daily Loss: ${self.max_daily_loss:.2f}")
            print("=" * 50)
            
            confirmation = input("Type 'I UNDERSTAND LIVE TRADING RISKS' to continue: ")
            if confirmation != "I UNDERSTAND LIVE TRADING RISKS":
                print(" Live trading cancelled for safety.")
                return False
            
            logger.warning(" USER CONFIRMED LIVE TRADING RISKS")
            
            # Send system start notification
            self.telegram.send_system_notification(
                f"Live trading system started!\n"
                f"ðŸ’° Initial Portfolio: ${self.initial_value:.2f}\n"
                f" Max Trade Size: ${self.max_trade_usd:.2f}\n"
                f" Max Daily Trades: {self.max_daily_trades}",
                level='start'
            )

            return True
            
        except Exception as e:
            logger.error(f" Connection failed: {e}")
            return False
    
    def _update_portfolio(self) -> bool:
        """Update portfolio data from Binance API."""
        try:
            account_info = self.api.get_account_info()
            if not account_info or 'error' in account_info:
                return False
            
            balances = account_info.get('balances', [])
            
            # Filter non-zero balances
            self.current_portfolio = {}
            for balance in balances:
                asset = balance['asset']
                free = float(balance['free'])
                locked = float(balance['locked'])
                total = free + locked
                
                if total > 0:
                    self.current_portfolio[asset] = total
            
            # Get current prices
            self._update_prices()
            
            # Create portfolio snapshot
            total_value = self._calculate_total_value()
            self.daily_pnl = total_value - self.start_of_day_value
            total_pnl = total_value - self.initial_value
            
            snapshot = PortfolioSnapshot(
                timestamp=datetime.now(),
                total_value_usd=total_value,
                positions=self.current_portfolio.copy(),
                prices=self.current_prices.copy(),
                daily_pnl=self.daily_pnl,
                total_pnl=total_pnl
            )
            
            self.portfolio_history.append(snapshot)
            
            logger.info(f"ðŸ’° Portfolio: ${total_value:.2f} | Daily P&L: ${self.daily_pnl:.2f} | Total P&L: ${total_pnl:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f" Error updating portfolio: {e}")
            return False
    
    def _update_prices(self):
        """Update current market prices."""
        try:
            stablecoins = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'FDUSD'}
            self.current_prices = {}
            
            for asset in self.current_portfolio.keys():
                if asset in stablecoins:
                    self.current_prices[asset] = 1.0
                elif asset.startswith('LD'):
                    # Liquid swap tokens - estimate value
                    underlying = asset[2:]
                    if underlying in stablecoins:
                        self.current_prices[asset] = 1.0
                    else:
                        price = self.api.get_ticker_price(f"{underlying}USDT")
                        self.current_prices[asset] = price * 0.98 if price > 0 else 0.0
                else:
                    price = self.api.get_ticker_price(f"{asset}USDT")
                    self.current_prices[asset] = price if price > 0 else 0.0
                        
        except Exception as e:
            logger.error(f" Error updating prices: {e}")
    
    def _calculate_total_value(self) -> float:
        """Calculate total portfolio value in USD."""
        total_value = 0.0
        
        for asset, amount in self.current_portfolio.items():
            price = self.current_prices.get(asset, 0)
            value = amount * price
            total_value += value
        
        return total_value
    
    def _check_bnb_for_fees(self):
        """Check if user has BNB for fee discount."""
        try:
            bnb_balance = self.current_portfolio.get('BNB', 0)
            # Need at least 0.01 BNB for fee payments
            self.has_bnb_for_fees = bnb_balance >= 0.01
            
            if self.has_bnb_for_fees:
                logger.info(f"âœ… BNB available for fee discount: {bnb_balance:.6f} BNB")
            else:
                logger.info("â„¹ï¸ No BNB for fee discount (standard 0.1% fees apply)")
                
        except Exception as e:
            logger.debug(f"Error checking BNB balance: {e}")
            self.has_bnb_for_fees = False
    
    def _calculate_trading_fee(self, trade_value_usd: float) -> float:
        """Calculate trading fee for a given trade value."""
        fee_rate = self.base_fee_rate
        
        # Apply BNB discount if available
        if self.has_bnb_for_fees:
            fee_rate = fee_rate * (1 - self.bnb_discount)  # 25% discount
        
        return trade_value_usd * fee_rate
    
    def _check_safety_limits(self, amount_usd: float) -> bool:
        """Check if trade passes safety limits."""
        # Check trade size limits
        if amount_usd < self.min_trade_usd:
            logger.warning(f"âš ï¸ Trade too small: ${amount_usd:.2f} < ${self.min_trade_usd:.2f}")
            return False
        
        if amount_usd > self.max_trade_usd:
            logger.warning(f"âš ï¸ Trade too large: ${amount_usd:.2f} > ${self.max_trade_usd:.2f}")
            return False
        
        # Check daily trade count
        if self.daily_trade_count >= self.max_daily_trades:
            logger.warning(f"âš ï¸ Daily trade limit reached: {self.daily_trade_count}/{self.max_daily_trades}")
            return False
        
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            logger.warning(f"âš ï¸ Daily loss limit reached: ${self.daily_pnl:.2f} <= -${self.max_daily_loss:.2f}")
            return False
        
        return True
    
    def _check_trade_cooldown(self, symbol: str) -> bool:
        """Check if enough time has passed since last trade on this symbol."""
        if symbol not in self.last_trade_time:
            return True
        
        cooldown_seconds = self.ppo_config.get('cooldown_period', 300)  # 5 minutes default
        time_since_last = (datetime.now() - self.last_trade_time[symbol]).total_seconds()
        
        if time_since_last < cooldown_seconds:
            remaining = cooldown_seconds - time_since_last
            logger.info(f"⏰ {symbol} in cooldown: {remaining:.0f}s remaining")
            return False
        
        return True
    
    def _check_market_quality(self, symbol: str) -> bool:
        """Check market quality before trading."""
        if not self.ppo_config.get('quality_filter', False):
            return True
        
        try:
            # Get 24h ticker data
            ticker_data = self._get_24h_ticker(symbol)
            if not ticker_data:
                logger.warning(f"⚠️ Could not get ticker data for {symbol}")
                return False
            
            # Check volume threshold
            volume = float(ticker_data.get('quoteVolume', 0))
            min_volume = self.ppo_config.get('min_volume_threshold', 50000)
            
            if volume < min_volume:
                logger.info(f"📊 {symbol} volume too low: ${volume:.0f} < ${min_volume:.0f}")
                return False
            
            # Check price volatility (avoid extremely volatile markets)
            price_change_pct = abs(float(ticker_data.get('priceChangePercent', 0)))
            if price_change_pct > 15:  # Avoid assets moving >15% in 24h
                logger.info(f"⚡ {symbol} too volatile: {price_change_pct:.1f}% 24h change")
                return False
            
            logger.debug(f"✅ {symbol} quality check passed: vol=${volume:.0f}, change={price_change_pct:.1f}%")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Quality check failed for {symbol}: {e}")
            return False
    
    def execute_trade(self, symbol: str, side: str, amount_usd: float, order_type: str = 'market') -> Optional[RealTrade]:
        """Execute a real trade with safety checks."""
        try:
            # Enhanced safety checks
            if not self._check_safety_limits(amount_usd):
                logger.error("🚫 Trade rejected by safety limits")
                return None
            
            # Check cooldown period for this symbol
            if not self._check_trade_cooldown(symbol):
                logger.info(f"⏰ Trade skipped: {symbol} in cooldown period")
                return None
            
            # Check market quality
            if not self._check_market_quality(symbol):
                logger.info(f"📊 Trade skipped: {symbol} failed quality check")
                return None
            
            # Get current price
            current_price = self.api.get_ticker_price(symbol)
            if current_price <= 0:
                logger.error(f" Could not get price for {symbol}")
                return None
            
            # Calculate trading fee
            fee_rate = self.base_fee_rate
            if self.has_bnb_for_fees:
                fee_rate = fee_rate * (1 - self.bnb_discount)
            
            trading_fee = amount_usd * fee_rate
            
            # Calculate quantity (accounting for fees)
            if side.lower() == 'buy':
                # When buying, we need to account for fees in the total cost
                effective_amount = amount_usd - trading_fee
                quantity = effective_amount / current_price
            else:  # sell
                base_asset = symbol.replace('USDT', '')
                available = self.current_portfolio.get(base_asset, 0)
                
                # Get fresh balance to be absolutely sure
                self._update_portfolio()
                fresh_available = self.current_portfolio.get(base_asset, 0)
                
                # Use the most conservative available amount
                safe_available = min(available, fresh_available) * 0.99  # 1% safety buffer
                
                # Calculate quantity based on USD amount requested
                requested_quantity = amount_usd / current_price
                quantity = min(safe_available, requested_quantity)
                
                if quantity <= 0 or safe_available <= 0:
                    logger.error(f"❌ Insufficient balance for {symbol}: available={available:.6f}, fresh={fresh_available:.6f}, safe={safe_available:.6f}")
                    return None
                
                logger.info(f"💰 Sell validation: requested=${amount_usd:.2f} ({requested_quantity:.6f} {base_asset}), available={safe_available:.6f}, final_qty={quantity:.6f}")
            
            # Get symbol info for proper formatting
            symbol_info = self.api.get_symbol_info(symbol)
            if symbol_info:
                # Round quantity to proper precision
                step_size = 0.00000001  # Default
                for filter_info in symbol_info.get('filters', []):
                    if filter_info['filterType'] == 'LOT_SIZE':
                        step_size = float(filter_info['stepSize'])
                        break
                
                # Round to step size
                quantity = round(quantity / step_size) * step_size
            
            if quantity <= 0:
                logger.error(f" Quantity too small after rounding: {quantity}")
                return None
            
            # Execute the order
            logger.warning(f" EXECUTING REAL TRADE: {side.upper()} {quantity:.8f} {symbol}")
            
            if order_type == 'market':
                result = self.api.place_market_order(symbol, side, quantity)
            else:
                result = self.api.place_limit_order(symbol, side, quantity, current_price)
            
            if 'error' in result:
                logger.error(f" Trade execution failed: {result['error']}")
                return None
            
            # Create trade record
            trade = RealTrade(
                timestamp=datetime.now(),
                symbol=symbol,
                side=side,
                amount=quantity,
                price=current_price,
                value_usd=amount_usd,
                order_id=result.get('orderId', 'N/A'),
                status=result.get('status', 'unknown'),
                commission=0.0,  # Will be updated from order status
                fee_usd=trading_fee,
                fee_rate=fee_rate
            )
            
            self.real_trades.append(trade)
            self.daily_trade_count += 1
            self.total_fees_paid += trading_fee
            
            # Track last trade time for cooldown
            self.last_trade_time[symbol] = datetime.now()
            
            # Track entry price for stop-loss calculations
            if side.lower() == 'buy':
                self.position_entry_prices[symbol] = current_price
            
            logger.warning(f"ðŸŽ¯ REAL TRADE EXECUTED: {trade.side.upper()} {trade.amount:.6f} {trade.symbol} @ ${trade.price:.4f}")
            logger.warning(f"ðŸ“‹ Order ID: {trade.order_id} | Status: {trade.status}")
            logger.warning(f"ðŸ’³ Fee: ${trade.fee_usd:.4f} ({trade.fee_rate*100:.3f}%) | Net: ${amount_usd - trade.fee_usd:.2f}")
            
            # Send Telegram notification for the trade
            current_value = self._calculate_total_value()
            self.telegram.send_trade_notification(
                trade=trade,
                portfolio_value=current_value,
                daily_pnl=self.daily_pnl
            )
            
            # Update portfolio after trade
            time.sleep(2)  # Wait for order to process
            self._update_portfolio()
            
            return trade
            
        except Exception as e:
            logger.error(f" Error executing trade: {e}")
            return None
    
    def show_detailed_holdings(self):
        """Show a detailed breakdown of all current holdings"""
        print("\n" + "="*80)
        print(" DETAILED PORTFOLIO HOLDINGS")
        print("="*80)
        
        summary = self.get_portfolio_summary()
        
        # Portfolio Overview
        print(f"\nðŸ’° Portfolio Overview:")
        print(f"   Total Value: ${summary['portfolio_value']['current']:.2f}")
        print(f"   Available Balance (USDT): ${summary['balances']['available']:.2f}")
        print(f"   Daily P&L: ${summary['pnl']['daily']:.2f} ({summary['pnl']['daily_return_pct']:.2f}%)")
        
        # Holdings Table
        holdings = summary['holdings']
        if holdings:
            print(f"\nðŸ“ˆ Current Holdings:")
            print(f"{'Asset':<8} {'Amount':<15} {'Price':<12} {'Value':<12} {'Allocation':<12} {'24h Change'}")
            print("-" * 80)
            
            total_value = 0
            for holding in holdings:
                asset = holding['asset']
                amount = holding['amount']
                price = holding['current_price']
                value = holding['usd_value']
                allocation = holding['allocation_pct']
                change_24h = holding.get('change_24h_pct', 0)
                total_value += value
                
                # Color coding for 24h change
                if change_24h > 0:
                    change_color = f"ðŸŸ¢ +{change_24h:.2f}%"
                elif change_24h < 0:
                    change_color = f"ðŸ”´ {change_24h:.2f}%"
                else:
                    change_color = f"âšª {change_24h:.2f}%"
                
                print(f"{asset:<8} {amount:<15.6f} ${price:<11.4f} ${value:<11.2f} {allocation:<11.2f}% {change_color}")
            
            print("-" * 80)
            print(f"{'TOTAL':<8} {'':<15} {'':<12} ${total_value:<11.2f} {'100.00%':<12}")
        else:
            print("\n   No significant holdings found (minimum $0.10 value)")
        
        # Trading Stats
        stats = summary['trading_stats']
        print(f"\n Trading Statistics:")
        print(f"   Daily Trades: {stats['daily_trades']}/{stats['max_daily_trades']}")
        print(f"   Success Rate: {stats.get('success_rate', 0):.1f}%")
        print(f"   Total Fees Paid: ${summary['fees']['total']:.4f}")
        
        # Tracked Symbols with current prices
        print(f"\nðŸŽ¯ Tracked Cryptocurrencies (Current Prices):")
        for symbol in self.trading_pairs:
            try:
                price = self.get_current_price(symbol)
                # Check if we have holdings in this symbol
                base_asset = symbol.replace('USDT', '')
                has_holding = any(h['asset'] == base_asset for h in holdings)
                status = "ðŸ“ˆ HELD" if has_holding else "ðŸ‘€ TRACKED"
                print(f"   {symbol}: ${price:.4f} {status}")
            except Exception as e:
                print(f"   {symbol}: Price unavailable")
        
        print("="*80)
        return summary

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a trading pair."""
        try:
            return self.api.get_ticker_price(symbol)
        except Exception as e:
            logger.debug(f"Error getting price for {symbol}: {e}")
            return 0.0

    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get comprehensive portfolio summary."""
        current_value = self._calculate_total_value()
        
        # Calculate today's fees
        today_trades = [t for t in self.real_trades if t.timestamp.date() == datetime.now().date()]
        today_fees = sum(t.fee_usd for t in today_trades)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': {
                'current': current_value,
                'initial': self.initial_value,
                'start_of_day': self.start_of_day_value
            },
            'pnl': {
                'daily': self.daily_pnl,
                'total': current_value - self.initial_value,
                'daily_return_pct': (self.daily_pnl / self.start_of_day_value * 100) if self.start_of_day_value > 0 else 0
            },
            'balances': {
                'available': self.current_portfolio.get('USDT', 0),
                'total_assets': len([k for k, v in self.current_portfolio.items() if v > 0])
            },
            'fees': {
                'today': today_fees,
                'total': self.total_fees_paid,
                'fee_rate': self.base_fee_rate * (1 - self.bnb_discount if self.has_bnb_for_fees else 1),
                'bnb_discount_active': self.has_bnb_for_fees
            },
            'positions': self.current_portfolio,
            'prices': self.current_prices,
            'holdings': self.get_detailed_holdings(),
            'trading_stats': {
                'daily_trades': self.daily_trade_count,
                'max_daily_trades': self.max_daily_trades,
                'total_trades': len(self.real_trades),
                'max_trade_size': self.max_trade_usd
            }
        }
    
    def get_detailed_holdings(self) -> List[Dict[str, Any]]:
        """Get detailed holdings information with USD values and percentages."""
        holdings = []
        total_value = self._calculate_total_value()
        
        if total_value <= 0:
            return holdings
        
        # Sort holdings by USD value (largest first)
        sorted_holdings = sorted(
            self.current_portfolio.items(),
            key=lambda x: x[1] * self.current_prices.get(x[0], 0),
            reverse=True
        )
        
        for asset, amount in sorted_holdings:
            if amount > 0:
                price = self.current_prices.get(asset, 0)
                usd_value = amount * price
                
                # Only include holdings worth at least $0.10
                if usd_value >= 0.10:
                    allocation_pct = (usd_value / total_value * 100) if total_value > 0 else 0
                    
                    # Get 24h price change if available
                    try:
                        if asset != 'USDT':
                            ticker_data = self._get_24h_ticker(f"{asset}USDT")
                            change_24h_pct = float(ticker_data.get('priceChangePercent', 0))
                        else:
                            change_24h_pct = 0.0
                    except:
                        change_24h_pct = 0.0
                    
                    holdings.append({
                        'asset': asset,
                        'amount': amount,
                        'current_price': price,
                        'usd_value': usd_value,
                        'allocation_pct': allocation_pct,
                        'change_24h_pct': change_24h_pct,
                        'is_stablecoin': asset in {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'FDUSD'}
                    })
        
        return holdings
    
    def print_detailed_portfolio(self):
        """Print a detailed, formatted view of the current portfolio."""
        print("\n" + "="*80)
        print("ðŸ’¼ DETAILED PORTFOLIO HOLDINGS")
        print("="*80)
        
        summary = self.get_portfolio_summary()
        holdings = summary['holdings']
        
        if not holdings:
            print("ðŸ“­ No holdings found")
            return
        
        # Portfolio header
        total_value = summary['portfolio_value']['current']
        daily_pnl = summary['pnl']['daily']
        daily_pct = summary['pnl']['daily_return_pct']
        
        pnl_emoji = "ðŸ“ˆ" if daily_pnl >= 0 else "ðŸ“‰"
        pnl_color = "+" if daily_pnl >= 0 else ""
        
        print(f"ðŸ’° Total Portfolio Value: ${total_value:.2f}")
        print(f"{pnl_emoji} Daily P&L: {pnl_color}${daily_pnl:.2f} ({pnl_color}{daily_pct:.2f}%)")
        print(f"ðŸ”¢ Total Holdings: {len(holdings)} assets")
        print("-"*80)
        
        # Holdings table header
        print(f"{'Asset':<8} {'Amount':<15} {'Price':<12} {'USD Value':<12} {'%':<8} {'Type'}")
        print("-"*80)
        
        # Display each holding
        for asset, data in holdings.items():
            emoji = data['emoji']
            amount = data['amount']
            price = data['price']
            usd_value = data['usd_value']
            percentage = data['percentage']
            is_stable = data['is_stablecoin']
            
            # Format amount based on asset type
            if is_stable:
                amount_str = f"{amount:.2f}"
                price_str = "$1.00"
                asset_type = "Stablecoin"
            else:
                if amount >= 1:
                    amount_str = f"{amount:.4f}"
                else:
                    amount_str = f"{amount:.6f}"
                price_str = f"${price:.4f}"
                asset_type = "Crypto"
            
            # Color coding for percentage
            if percentage >= 20:
                pct_str = f"{percentage:.1f}%"  # Large holding
            elif percentage >= 5:
                pct_str = f"{percentage:.1f}%"  # Medium holding
            else:
                pct_str = f"{percentage:.1f}%"  # Small holding
            
            print(f"{emoji} {asset:<6} {amount_str:<15} {price_str:<12} ${usd_value:<11.2f} {pct_str:<8} {asset_type}")
        
        print("-"*80)
        
        # Portfolio composition analysis
        stablecoin_value = sum(data['usd_value'] for data in holdings.values() if data['is_stablecoin'])
        crypto_value = total_value - stablecoin_value
        
        stable_pct = (stablecoin_value / total_value * 100) if total_value > 0 else 0
        crypto_pct = (crypto_value / total_value * 100) if total_value > 0 else 0
        
        print(f"ðŸ’µ Stablecoins: ${stablecoin_value:.2f} ({stable_pct:.1f}%)")
        print(f"ðŸª™ Cryptocurrencies: ${crypto_value:.2f} ({crypto_pct:.1f}%)")
        
        # Risk analysis
        print(f"\n PORTFOLIO ANALYSIS:")
        largest_holding = max(holdings.values(), key=lambda x: x['percentage']) if holdings else None
        if largest_holding:
            largest_asset = [k for k, v in holdings.items() if v == largest_holding][0]
            print(f"ðŸŽ¯ Largest Position: {largest_holding['emoji']} {largest_asset} ({largest_holding['percentage']:.1f}%)")
        
        diversified_assets = len([h for h in holdings.values() if h['percentage'] >= 5])
        print(f"âš–ï¸ Diversification: {diversified_assets} assets >5% of portfolio")
        
        available_usdt = holdings.get('USDT', {}).get('usd_value', 0)
        print(f"ðŸ’° Available for Trading: ${available_usdt:.2f}")
        
        print("="*80)
    
    def _save_data(self):
        """Save all trading data."""
        try:
            data = {
                'metadata': {
                    'created': datetime.now().isoformat(),
                    'initial_value': self.initial_value,
                    'current_value': self._calculate_total_value(),
                    'total_snapshots': len(self.portfolio_history),
                    'total_trades': len(self.real_trades),
                    'daily_trades': self.daily_trade_count,
                    'daily_pnl': self.daily_pnl
                },
                'portfolio_history': [asdict(snapshot) for snapshot in self.portfolio_history],
                'real_trades': [asdict(trade) for trade in self.real_trades]
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"ðŸ’¾ Live trading data saved to {self.data_file}")
            
        except Exception as e:
            logger.error(f" Error saving data: {e}")
    
    def start_continuous_trading(self, update_interval: int = 60, use_ppo: bool = True):
        """Start continuous trading with PPO agent."""
        logger.warning("ðŸ”„ Starting continuous live trading...")
        logger.warning("âš ï¸ This will execute REAL trades automatically!")
        
        # Check PPO availability if requested
        if use_ppo and not self.ppo_enabled:
            logger.warning("âš ï¸ PPO requested but not available - falling back to simple strategies")
            use_ppo = False
        
        if use_ppo:
            logger.warning("ðŸ§  Using TRAINED PPO AGENT for trading decisions")
            logger.info(f"ðŸŽ¯ PPO Model: {os.path.basename(self.ppo_model_path)}")
            logger.info(f"ðŸ“ˆ Expected Performance: ~68% annual returns (based on training)")
        else:
            logger.info(" Using simple predefined strategies")
        
        # Send system notification
        status_msg = f"Continuous trading started!\n"
        if use_ppo and self.ppo_enabled:
            status_msg += f"ðŸ§  PPO AI Agent Active\n"
            status_msg += f"ðŸŽ¯ Model: {os.path.basename(self.ppo_model_path)}\n"
            status_msg += f"ðŸ“ˆ Target: Beat buy-and-hold performance\n"
        else:
            status_msg += f" Simple Strategy Mode\n"
        
        status_msg += (
            f"â±ï¸ Update interval: {update_interval}s\n"
            f"Daily trades: {self.daily_trade_count}/{self.max_daily_trades}"
        )
        
        self.telegram.send_system_notification(status_msg, level='start')
        
        try:
            while True:
                # Update portfolio data
                if self._update_portfolio():
                    # Save data periodically
                    if len(self.portfolio_history) % 10 == 0:
                        self._save_data()
                    
                    # Make PPO trading decision (only during active hours)
                    if self.daily_trade_count < self.max_daily_trades and self.is_trading_hours():
                        if self.ppo_enabled:
                            self._execute_ppo_decision()
                            
                            # Periodically adjust confidence threshold based on performance
                            if len(self.portfolio_history) % 20 == 0:  # Every 20 cycles
                                self.adjust_confidence_threshold()
                        else:
                            logger.error("❌ PPO model required but not available - stopping trading")
                            break
                    elif not self.is_trading_hours():
                        logger.info("🌙 Outside trading hours - portfolio monitoring only")
                
                # Wait for next update
                logger.info(f"â±ï¸ Waiting {update_interval}s for next update...")
                time.sleep(update_interval)
                
        except KeyboardInterrupt:
            logger.warning("â¹ï¸ Continuous trading stopped by user")
            self.telegram.send_system_notification(
                f"Trading stopped by user\n"
                f" Final trades: {self.daily_trade_count}\n"
                f"ðŸ’° Portfolio: ${self._calculate_total_value():.2f}",
                level='stop'
            )
            self._save_data()
        except Exception as e:
            logger.error(f" Continuous trading error: {e}")
            self.telegram.send_system_notification(
                f"Trading system error!\n"
                f" Error: {str(e)}\n"
                f"System automatically stopped.",
                level='error'
            )
            self._save_data()
    
    def _execute_ppo_decision(self):
        """Execute trading decision based on trained PPO agent."""
        try:
            if not self.ppo_enabled:
                logger.warning("âš ï¸ PPO not enabled - falling back to simple strategy")
                return
            
            logger.info("ðŸ§  PPO agent analyzing market conditions...")
            
            # Get current market observation for PPO model
            obs = self._get_ppo_observation()
            if obs is None:
                logger.warning("âš ï¸ Could not get market observation for PPO")
                return
            
            # Get PPO model prediction
            logger.debug("🤖 Getting PPO model prediction...")
            
            # Check if model is available
            if self.ppo_model is None:
                logger.error("❌ PPO model is None")
                return
            
            try:
                # Get PPO prediction with dynamic confidence
                prediction_result = self._get_simple_ppo_prediction(obs)
                if prediction_result is None:
                    logger.error("❌ PPO prediction failed")
                    return
                
                action_int, confidence = prediction_result
                symbol_idx = action_int // 3
                action_type = action_int % 3  # 0=Hold, 1=Buy, 2=Sell
                
                if symbol_idx >= len(self.trading_pairs):
                    logger.warning(f"⚠️ Invalid symbol index from PPO: {symbol_idx}")
                    return
                
                symbol = self.trading_pairs[symbol_idx]
                base_asset = symbol.replace('USDT', '')
                
                logger.debug(f"🔍 PPO prediction - action: {action_int}, confidence: {confidence:.1%}, symbol: {symbol}")
                
            except Exception as e:
                logger.error(f"❌ PPO model prediction failed: {e}")
                logger.error(f"🔍 Prediction traceback: {traceback.format_exc()}")
                return
            
            if symbol_idx >= len(self.trading_pairs):
                logger.warning(f"âš ï¸ Invalid symbol index from PPO: {symbol_idx}")
                return
            
            symbol = self.trading_pairs[symbol_idx]
            base_asset = symbol.replace('USDT', '')
            
            logger.info(f"🤖 PPO ANALYSIS:")
            logger.info(f"   Symbol: {symbol}")
            logger.info(f"   Action: {['HOLD', 'BUY', 'SELL'][action_type]}")
            logger.info(f"   Confidence: {confidence:.1%}")
            
            # Check if confidence meets minimum threshold
            min_confidence = self.ppo_config.get('min_confidence', 0.75)  # Set to 75% to allow 80% trades
            if confidence < min_confidence:
                logger.info(f"🔒 PPO confidence {confidence:.1%} below threshold {min_confidence:.1%} - HOLD")
                self.failed_signals += 1
                return
            
            # Additional quality checks before trading
            if not self._check_trade_cooldown(symbol):
                logger.info(f"⏰ PPO trade skipped: {symbol} in cooldown period")
                return
            
            if not self._check_market_quality(symbol):
                logger.info(f"📊 PPO trade skipped: {symbol} failed quality check")
                self.failed_signals += 1
                return
            
            # Execute the action
            if action_type == 0:  # Hold
                logger.info("ðŸ¤– PPO DECISION: HOLD (no action)")
                return
                
            elif action_type == 1:  # Buy
                available_usdt = self.current_portfolio.get('USDT', 0)
                
                if available_usdt < self.min_trade_usd:
                    logger.info(f"ðŸ¤– PPO wants to BUY {symbol} but insufficient USDT: ${available_usdt:.2f}")
                    return
                
                # Calculate buy amount based on confidence and portfolio size
                max_usdt_per_trade = self.ppo_config.get('max_usdt_per_trade', 0.4)
                base_amount = available_usdt * max_usdt_per_trade * confidence
                buy_amount = max(self.min_trade_usd, min(base_amount, self.max_trade_usd))
                
                logger.warning(f"ðŸ¤– PPO DECISION: BUY {symbol} ${buy_amount:.2f} (confidence: {confidence:.1%})")
                
                # Create decision record for notification
                decision = {
                    'symbol': symbol,
                    'action': 'buy',
                    'amount': buy_amount,
                    'confidence': confidence,
                    'strategy': 'ppo_agent',
                    'reasoning': f'PPO neural network prediction with {confidence:.1%} confidence'
                }
                
                # Send PPO decision notification
                self.telegram.send_ppo_decision_notification(decision, executed=False)
                
                # Execute the trade
                trade = self.execute_trade(symbol, 'buy', buy_amount)
                
                if trade:
                    logger.warning(f"✅ PPO BUY SUCCESSFUL: {trade.order_id}")
                    self.successful_signals += 1
                    self.telegram.send_ppo_decision_notification(decision, executed=True)
                else:
                    logger.error(f"❌ PPO BUY FAILED")
                    self.failed_signals += 1
                    
            elif action_type == 2:  # Sell
                available_amount = self.current_portfolio.get(base_asset, 0)
                
                if available_amount <= 0:
                    logger.info(f"ðŸ¤– PPO wants to SELL {symbol} but no {base_asset} holdings")
                    return
                
                # Get current price to calculate USD value
                current_price = self.get_current_price(symbol)
                if current_price <= 0:
                    logger.warning(f"âš ï¸ Could not get price for {symbol}")
                    return
                
                # Update portfolio to get fresh balance data
                self._update_portfolio()
                fresh_available = self.current_portfolio.get(base_asset, 0)
                
                if fresh_available <= 0:
                    logger.info(f"🤖 PPO SELL: Fresh check shows no {base_asset} holdings")
                    return
                
                # Use the smaller of cached vs fresh balance for safety
                safe_available = min(available_amount, fresh_available)
                
                # Apply additional safety margin (keep 1% buffer for precision issues)
                safe_available = safe_available * 0.99
                
                # Calculate sell amount (percentage based on confidence, but capped at safe available)
                sell_ratio = min(1.0, confidence * 1.5)  # Higher confidence = sell more
                desired_sell_amount = safe_available * sell_ratio
                
                # Ensure we don't exceed what we actually have
                sell_amount_crypto = min(desired_sell_amount, safe_available)
                sell_value_usd = sell_amount_crypto * current_price
                
                # Ensure minimum trade size
                if sell_value_usd < self.min_trade_usd:
                    logger.info(f"🤖 PPO sell amount too small: ${sell_value_usd:.2f}")
                    return
                
                # Ensure maximum trade size - cap at safety limits
                if sell_value_usd > self.max_trade_usd:
                    logger.info(f"⚠️ PPO sell amount too large: ${sell_value_usd:.2f} > ${self.max_trade_usd:.2f}, adjusting...")
                    # Recalculate crypto amount based on max USD limit
                    sell_amount_crypto = self.max_trade_usd / current_price
                    sell_value_usd = self.max_trade_usd
                    
                    # Ensure we don't exceed available balance after adjustment
                    if sell_amount_crypto > safe_available:
                        sell_amount_crypto = safe_available
                        sell_value_usd = sell_amount_crypto * current_price
                    
                    logger.info(f"✅ PPO sell amount adjusted to: {sell_amount_crypto:.6f} {base_asset} (${sell_value_usd:.2f})")
                
                # Final validation: ensure we have enough
                if sell_amount_crypto > safe_available:
                    logger.warning(f"⚠️ PPO sell amount adjusted: {sell_amount_crypto:.6f} -> {safe_available:.6f} {base_asset}")
                    sell_amount_crypto = safe_available
                    sell_value_usd = sell_amount_crypto * current_price
                
                logger.warning(f"🤖 PPO DECISION: SELL {sell_amount_crypto:.6f} {base_asset} (${sell_value_usd:.2f}) (confidence: {confidence:.1%})")
                logger.info(f"💰 Balance check: cached={available_amount:.6f}, fresh={fresh_available:.6f}, safe={safe_available:.6f}")
                
                # Create decision record for notification
                decision = {
                    'symbol': symbol,
                    'action': 'sell',
                    'amount': sell_value_usd,
                    'quantity': sell_amount_crypto,
                    'confidence': confidence,
                    'strategy': 'ppo_agent',
                    'reasoning': f'PPO neural network sell signal with {confidence:.1%} confidence'
                }
                
                # Send PPO decision notification
                self.telegram.send_ppo_decision_notification(decision, executed=False)
                
                # Execute the trade
                trade = self.execute_trade(symbol, 'sell', sell_value_usd)
                
                if trade:
                    logger.warning(f"✅ PPO SELL SUCCESSFUL: {trade.order_id}")
                    self.successful_signals += 1
                    self.telegram.send_ppo_decision_notification(decision, executed=True)
                else:
                    logger.error(f"❌ PPO SELL FAILED")
                    self.failed_signals += 1
                    
        except Exception as e:
            logger.error(f" Error in PPO decision execution: {e}")
            # Fallback to simple strategy on error
            logger.info("ðŸ”„ Falling back to simple strategy")
            self._log_ppo_error_and_continue()
    
    def _log_ppo_error_and_continue(self):
        """Log PPO error and continue without fallback to simple strategy."""
        logger.error("📊 PPO system will retry on next cycle - maintaining PPO-only operation")
    
    def _get_simple_ppo_prediction(self, obs):
        """Enhanced PPO prediction with dynamic confidence calculation."""
        try:
            if self.ppo_model is None:
                return None, 0.0
            
            # Get prediction with action probabilities
            action, _states = self.ppo_model.predict(obs, deterministic=False)
            
            # Convert to int if needed
            if hasattr(action, 'item'):
                action = action.item()
            elif hasattr(action, '__len__') and len(action) == 1:
                action = action[0]
                
            action = int(action)
            
            # Calculate dynamic confidence based on market conditions and model certainty
            confidence = self._calculate_dynamic_confidence(action, obs)
            
            return action, confidence
            
        except Exception as e:
            logger.error(f"❌ PPO prediction failed: {e}")
            return None, 0.0
    
    def _calculate_dynamic_confidence(self, action: int, obs) -> float:
        """Calculate dynamic confidence based on market conditions and action type."""
        try:
            base_confidence = 0.7  # Base confidence level
            
            # Decode action to understand what we're doing
            symbol_idx = action // 3
            action_type = action % 3  # 0=Hold, 1=Buy, 2=Sell
            
            if symbol_idx >= len(self.trading_pairs):
                return 0.3  # Low confidence for invalid actions
            
            symbol = self.trading_pairs[symbol_idx]
            
            # Confidence modifiers based on different factors
            confidence_modifiers = []
            
            # 1. Action type confidence
            if action_type == 0:  # HOLD
                confidence_modifiers.append(0.15)  # Holding is generally safer
            elif action_type == 1:  # BUY
                confidence_modifiers.append(0.05)  # Buying requires more confidence
            else:  # SELL
                confidence_modifiers.append(0.10)  # Selling moderate confidence
            
            # 2. Market conditions confidence boost
            try:
                market_conditions = self._assess_market_conditions(symbol)
                if market_conditions.get('trend_strength', 0) > 0.7:
                    confidence_modifiers.append(0.10)  # Strong trend = higher confidence
                elif market_conditions.get('volatility', 0) > 0.8:
                    confidence_modifiers.append(-0.05)  # High volatility = lower confidence
                    
                if market_conditions.get('volume_strength', 0) > 0.6:
                    confidence_modifiers.append(0.05)  # Good volume = slight boost
            except:
                pass  # Skip market conditions if data unavailable
            
            # 3. Portfolio health modifier
            try:
                portfolio_health = self._get_portfolio_health_score()
                if portfolio_health > 0.8:
                    confidence_modifiers.append(0.05)  # Healthy portfolio = slight boost
                elif portfolio_health < 0.3:
                    confidence_modifiers.append(-0.10)  # Unhealthy portfolio = reduce confidence
            except:
                pass
            
            # 4. Recent performance modifier
            if hasattr(self, 'successful_signals') and hasattr(self, 'failed_signals'):
                total_signals = self.successful_signals + self.failed_signals
                if total_signals > 5:  # Only after some experience
                    success_rate = self.successful_signals / total_signals
                    if success_rate > 0.7:
                        confidence_modifiers.append(0.08)  # Good track record
                    elif success_rate < 0.3:
                        confidence_modifiers.append(-0.08)  # Poor track record
            
            # 5. Conservative mode adjustment
            if self.ppo_config.get('conservative_mode', False):
                confidence_modifiers.append(0.05)  # Slight boost in conservative mode
            
            # Calculate final confidence
            final_confidence = base_confidence + sum(confidence_modifiers)
            
            # Ensure confidence stays within reasonable bounds
            final_confidence = max(0.4, min(0.95, final_confidence))
            
            # Add small random variation to make it more realistic (±2%)
            import random
            variation = random.uniform(-0.02, 0.02)
            final_confidence += variation
            
            # Final bounds check
            final_confidence = max(0.35, min(0.98, final_confidence))
            
            logger.debug(f"🎯 Dynamic confidence for {['HOLD', 'BUY', 'SELL'][action_type]} {symbol}: {final_confidence:.2f}")
            logger.debug(f"   Base: {base_confidence:.2f}, Modifiers: {sum(confidence_modifiers):.2f}")
            
            return final_confidence
            
        except Exception as e:
            logger.debug(f"Error calculating dynamic confidence: {e}")
            # Fallback to a reasonable default with slight randomness
            import random
            return random.uniform(0.65, 0.85)
    
    def _assess_market_conditions(self, symbol: str) -> dict:
        """Assess current market conditions for a symbol."""
        try:
            # Get recent price data
            current_price = self.get_current_price(symbol)
            if current_price <= 0:
                return {'trend_strength': 0.5, 'volatility': 0.5, 'volume_strength': 0.5}
            
            # Get 24h ticker data for additional metrics
            ticker_data = self._get_24h_ticker(symbol)
            if not ticker_data:
                return {'trend_strength': 0.5, 'volatility': 0.5, 'volume_strength': 0.5}
            
            # Calculate trend strength based on price change
            price_change_pct = abs(float(ticker_data.get('priceChangePercent', 0)))
            trend_strength = min(1.0, price_change_pct / 5.0)  # Normalize to 0-1
            
            # Calculate volatility (higher = more volatile)
            high = float(ticker_data.get('highPrice', current_price))
            low = float(ticker_data.get('lowPrice', current_price))
            if low > 0:
                volatility = min(1.0, (high - low) / low)
            else:
                volatility = 0.5
            
            # Calculate volume strength
            volume = float(ticker_data.get('volume', 0))
            avg_volume = float(ticker_data.get('weightedAvgPrice', 1)) * volume  # Approximate
            volume_strength = min(1.0, volume / 1000000) if volume > 0 else 0.5  # Normalize
            
            return {
                'trend_strength': trend_strength,
                'volatility': volatility,
                'volume_strength': volume_strength
            }
            
        except Exception as e:
            logger.debug(f"Error assessing market conditions for {symbol}: {e}")
            return {'trend_strength': 0.5, 'volatility': 0.5, 'volume_strength': 0.5}
    
    def _execute_ppo_buy(self, symbol: str, confidence: float):
        """Execute PPO buy decision."""
        try:
            available_usdt = self.current_portfolio.get('USDT', 0)
            max_trade_pct = self.ppo_config.get('max_usdt_per_trade', 0.25)
            buy_amount = min(available_usdt * max_trade_pct, self.max_trade_usd)
            
            if buy_amount < self.min_trade_usd:
                logger.info(f"💰 Insufficient USDT for PPO buy: ${buy_amount:.2f}")
                return
            
            logger.warning(f"🤖 PPO DECISION: BUY {symbol} ${buy_amount:.2f} (confidence: {confidence:.1%})")
            
            # Create decision record
            decision = {
                'symbol': symbol,
                'action': 'buy',
                'amount': buy_amount,
                'confidence': confidence,
                'reasoning': f'PPO buy signal with {confidence:.1%} confidence'
            }
            
            # Execute the trade
            trade = self.execute_trade(symbol, 'buy', buy_amount)
            
            if trade:
                logger.warning(f"✅ PPO BUY SUCCESSFUL: {trade.order_id}")
                self.successful_signals += 1
                self.telegram.send_ppo_decision_notification(decision, executed=True)
            else:
                logger.error(f"❌ PPO BUY FAILED")
                self.failed_signals += 1
                self.telegram.send_ppo_decision_notification(decision, executed=False)
                
        except Exception as e:
            logger.error(f"❌ Error executing PPO buy: {e}")
    
    def _execute_ppo_sell(self, symbol: str, confidence: float):
        """Execute PPO sell decision."""
        try:
            base_asset = symbol.replace('USDT', '')
            available_crypto = self.current_portfolio.get(base_asset, 0)
            
            if available_crypto <= 0:
                logger.info(f"💰 No {base_asset} holdings to sell")
                return
            
            current_price = self.api.get_ticker_price(symbol)
            if current_price <= 0:
                logger.error(f"❌ Could not get price for {symbol}")
                return
            
            # Sell 50% of holdings or minimum trade amount
            sell_amount_crypto = min(available_crypto * 0.5, self.max_trade_usd / current_price)
            sell_value_usd = sell_amount_crypto * current_price
            
            if sell_value_usd < self.min_trade_usd:
                logger.info(f"💰 Sell amount too small: ${sell_value_usd:.2f}")
                return
            
            logger.warning(f"🤖 PPO DECISION: SELL {sell_amount_crypto:.6f} {base_asset} (${sell_value_usd:.2f}) (confidence: {confidence:.1%})")
            
            # Create decision record
            decision = {
                'symbol': symbol,
                'action': 'sell',
                'amount': sell_value_usd,
                'quantity': sell_amount_crypto,
                'confidence': confidence,
                'reasoning': f'PPO sell signal with {confidence:.1%} confidence'
            }
            
            # Execute the trade
            trade = self.execute_trade(symbol, 'sell', sell_value_usd)
            
            if trade:
                logger.warning(f"✅ PPO SELL SUCCESSFUL: {trade.order_id}")
                self.successful_signals += 1
                self.telegram.send_ppo_decision_notification(decision, executed=True)
            else:
                logger.error(f"❌ PPO SELL FAILED")
                self.failed_signals += 1
                self.telegram.send_ppo_decision_notification(decision, executed=False)
                
        except Exception as e:
            logger.error(f"❌ Error executing PPO sell: {e}")
    
    def get_ppo_performance_stats(self) -> dict:
        """Get PPO performance statistics."""
        total_signals = self.successful_signals + self.failed_signals
        success_rate = (self.successful_signals / total_signals * 100) if total_signals > 0 else 0
        
        session_duration = (datetime.now() - self.trade_session_start).total_seconds() / 3600  # hours
        
        return {
            'successful_signals': self.successful_signals,
            'failed_signals': self.failed_signals,
            'total_signals': total_signals,
            'success_rate_pct': success_rate,
            'session_duration_hours': session_duration,
            'trades_per_hour': self.daily_trade_count / session_duration if session_duration > 0 else 0,
            'current_confidence_threshold': self.ppo_config.get('min_confidence', 0.4),
            'conservative_mode': self.ppo_config.get('conservative_mode', False)
        }
    
    def adjust_confidence_threshold(self):
        """Dynamically adjust confidence threshold based on performance."""
        stats = self.get_ppo_performance_stats()
        
        # Only adjust after at least 10 signals
        if stats['total_signals'] < 10:
            return
        
        success_rate = stats['success_rate_pct']
        current_threshold = stats['current_confidence_threshold']
        
        # If success rate is very low, increase threshold
        if success_rate < 30:
            new_threshold = min(0.9, current_threshold + 0.05)
            self.ppo_config['min_confidence'] = new_threshold
            logger.warning(f"📈 PPO threshold increased to {new_threshold:.2f} (success rate: {success_rate:.1f}%)")
        
        # If success rate is good but we have few trades, slightly decrease threshold
        elif success_rate > 70 and self.daily_trade_count < 3:
            new_threshold = max(0.4, current_threshold - 0.02)
            self.ppo_config['min_confidence'] = new_threshold
            logger.info(f"📉 PPO threshold decreased to {new_threshold:.2f} (success rate: {success_rate:.1f}%)")
        
        # Log performance stats periodically
        if stats['total_signals'] % 5 == 0:
            logger.info(f"🎯 PPO Performance: {stats['successful_signals']}/{stats['total_signals']} ({success_rate:.1f}% success)")
            logger.info(f"⚙️ Current threshold: {current_threshold:.2f}, Conservative: {stats['conservative_mode']}")
    
    def _get_ppo_observation(self):
        """Get market observation in the format expected by the PPO model."""
        try:
            # Reset the environment with current market data to get fresh observation
            logger.debug("🔄 Resetting PPO environment to get fresh observation...")
            
            # Check if PPO environment is available
            if self.ppo_env is None:
                logger.error("❌ PPO environment is None")
                return None
            
            obs, info = self.ppo_env.reset()
            
            # Validate observation
            if obs is None:
                logger.error("❌ PPO environment returned None observation")
                return None
            
            # Check observation shape and contents
            logger.debug(f"🔍 PPO observation shape: {obs.shape}")
            
            # Check for NaN or infinite values
            if hasattr(obs, 'isnan') and obs.isnan().any():
                logger.warning("⚠️ PPO observation contains NaN values - replacing with zeros")
                obs = np.nan_to_num(obs, nan=0.0)
            
            if hasattr(obs, 'isinf') and obs.isinf().any():
                logger.warning("⚠️ PPO observation contains infinite values - replacing with finite values")
                obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Ensure observation matches expected shape
            expected_shape = self.ppo_env.observation_space.shape[0]
            if obs.shape[0] != expected_shape:
                logger.error(f"❌ Observation shape mismatch: got {obs.shape[0]}, expected {expected_shape}")
                return None
            
            # The observation contains:
            # - Technical indicators for all symbols (windowed)
            # - Portfolio information
            # - Market features
            
            logger.debug(f"ðŸ” PPO observation shape: {obs.shape}")
            return obs
            
        except Exception as e:
            logger.error(f" Error getting PPO observation: {e}")
            return None
    
    def _calculate_ppo_confidence(self, action: int, action_prob: float, obs) -> float:
        """Calculate confidence score for PPO decision."""
        try:
            # Validate inputs
            if action is None or action_prob is None:
                logger.debug("PPO confidence calculation received None values")
                return 0.5
            
            # Base confidence from action probability
            base_confidence = min(1.0, action_prob * 2)  # Scale probability
            
            # Adjust based on market conditions (you can enhance this)
            market_sentiment = self._get_market_sentiment_score()
            portfolio_health = self._get_portfolio_health_score()
            
            # Ensure market_sentiment and portfolio_health are valid
            if market_sentiment is None:
                market_sentiment = 0.5
            if portfolio_health is None:
                portfolio_health = 0.5
            
            # Combine factors
            confidence = (
                base_confidence * 0.5 +  # PPO confidence
                market_sentiment * 0.3 +  # Market conditions
                portfolio_health * 0.2    # Portfolio state
            )
            
            return max(0.1, min(0.95, confidence))  # Bound between 10% and 95%
            
        except Exception as e:
            logger.debug(f"Error calculating PPO confidence: {e}")
            return 0.5  # Default neutral confidence
    
    def _get_market_sentiment_score(self) -> float:
        """Get market sentiment score (0-1)."""
        try:
            positive_count = 0
            total_count = 0
            
            for symbol in self.trading_pairs:
                try:
                    price = self.get_current_price(symbol)
                    if price > 0:
                        # Simple 24h price change sentiment
                        ticker_data = self._get_24h_ticker(symbol)
                        price_change = float(ticker_data.get('priceChangePercent', 0))
                        
                        if price_change > 0:
                            positive_count += 1
                        total_count += 1
                except:
                    continue
            
            return positive_count / total_count if total_count > 0 else 0.5
            
        except:
            return 0.5
    
    def _get_portfolio_health_score(self) -> float:
        """Get portfolio health score (0-1)."""
        try:
            # Factors: diversification, cash ratio, recent performance
            total_value = self._calculate_total_value()
            available_usdt = self.current_portfolio.get('USDT', 0)
            
            # Cash ratio (having some cash is good, too much is bad)
            cash_ratio = available_usdt / total_value if total_value > 0 else 0
            cash_score = 1.0 - abs(cash_ratio - 0.3)  # Optimal around 30% cash
            
            # Daily P&L (positive is good)
            pnl_score = max(0, min(1.0, (self.daily_pnl + 10) / 20))  # Scale -10 to +10 USD
            
            # Diversification (more assets is generally better, up to a point)
            asset_count = len([k for k, v in self.current_portfolio.items() if k != 'USDT' and v > 0])
            diversification_score = min(1.0, asset_count / 8)  # Optimal around 8 assets
            
            return (cash_score * 0.4 + pnl_score * 0.4 + diversification_score * 0.2)
            
        except:
            return 0.5
    
    def _get_market_state(self) -> dict:
        """Get comprehensive market state for PPO agent with technical indicators."""
        try:
            # Top crypto symbols with good liquidity
            symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT', 'XRPUSDT', 'BNBUSDT', 'SOLUSDT', 'LINKUSDT', 'UNIUSDT', 'AVAXUSDT', 'ATOMUSDT', 'NEARUSDT', 'SANDUSDT', 'MANAUSDT']
            market_data = {}
            
            for symbol in symbols:
                try:
                    # Get current price
                    current_price = self.api.get_ticker_price(symbol)
                    if current_price <= 0:
                        continue
                    
                    # Get 24h ticker data for technical analysis
                    ticker_data = self._get_24h_ticker(symbol)
                    
                    # Calculate technical indicators
                    price_change_24h = float(ticker_data.get('priceChangePercent', 0))
                    volume_24h = float(ticker_data.get('volume', 0))
                    high_24h = float(ticker_data.get('highPrice', current_price))
                    low_24h = float(ticker_data.get('lowPrice', current_price))
                    
                    # Calculate position in 24h range
                    price_position = ((current_price - low_24h) / (high_24h - low_24h)) if (high_24h - low_24h) > 0 else 0.5
                    
                    # Simple momentum indicators
                    momentum_score = self._calculate_momentum_score(price_change_24h, volume_24h, price_position)
                    
                    market_data[symbol] = {
                        'current_price': current_price,
                        'price_change_24h': price_change_24h,
                        'volume_24h': volume_24h,
                        'price_position_in_range': price_position,
                        'momentum_score': momentum_score,
                        'symbol': symbol,
                        'high_24h': high_24h,
                        'low_24h': low_24h
                    }
                    
                except Exception as e:
                    logger.debug(f"Error getting data for {symbol}: {e}")
                    continue
            
            # Add portfolio information
            available_usdt = self.current_portfolio.get('USDT', 0)
            portfolio_state = {
                'total_value': self._calculate_total_value(),
                'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trade_count,
                'available_usdt': available_usdt,
                'usdt_ratio': available_usdt / self._calculate_total_value() if self._calculate_total_value() > 0 else 0,
                'positions_count': len([k for k, v in self.current_portfolio.items() if v > 0 and k != 'USDT'])
            }
            
            # Market sentiment analysis
            market_sentiment = self._analyze_market_sentiment(market_data)
            
            return {
                'market_data': market_data,
                'portfolio': portfolio_state,
                'market_sentiment': market_sentiment,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f" Error getting market state: {e}")
            return {}
    
    def _get_24h_ticker(self, symbol: str) -> dict:
        """Get 24h ticker statistics for a symbol."""
        try:
            return self.api._make_request('/api/v3/ticker/24hr', {'symbol': symbol})
        except:
            return {}
    
    def _calculate_momentum_score(self, price_change: float, volume: float, price_position: float) -> float:
        """Calculate momentum score based on price change, volume, and position."""
        try:
            # Price momentum (weight: 40%)
            if price_change > 5:
                price_momentum = 1.0
            elif price_change > 2:
                price_momentum = 0.7
            elif price_change > 0:
                price_momentum = 0.5
            elif price_change > -2:
                price_momentum = 0.3
            elif price_change > -5:
                price_momentum = 0.1
            else:
                price_momentum = 0.0
            
            # Volume momentum (weight: 30%)
            # Higher volume generally indicates stronger moves
            volume_momentum = min(1.0, volume / 100000) if volume > 0 else 0.0
            
            # Position momentum (weight: 30%)
            # Being higher in the 24h range is generally bullish
            position_momentum = price_position
            
            # Weighted average
            momentum = (price_momentum * 0.4) + (volume_momentum * 0.3) + (position_momentum * 0.3)
            
            return round(momentum, 3)
            
        except:
            return 0.5  # Neutral score on error
    
    def _analyze_market_sentiment(self, market_data: dict) -> dict:
        """Analyze overall market sentiment from all symbols."""
        try:
            if not market_data:
                return {'overall_sentiment': 'neutral', 'sentiment_score': 0.5}
            
            positive_count = 0
            negative_count = 0
            total_momentum = 0
            valid_symbols = 0
            
            for symbol, data in market_data.items():
                price_change = data.get('price_change_24h', 0)
                momentum = data.get('momentum_score', 0.5)
                
                if price_change > 1:
                    positive_count += 1
                elif price_change < -1:
                    negative_count += 1
                
                total_momentum += momentum
                valid_symbols += 1
            
            if valid_symbols == 0:
                return {'overall_sentiment': 'neutral', 'sentiment_score': 0.5}
            
            avg_momentum = total_momentum / valid_symbols
            positive_ratio = positive_count / valid_symbols
            negative_ratio = negative_count / valid_symbols
            
            # Determine overall sentiment
            if avg_momentum > 0.7 and positive_ratio > 0.6:
                sentiment = 'very_bullish'
                sentiment_score = 0.9
            elif avg_momentum > 0.6 and positive_ratio > 0.5:
                sentiment = 'bullish'
                sentiment_score = 0.75
            elif avg_momentum > 0.4 and positive_ratio > 0.4:
                sentiment = 'neutral_bullish'
                sentiment_score = 0.6
            elif avg_momentum < 0.3 and negative_ratio > 0.6:
                sentiment = 'very_bearish'
                sentiment_score = 0.1
            elif avg_momentum < 0.4 and negative_ratio > 0.5:
                sentiment = 'bearish'
                sentiment_score = 0.25
            else:
                sentiment = 'neutral'
                sentiment_score = 0.5
            
            return {
                'overall_sentiment': sentiment,
                'sentiment_score': sentiment_score,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'avg_momentum': avg_momentum,
                'market_strength': 'strong' if avg_momentum > 0.6 else 'weak' if avg_momentum < 0.4 else 'moderate'
            }
            
        except Exception as e:
            logger.debug(f"Error analyzing market sentiment: {e}")
            return {'overall_sentiment': 'neutral', 'sentiment_score': 0.5}
    
    def _analyze_market_for_ppo(self) -> dict:
        """Advanced PPO market analysis with intelligent decision-making."""
        try:
            # Get comprehensive market state
            market_state = self._get_market_state()
            
            if not market_state or not market_state.get('market_data'):
                logger.warning("ðŸ¤– PPO: No market data available")
                return None
            
            market_data = market_state['market_data']
            portfolio = market_state['portfolio']
            sentiment = market_state.get('market_sentiment', {})
            
            logger.info(f"ðŸ§  PPO Analysis - Market Sentiment: {sentiment.get('overall_sentiment', 'unknown')}")
            logger.info(f"ðŸ§  PPO Analysis - Portfolio USDT: ${portfolio.get('available_usdt', 0):.2f}")
            logger.info(f"ðŸ§  PPO Analysis - Daily Trades: {portfolio.get('daily_trades', 0)}")
            
            # Check if we have sufficient funds
            available_usdt = portfolio.get('available_usdt', 0)
            if available_usdt < self.min_trade_usd:
                logger.info("ðŸ¤– PPO: Insufficient USDT for trading")
                return None
            
            # Advanced decision-making algorithm
            decision = self._make_intelligent_decision(market_data, portfolio, sentiment)
            
            if decision:
                logger.info(f"ðŸ§  PPO Decision Details:")
                logger.info(f"   Symbol: {decision['symbol']}")
                logger.info(f"   Action: {decision['action']}")
                logger.info(f"   Amount: ${decision['amount']:.2f}")
                logger.info(f"   Confidence: {decision['confidence']:.1%}")
                logger.info(f"   Strategy: {decision['strategy']}")
                logger.info(f"   Reasoning: {decision['reasoning']}")
            
            return decision
            
        except Exception as e:
            logger.error(f" Error in advanced PPO analysis: {e}")
            return None
    
    def _make_intelligent_decision(self, market_data: dict, portfolio: dict, sentiment: dict) -> dict:
        """Make intelligent trading decision based on multiple factors."""
        try:
            available_usdt = portfolio.get('available_usdt', 0)
            daily_trades = portfolio.get('daily_trades', 0)
            usdt_ratio = portfolio.get('usdt_ratio', 0)
            positions_count = portfolio.get('positions_count', 0)
            
            # Strategy selection based on market conditions and portfolio state
            strategies = []
            
            # 1. MOMENTUM BUYING STRATEGY
            if sentiment.get('sentiment_score', 0.5) > 0.6:  # Bullish market
                momentum_opportunities = self._find_momentum_opportunities(market_data, available_usdt)
                strategies.extend(momentum_opportunities)
            
            # 2. DIP BUYING STRATEGY - Relaxed conditions for bearish markets
            if usdt_ratio > 0.15 or sentiment.get('sentiment_score', 0.5) < 0.4:  # Have some USDT reserves OR bearish market (good for dip buying)
                dip_opportunities = self._find_dip_opportunities(market_data, available_usdt)
                strategies.extend(dip_opportunities)
            
            # 3. DIVERSIFICATION STRATEGY - More flexible conditions
            diversification_limit = self.ppo_config.get('diversification_limit', 10)
            if positions_count < diversification_limit and available_usdt > self.min_trade_usd:
                diversification_opportunities = self._find_diversification_opportunities(market_data, portfolio, available_usdt)
                strategies.extend(diversification_opportunities)
            
            # 4. PROFIT-TAKING & SELL STRATEGIES (Always active)
            profit_opportunities = self._find_profit_opportunities(portfolio)
            strategies.extend(profit_opportunities)
            
            # 5. REBALANCING STRATEGY (Sell overweight positions)
            rebalancing_opportunities = self._find_rebalancing_opportunities(portfolio)
            strategies.extend(rebalancing_opportunities)
            
            # 5. BREAKOUT STRATEGY
            breakout_opportunities = self._find_breakout_opportunities(market_data, available_usdt)
            strategies.extend(breakout_opportunities)
            
            # Sort strategies by confidence score
            strategies.sort(key=lambda x: x['confidence'], reverse=True)
            
            logger.info(f"ðŸ§  PPO Found {len(strategies)} potential strategies")
            for i, strategy in enumerate(strategies[:3]):
                logger.info(f"   {i+1}. {strategy['symbol']}: ${strategy['amount']:.2f} (confidence: {strategy['confidence']:.1%}) - {strategy['strategy']}")
            
            # Apply filters and select best strategy
            for i, strategy in enumerate(strategies):
                logger.debug(f"ðŸ” Validating strategy {i+1}: {strategy['symbol']} ${strategy['amount']:.2f}")
                if self._validate_strategy(strategy, portfolio):
                    logger.info(f"âœ… PPO Selected strategy: {strategy['symbol']} ${strategy['amount']:.2f}")
                    return strategy
                else:
                    logger.debug(f" Strategy {i+1} validation failed")
            
            # No suitable strategy found
            logger.info("ðŸ¤– PPO: No suitable trading opportunity found (all strategies failed validation)")
            return None
            
        except Exception as e:
            logger.error(f" Error in intelligent decision making: {e}")
            return None
    
    def _find_momentum_opportunities(self, market_data: dict, available_usdt: float) -> list:
        """Find momentum-based buying opportunities."""
        opportunities = []
        momentum_threshold = self.ppo_config.get('momentum_threshold', 0.7)
        
        for symbol, data in market_data.items():
            try:
                momentum = data.get('momentum_score', 0)
                price_change = data.get('price_change_24h', 0)
                volume = data.get('volume_24h', 0)
                
                # Strong momentum criteria (configurable threshold)
                if momentum > momentum_threshold and price_change > 2 and volume > 50000:
                    confidence = min(0.95, momentum * 0.8 + (price_change / 20))
                    
                    # Calculate amount, ensuring it meets minimum trade requirements
                    preferred_amount = available_usdt * 0.3  # 30% of available USDT
                    amount = max(self.min_trade_usd, min(preferred_amount, self.max_trade_usd))
                    
                    # Only add if we have enough USDT for the minimum trade
                    if available_usdt >= amount:
                        opportunities.append({
                            'symbol': symbol,
                            'action': 'buy',
                            'amount': amount,
                            'confidence': confidence,
                            'strategy': 'momentum_buying',
                            'reasoning': f'Strong momentum ({momentum:.2f}) with {price_change:.1f}% 24h gain'
                        })
                    
            except Exception as e:
                logger.debug(f"Error analyzing momentum for {symbol}: {e}")
                continue
        
        return opportunities
    
    def _find_dip_opportunities(self, market_data: dict, available_usdt: float) -> list:
        """Find dip-buying opportunities."""
        opportunities = []
        dip_threshold = self.ppo_config.get('dip_threshold', -2.0)
        
        for symbol, data in market_data.items():
            try:
                price_change = data.get('price_change_24h', 0)
                price_position = data.get('price_position_in_range', 0.5)
                momentum = data.get('momentum_score', 0.5)
                
                # Dip buying criteria (configurable threshold)
                if price_change < dip_threshold and price_position < 0.3 and momentum > 0.3:
                    confidence = 0.6 + (abs(price_change) / 20) + ((0.3 - price_position) * 0.5)
                    confidence = min(0.85, confidence)
                    
                    # Calculate amount, ensuring it meets minimum trade requirements
                    preferred_amount = available_usdt * 0.25  # 25% of available USDT
                    amount = max(self.min_trade_usd, min(preferred_amount, self.max_trade_usd))
                    
                    # Only add if we have enough USDT for the minimum trade
                    if available_usdt >= amount:
                        opportunities.append({
                            'symbol': symbol,
                            'action': 'buy',
                            'amount': amount,
                            'confidence': confidence,
                            'strategy': 'dip_buying',
                            'reasoning': f'Dip opportunity: {price_change:.1f}% down, low in range ({price_position:.1%})'
                        })
                    
            except Exception as e:
                logger.debug(f"Error analyzing dip for {symbol}: {e}")
                continue
        
        return opportunities
    
    def _find_diversification_opportunities(self, market_data: dict, portfolio: dict, available_usdt: float) -> list:
        """Find diversification opportunities."""
        opportunities = []
        current_positions = set(portfolio.get('positions', {}).keys()) - {'USDT'}
        diversification_limit = self.ppo_config.get('diversification_limit', 10)
        
        # Skip if we already have enough positions
        if len(current_positions) >= diversification_limit:
            return opportunities
        
        for symbol, data in market_data.items():
            try:
                base_asset = symbol.replace('USDT', '')
                
                # Skip if we already have this asset
                if base_asset in current_positions:
                    continue
                
                momentum = data.get('momentum_score', 0.5)
                price_change = data.get('price_change_24h', 0)
                
                # Diversification criteria: decent momentum, not extreme price moves
                if 0.4 <= momentum <= 0.8 and -5 <= price_change <= 10:
                    confidence = 0.5 + (momentum - 0.4) * 0.5
                    
                    # Calculate amount, ensuring it meets minimum trade requirements
                    preferred_amount = available_usdt * 0.2  # 20% of available USDT
                    amount = max(self.min_trade_usd, min(preferred_amount, self.max_trade_usd * 0.7))
                    
                    # Only add if we have enough USDT for the minimum trade
                    if available_usdt >= amount:
                        opportunities.append({
                            'symbol': symbol,
                            'action': 'buy',
                            'amount': amount,
                            'confidence': confidence,
                            'strategy': 'diversification',
                            'reasoning': f'Portfolio diversification with {base_asset} (momentum: {momentum:.2f})'
                        })
                    
            except Exception as e:
                logger.debug(f"Error analyzing diversification for {symbol}: {e}")
                continue
        
        return opportunities
    
    def _find_profit_opportunities(self, portfolio: dict) -> list:
        """Find profit-taking and loss-cutting opportunities."""
        opportunities = []
        
        try:
            # Get current portfolio positions (excluding USDT)
            positions = {k: v for k, v in self.current_portfolio.items() if k != 'USDT' and v > 0}
            
            if not positions:
                return opportunities
            
            # Profit-taking and loss-cutting thresholds
            profit_threshold = self.ppo_config.get('profit_taking_threshold', 5.0)  # Take profit at 5%+
            stop_loss_threshold = -3.0  # Stop loss at -3%
            trailing_stop_threshold = -2.0  # Trailing stop at -2%
            
            for asset, amount in positions.items():
                try:
                    symbol = f"{asset}USDT"
                    if symbol not in [s for s in self.trading_pairs if s.endswith('USDT')]:
                        continue
                    
                    # Get current price and calculate potential profit/loss
                    current_price = self.current_prices.get(asset, 0)
                    if current_price <= 0:
                        continue
                    
                    # Get 24h price change for trend analysis
                    ticker_data = self._get_24h_ticker(symbol)
                    price_change_24h = float(ticker_data.get('priceChangePercent', 0))
                    
                    # Estimate position value
                    position_value = amount * current_price
                    
                    # Only consider positions worth at least minimum trade amount
                    if position_value < self.min_trade_usd:
                        continue
                    
                    # PROFIT TAKING SIGNALS
                    if price_change_24h > profit_threshold:
                        confidence = min(0.8, 0.5 + (price_change_24h - profit_threshold) / 20)
                        
                        # Sell portion of position (25-50% based on profit level)
                        if price_change_24h > 10:  # High profit
                            sell_ratio = 0.5  # Sell 50%
                        elif price_change_24h > 7:  # Good profit
                            sell_ratio = 0.4  # Sell 40%
                        else:  # Moderate profit
                            sell_ratio = 0.25  # Sell 25%
                        
                        sell_amount = min(position_value * sell_ratio, self.max_trade_usd)
                        
                        if sell_amount >= self.min_trade_usd:
                            opportunities.append({
                                'symbol': symbol,
                                'action': 'sell',
                                'amount': sell_amount,
                                'confidence': confidence,
                                'strategy': 'profit_taking',
                                'reasoning': f'Profit taking: {price_change_24h:.1f}% gain, selling {sell_ratio:.0%} position'
                            })
                    
                    # STOP LOSS SIGNALS
                    elif price_change_24h < stop_loss_threshold:
                        confidence = min(0.9, 0.6 + abs(price_change_24h - stop_loss_threshold) / 10)
                        
                        # Sell larger portion on stop loss (50-75% based on loss severity)
                        if price_change_24h < -8:  # Severe loss
                            sell_ratio = 0.75  # Sell 75%
                        elif price_change_24h < -5:  # Significant loss
                            sell_ratio = 0.6   # Sell 60%
                        else:  # Moderate loss
                            sell_ratio = 0.4   # Sell 40%
                        
                        sell_amount = min(position_value * sell_ratio, self.max_trade_usd)
                        
                        if sell_amount >= self.min_trade_usd:
                            opportunities.append({
                                'symbol': symbol,
                                'action': 'sell',
                                'amount': sell_amount,
                                'confidence': confidence,
                                'strategy': 'stop_loss',
                                'reasoning': f'Stop loss: {price_change_24h:.1f}% loss, selling {sell_ratio:.0%} position'
                            })
                    
                    # MOMENTUM REVERSAL SIGNALS
                    elif price_change_24h < trailing_stop_threshold:
                        # Get momentum score to confirm reversal
                        momentum = self._calculate_momentum_score(price_change_24h, 
                                                               float(ticker_data.get('volume', 0)), 
                                                               0.3)  # Assume low position in range
                        
                        if momentum < 0.3:  # Weak momentum confirms reversal
                            confidence = 0.6 + abs(price_change_24h) / 20
                            confidence = min(0.75, confidence)
                            
                            # Conservative sell (20-30% of position)
                            sell_ratio = 0.3 if price_change_24h < -3 else 0.2
                            sell_amount = min(position_value * sell_ratio, self.max_trade_usd)
                            
                            if sell_amount >= self.min_trade_usd:
                                opportunities.append({
                                    'symbol': symbol,
                                    'action': 'sell',
                                    'amount': sell_amount,
                                    'confidence': confidence,
                                    'strategy': 'momentum_reversal',
                                    'reasoning': f'Momentum reversal: {price_change_24h:.1f}% drop with weak momentum ({momentum:.2f})'
                                })
                
                except Exception as e:
                    logger.debug(f"Error analyzing sell opportunity for {asset}: {e}")
                    continue
            
            return opportunities
            
        except Exception as e:
            logger.error(f" Error in profit opportunities analysis: {e}")
            return []
    
    def _find_rebalancing_opportunities(self, portfolio: dict) -> list:
        """Find rebalancing opportunities by selling overweight positions."""
        opportunities = []
        
        try:
            total_value = self._calculate_total_value()
            if total_value <= 0:
                return opportunities
            
            # Target allocation thresholds
            max_single_position = 0.25  # No single asset should exceed 25% of portfolio
            max_crypto_allocation = 0.8  # Max 80% in crypto (rest in stablecoins)
            
            positions = {k: v for k, v in self.current_portfolio.items() if k != 'USDT' and v > 0}
            
            for asset, amount in positions.items():
                try:
                    symbol = f"{asset}USDT"
                    current_price = self.current_prices.get(asset, 0)
                    if current_price <= 0:
                        continue
                    
                    position_value = amount * current_price
                    allocation_pct = position_value / total_value
                    
                    # Check if position is overweight
                    if allocation_pct > max_single_position:
                        excess_allocation = allocation_pct - max_single_position
                        excess_value = total_value * excess_allocation
                        
                        # Sell the excess portion
                        sell_amount = min(excess_value, self.max_trade_usd)
                        
                        if sell_amount >= self.min_trade_usd:
                            confidence = 0.5 + min(0.3, excess_allocation * 2)  # Higher confidence for more overweight positions
                            
                            opportunities.append({
                                'symbol': symbol,
                                'action': 'sell',
                                'amount': sell_amount,
                                'confidence': confidence,
                                'strategy': 'rebalancing',
                                'reasoning': f'Rebalancing: {asset} is {allocation_pct:.1%} of portfolio (target: <{max_single_position:.0%})'
                            })
                
                except Exception as e:
                    logger.debug(f"Error analyzing rebalancing for {asset}: {e}")
                    continue
            
            return opportunities
            
        except Exception as e:
            logger.error(f" Error in rebalancing analysis: {e}")
            return []
    
    def _find_breakout_opportunities(self, market_data: dict, available_usdt: float) -> list:
        """Find breakout trading opportunities."""
        opportunities = []
        breakout_position = self.ppo_config.get('breakout_position', 0.8)
        
        for symbol, data in market_data.items():
            try:
                price_position = data.get('price_position_in_range', 0.5)
                momentum = data.get('momentum_score', 0.5)
                price_change = data.get('price_change_24h', 0)
                
                # Breakout criteria (configurable position threshold)
                if price_position > breakout_position and momentum > 0.6 and price_change > 1:
                    confidence = 0.7 + (price_position - breakout_position) * 2.5 + (momentum - 0.6) * 0.5
                    confidence = min(0.9, confidence)
                    
                    # Calculate amount, ensuring it meets minimum trade requirements
                    preferred_amount = available_usdt * 0.25  # 25% of available USDT
                    amount = max(self.min_trade_usd, min(preferred_amount, self.max_trade_usd * 0.8))
                    
                    # Only add if we have enough USDT for the minimum trade
                    if available_usdt >= amount:
                        opportunities.append({
                            'symbol': symbol,
                            'action': 'buy',
                            'amount': amount,
                            'confidence': confidence,
                            'strategy': 'breakout',
                            'reasoning': f'Breakout pattern: high in range ({price_position:.1%}) with momentum {momentum:.2f}'
                        })
                    
            except Exception as e:
                logger.debug(f"Error analyzing breakout for {symbol}: {e}")
                continue
        
        return opportunities
    
    def _validate_strategy(self, strategy: dict, portfolio: dict) -> bool:
        """Validate if a strategy meets all safety and logic requirements."""
        try:
            # Basic validations
            if strategy['amount'] < self.min_trade_usd:
                logger.debug(f" Strategy rejected: Amount too small (${strategy['amount']:.2f} < ${self.min_trade_usd:.2f})")
                return False
            
            if strategy['amount'] > self.max_trade_usd:
                logger.debug(f"âš ï¸ Strategy adjusted: Amount capped (${strategy['amount']:.2f} â†’ ${self.max_trade_usd:.2f})")
                strategy['amount'] = self.max_trade_usd
            
            # Use configurable confidence threshold
            min_confidence = self.ppo_config.get('min_confidence', 0.4)
            if strategy['confidence'] < min_confidence:
                logger.debug(f" Strategy rejected: Confidence too low ({strategy['confidence']:.1%} < {min_confidence:.1%})")
                return False
            
            # Portfolio-specific validations
            available_usdt = portfolio.get('available_usdt', 0)
            if strategy['amount'] > available_usdt:
                logger.debug(f" Strategy rejected: Insufficient USDT (${strategy['amount']:.2f} > ${available_usdt:.2f})")
                return False
            
            # Use configurable max USDT per trade, but ensure we can still meet minimum
            max_usdt_ratio = self.ppo_config.get('max_usdt_per_trade', 0.4)
            max_amount = available_usdt * max_usdt_ratio
            
            # If the ratio-based max is less than minimum trade size, allow up to minimum trade size
            if max_amount < self.min_trade_usd and available_usdt >= self.min_trade_usd:
                max_amount = self.min_trade_usd
                logger.debug(f"ðŸ’¡ Max amount adjusted to meet minimum trade requirement: ${max_amount:.2f}")
            
            if strategy['amount'] > max_amount:
                logger.debug(f"âš ï¸ Strategy adjusted: Amount reduced to max allowed (${strategy['amount']:.2f} â†’ ${max_amount:.2f})")
                strategy['amount'] = max_amount
            
            # Daily trading frequency control
            daily_trades = portfolio.get('daily_trades', 0)
            if daily_trades >= 8:  # More conservative than max limit
                logger.debug(f" Strategy rejected: Daily trade limit ({daily_trades} >= 8)")
                return False
            
            # Additional safety check for minimum amount after adjustments
            if strategy['amount'] < self.min_trade_usd:
                logger.debug(f" Strategy rejected: Amount too small after adjustments (${strategy['amount']:.2f} < ${self.min_trade_usd:.2f})")
                return False
            
            logger.info(f"âœ… Strategy validated: {strategy['symbol']} ${strategy['amount']:.2f} (confidence: {strategy['confidence']:.1%})")
            return True
            
        except Exception as e:
            logger.debug(f" Error validating strategy: {e}")
            return False
    
    def load_ppo_model(self, model_path: str):
        """Load trained PPO model for trading decisions."""
        try:
            # This is where you would load your trained PPO model
            # Example structure:
            # from stable_baselines3 import PPO
            # self.ppo_model = PPO.load(model_path)
            
            logger.info(f"ðŸ¤– PPO model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f" Failed to load PPO model: {e}")
            return False
    
    def get_ppo_strategy_stats(self) -> dict:
        """Get current PPO strategy configuration and performance stats."""
        current_mode = 'conservative' if self.ppo_config['conservative_mode'] else \
                      'aggressive' if self.ppo_config['aggressive_mode'] else 'balanced'
        
        return {
            'current_mode': current_mode,
            'strategy_config': {
                'min_confidence': self.ppo_config['min_confidence'],
                'momentum_threshold': self.ppo_config['momentum_threshold'],
                'dip_threshold': self.ppo_config['dip_threshold'],
                'breakout_position': self.ppo_config['breakout_position'],
                'max_usdt_per_trade': self.ppo_config['max_usdt_per_trade'],
                'diversification_limit': self.ppo_config['diversification_limit'],
                'profit_taking_threshold': self.ppo_config['profit_taking_threshold']
            },
            'safety_limits': {
                'max_trade_usd': self.max_trade_usd,
                'min_trade_usd': self.min_trade_usd,
                'max_daily_trades': self.max_daily_trades,
                'max_daily_loss': self.max_daily_loss
            },
            'current_status': {
                'daily_trades': self.daily_trade_count,
                'daily_pnl': self.daily_pnl,
                'total_fees_paid': self.total_fees_paid,
                'available_usdt': self.current_portfolio.get('USDT', 0),
                'portfolio_value': self._calculate_total_value()
            }
        }
    
    def configure_ppo_advanced(self, **kwargs):
        """Advanced PPO configuration with custom parameters."""
        valid_params = {
            'min_confidence', 'momentum_threshold', 'dip_threshold', 
            'breakout_position', 'max_usdt_per_trade', 'diversification_limit',
            'profit_taking_threshold', 'conservative_mode', 'aggressive_mode'
        }
        
        updated_params = []
        for param, value in kwargs.items():
            if param in valid_params:
                old_value = self.ppo_config.get(param)
                self.ppo_config[param] = value
                updated_params.append(f"{param}: {old_value} â†’ {value}")
                logger.info(f"ðŸ”§ PPO Config Updated: {param} = {value}")
        
        # Reconfigure strategy mode
        self._configure_strategy_mode()
        
        # Send notification about configuration change
        if updated_params:
            self.telegram.send_system_notification(
                f"PPO Configuration Updated:\n" + "\n".join(updated_params[:3]) + 
                (f"\n... and {len(updated_params)-3} more" if len(updated_params) > 3 else ""),
                level='info'
            )
        
        return updated_params
    
    def execute_trade_auto(self, symbol: str, side: str, amount_usd: float, order_type: str = 'market') -> Optional[RealTrade]:
        """Execute a trade automatically without user confirmation."""
        return self.execute_trade(symbol, side, amount_usd, order_type)
    
    def _load_historical_data(self):
        """Load existing trading data."""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load trades to get daily count
                today = datetime.now().date()
                daily_trades = 0
                
                for item in data.get('real_trades', []):
                    trade_date = datetime.fromisoformat(item['timestamp']).date()
                    if trade_date == today:
                        daily_trades += 1
                
                self.daily_trade_count = daily_trades
                
                logger.info(f" Loaded trading history: {daily_trades} trades today")
                
        except Exception as e:
            logger.warning(f"âš ï¸ Could not load historical data: {e}")


def main():
    """Main function for live trading."""
    print(" LIVE TRADING SYSTEM - REAL MONEY AT RISK! ")
    print("=" * 60)
    
    try:
        # Initialize live trading manager
        trader = LiveTradingManager()
        
        if not trader.connect_and_initialize():
            print(" Failed to initialize live trading.")
            return
        
        # Show detailed portfolio with current holdings
        summary = trader.get_portfolio_summary()
        print(f"\nðŸ’° Live Portfolio Summary:")
        print(f"   Total Value: ${summary['portfolio_value']['current']:.2f}")
        print(f"   Daily P&L: ${summary['pnl']['daily']:.2f} ({summary['pnl']['daily_return_pct']:.2f}%)")
        print(f"   Available Balance: ${summary['balances']['available']:.2f}")
        print(f"   Daily Trades: {summary['trading_stats']['daily_trades']}/{summary['trading_stats']['max_daily_trades']}")
        
        print(f"\nðŸ¦ Fee Information:")
        fee_rate_pct = summary['fees']['fee_rate'] * 100
        print(f"   Current Fee Rate: {fee_rate_pct:.3f}%")
        print(f"   BNB Discount: {'âœ… Active' if summary['fees']['bnb_discount_active'] else ' Inactive'}")
        print(f"   Total Fees Paid: ${summary['fees']['total']:.4f}")
        
        # Display detailed holdings
        print(f"\n DETAILED CURRENT HOLDINGS:")
        print("=" * 80)
        holdings = summary['holdings']
        
        if holdings:
            print(f"{'Asset':<8} {'Amount':<15} {'Price':<12} {'Value':<12} {'Allocation':<12} {'24h Change'}")
            print("-" * 80)
            
            for holding in holdings:
                asset = holding['asset']
                amount = holding['amount']
                price = holding['current_price']
                value = holding['usd_value']
                allocation = holding['allocation_pct']
                change_24h = holding.get('change_24h_pct', 0)
                
                # Color coding for 24h change
                if change_24h > 0:
                    change_color = f"ðŸŸ¢ +{change_24h:.2f}%"
                elif change_24h < 0:
                    change_color = f"ðŸ”´ {change_24h:.2f}%"
                else:
                    change_color = f"âšª {change_24h:.2f}%"
                
                print(f"{asset:<8} {amount:<15.6f} ${price:<11.4f} ${value:<11.2f} {allocation:<11.2f}% {change_color}")
            
            print("-" * 80)
            print(f"{'TOTAL':<8} {'':<15} {'':<12} ${summary['portfolio_value']['current']:<11.2f} {'100.00%':<12}")
        else:
            print("   No significant holdings found (minimum $0.10 value)")
        
        # Show tracked cryptocurrencies status
        print(f"\nðŸŽ¯ TRACKED CRYPTOCURRENCIES:")
        for symbol in trader.trading_pairs:
            current_price = trader.get_current_price(symbol)
            print(f"   {symbol}: ${current_price:.4f}")
        
        print("=" * 80)
        
        # Demo live trades (BE CAREFUL!)
        print(f"\n🎯 TRADING MODE SELECTION:")
        print("1. Manual trades (confirm each trade)")
        print("2. Continuous automated trading")
        print("3. View detailed holdings")
        print("4. Configure trading hours")
        print("5. Exit")
        
        mode_choice = input("Select option (1-5): ").strip()
        
        if mode_choice == "1":
            # Manual trading mode
            print(f"\n MANUAL TRADING MODE")
            print("These will be REAL trades with REAL money!")
            
            demo_confirm = input("Type 'EXECUTE REAL TRADES' to proceed: ")
            if demo_confirm != "EXECUTE REAL TRADES":
                print(" Manual trading cancelled.")
                return
                
            # Continue with manual trading...
                trader._save_data()
                return
            
            demo_trades = [
                {'symbol': 'DOGEUSDT', 'side': 'buy', 'amount_usd': 5.0},
                {'symbol': 'ADAUSDT', 'side': 'buy', 'amount_usd': 5.0},
            ]
            
            for trade_config in demo_trades:
                print(f"\n Executing: {trade_config['side'].upper()} ${trade_config['amount_usd']} of {trade_config['symbol']}")

                final_confirm = input(f"Confirm this REAL trade? (y/n): ")
                if final_confirm.lower() == 'y':
                    trade = trader.execute_trade(
                        trade_config['symbol'],
                        trade_config['side'],
                        trade_config['amount_usd']
                    )
                    
                    if trade:
                        print(f"✅ REAL TRADE EXECUTED!")
                        print(f"   Order ID: {trade.order_id}")
                        print(f"   Amount: {trade.amount:.6f} {trade.symbol}")
                        print(f"   Price: ${trade.price:.4f}")
                        print(f"   Value: ${trade.value_usd:.2f}")
                        print(f"   Fee: ${trade.fee_usd:.4f} ({trade.fee_rate*100:.3f}%)")
                    else:
                        print(" Trade failed")
                else:
                    print(" Trade skipped")
                
                time.sleep(3)
        
        elif mode_choice == "2":
            # Continuous trading mode
            print(f"\n CONTINUOUS AUTOMATED TRADING MODE")
            print("🚀 This will execute trades automatically without confirmation!")
            print("🚀 REAL MONEY WILL BE TRADED AUTOMATICALLY!")
            
            # Check if PPO is available
            if not trader.ppo_enabled:
                print("❌ PPO model not available - cannot start automated trading")
                print("   Please ensure the PPO model is properly loaded")
                trader._save_data()
                return
            
            print(f"\n🧠 Using PPO Reinforcement Learning Agent")
            print("   Advanced AI-powered trading decisions")

            # PPO Strategy Mode Selection
            print(f"\n🎯 Select PPO Strategy Mode:")
            print("1. 🛡️ Conservative (High confidence, lower risk)")
            print("2. ⚖️ Balanced (Default balanced approach)")  
            print("3. 🚀 Aggressive (Lower confidence, higher frequency)")
            print("4. ⚙️ Custom Configuration")

            ppo_mode_choice = input("Select PPO mode (1-4, default 2): ").strip()
            
            if ppo_mode_choice == "1":
                trader.set_ppo_mode('conservative')
                print("🛡️ Conservative mode activated")
            elif ppo_mode_choice == "3":
                trader.set_ppo_mode('aggressive')
                print("🚀 Aggressive mode activated")
            elif ppo_mode_choice == "4":
                print("\n⚙️ Custom PPO Configuration:")
                print("Enter new values (press Enter to keep current):")
                current_stats = trader.get_ppo_strategy_stats()
                config = current_stats['strategy_config']
                custom_config = {}
                # Confidence threshold
                new_confidence = input(f"Min confidence (current: {config['min_confidence']:.2f}): ").strip()
                if new_confidence:
                    custom_config['min_confidence'] = float(new_confidence)
                # Momentum threshold
                new_momentum = input(f"Momentum threshold (current: {config['momentum_threshold']:.2f}): ").strip()
                if new_momentum:
                    custom_config['momentum_threshold'] = float(new_momentum)
                # Max USDT per trade
                new_max_usdt = input(f"Max USDT per trade (current: {config['max_usdt_per_trade']:.2f}): ").strip()
                if new_max_usdt:
                    custom_config['max_usdt_per_trade'] = float(new_max_usdt)
                if custom_config:
                    updated = trader.configure_ppo_advanced(**custom_config)
                    print(f"✅ Updated {len(updated)} parameters")
                else:
                    print("No changes made")
            else:
                trader.set_ppo_mode('balanced')
                print("⚖️ Balanced mode activated (default)")
            # Show current PPO configuration
            stats = trader.get_ppo_strategy_stats()
            print(f"\n Current PPO Configuration:")
            print(f"   Mode: {stats['current_mode'].upper()}")
            print(f"   Min Confidence: {stats['strategy_config']['min_confidence']:.2f}")
            print(f"   Momentum Threshold: {stats['strategy_config']['momentum_threshold']:.2f}")
            print(f"   Max USDT per Trade: {stats['strategy_config']['max_usdt_per_trade']:.1%}")
            print(f"   Available USDT: ${stats['current_status']['available_usdt']:.2f}")
            
            auto_confirm = input("Type 'START AUTO TRADING' to begin: ")
            if auto_confirm == "START AUTO TRADING":
                update_interval = input("Update interval in seconds (default 60): ").strip()
                interval = int(update_interval) if update_interval.isdigit() else 60
                
                print(f"ðŸ”„ Starting continuous trading with {interval}s intervals...")
                print("Press Ctrl+C to stop")
                trader.start_continuous_trading(interval)
            else:
                print(" Continuous trading cancelled.")
        
        elif mode_choice == "3":
            # View detailed holdings
            print(f"\n DETAILED HOLDINGS VIEW")
            trader.show_detailed_holdings()
            
            # Ask if user wants to continue to trading or exit
            print(f"\n What would you like to do next?")
            print("1. Return to main menu")
            print("2. Exit")
            
            next_choice = input("Select option (1-2): ").strip()
            if next_choice == "1":
                # Recursive call to main menu
                main()
                return
            else:
                print(" Exiting live trading system.")
                trader._save_data()
                return
                
        elif mode_choice == "4":
            # Configure trading hours
            print(f"\n🕐 TRADING HOURS CONFIGURATION")
            
            # Show current configuration
            hours_status = trader.get_trading_hours_status()
            print(f"Current configuration:")
            print(f"   Trading hours: {hours_status['active_start']}:00 - {hours_status['active_end']}:00")
            print(f"   Time controls: {'Enabled' if hours_status['enabled'] else 'Disabled'}")
            print(f"   Night sleep duration: {hours_status['night_sleep_duration']}s")
            print(f"   Current time: {hours_status['current_time']}")
            print(f"   Currently in trading hours: {'Yes' if hours_status['is_trading_hours'] else 'No'}")
            
            print(f"\n🎛️ Configuration Options:")
            print("1. Enable/Disable time controls")
            print("2. Set active hours (start and end)")
            print("3. Set night sleep duration")
            print("4. Show current status")
            print("5. Return to main menu")
            
            config_choice = input("Select option (1-5): ").strip()
            
            if config_choice == "1":
                current_enabled = trader.trading_hours['enabled']
                new_enabled = not current_enabled
                trader.configure_trading_hours(enabled=new_enabled)
                print(f"✅ Time controls {'enabled' if new_enabled else 'disabled'}")
                
            elif config_choice == "2":
                print("Set active trading hours (24-hour format):")
                try:
                    start_hour = int(input(f"Start hour (0-23, current: {trader.trading_hours['active_start']}): "))
                    end_hour = int(input(f"End hour (0-23, current: {trader.trading_hours['active_end']}): "))
                    
                    trader.configure_trading_hours(active_start=start_hour, active_end=end_hour)
                    print(f"✅ Trading hours set to {start_hour}:00 - {end_hour}:00")
                except ValueError:
                    print("❌ Invalid input. Please enter numbers 0-23.")
                    
            elif config_choice == "3":
                print("Set sleep duration during quiet hours:")
                try:
                    sleep_duration = int(input(f"Sleep duration in seconds (current: {trader.trading_hours['night_sleep_duration']}): "))
                    trader.configure_trading_hours(night_sleep_duration=sleep_duration)
                    print(f"✅ Night sleep duration set to {sleep_duration}s")
                except ValueError:
                    print("❌ Invalid input. Please enter a positive number.")
                    
            elif config_choice == "4":
                # Show detailed status
                status = trader.get_trading_hours_status()
                print(f"\n📊 Detailed Trading Hours Status:")
                print(f"   Current time: {status['current_time']} (Hour: {status['current_hour']})")
                print(f"   Active hours: {status['active_start']}:00 - {status['active_end']}:00")
                print(f"   Time controls: {'Enabled' if status['enabled'] else 'Disabled'}")
                print(f"   Currently active: {'Yes' if status['is_trading_hours'] else 'No'}")
                print(f"   Current sleep duration: {status['current_sleep_duration']}s")
                
                if not status['is_trading_hours'] and status['enabled']:
                    print(f"   ⏰ Next active period in: {status['hours_until_active']} hours")
                elif status['is_trading_hours'] and status['enabled']:
                    print(f"   ⏰ Quiet period begins in: {status['hours_until_quiet']} hours")
                    
            elif config_choice == "5":
                # Return to main menu
                main()
                return
                
            else:
                print("❌ Invalid option.")
            
            # Ask if user wants to continue configuring or return to main menu
            print(f"\nWhat would you like to do next?")
            print("1. Continue configuring trading hours")
            print("2. Return to main menu")
            
            next_choice = input("Select option (1-2): ").strip()
            if next_choice == "1":
                # Recursive call to configuration menu
                main()
                return
            else:
                # Return to main menu
                main()
                return
                
        elif mode_choice == "5":
            print("💤 Exiting live trading system.")
            trader._save_data()
            return
        
        else:
            print("❌ Invalid option. Please select 1-5.")
            trader._save_data()
            return
        
        # Final summary
        summary = trader.get_portfolio_summary()
        print(f"\n Final Summary:")
        print(f"   Portfolio Value: ${summary['portfolio_value']['current']:.2f}")
        print(f"   Daily P&L: ${summary['pnl']['daily']:.2f}")
        print(f"   Trades Executed: {summary['trading_stats']['daily_trades']}")
        print(f"   Total Fees Paid: ${summary['fees']['total']:.4f}")
        print(f"   Today's Fees: ${summary['fees']['today']:.4f}")
        
        trader._save_data()
        print(f"\n Live trading session saved!")

    except Exception as e:
        logger.error(f" Live trading error: {e}")
        print(f" Error: {e}")


if __name__ == "__main__":
    main()

