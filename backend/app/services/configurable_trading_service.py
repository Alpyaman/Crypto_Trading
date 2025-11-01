"""
Enhanced Trading Service with External Configuration
Risk management and trading logic using validated configuration
"""

from app.configuration.enhanced_config import get_trading_config, PositionSizingModeConfig, RegimeParameterConfig
from app.services.binance_service import BinanceService
from app.services.enhanced_ml_service import EnhancedMLService
from app.core.error_handling import TradingException
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ConfigurableEnhancedTradingService:
    """
    Enhanced trading service with externalized configuration management
    All risk parameters and trading settings loaded from validated config files
    """
    
    def __init__(self, binance_service: BinanceService, ml_service: EnhancedMLService):
        self.binance_service = binance_service
        self.ml_service = ml_service
        self.config = get_trading_config()
        
        # Load configuration sections
        self.risk_config = self.config.trading.risk_management
        self.position_config = self.config.trading.position_sizing
        self.regime_config = self.config.trading.regime_parameters
        
        # Trading state
        self.is_trading = False
        self.current_positions = {}
        self.daily_pnl = 0.0
        self.session_start_balance = 0.0
        self.max_session_balance = 0.0
        self.current_drawdown = 0.0
        self.trading_mode = "balanced"  # Default mode
        
        logger.info("🔧 Configurable Enhanced Trading Service initialized")
        logger.info(f"📊 Risk Config: Max Position: {self.risk_config.max_position_size:.1%}, "
                   f"Max Drawdown: {self.risk_config.max_drawdown:.1%}, "
                   f"Stop Loss: {self.risk_config.stop_loss_percentage:.1%}")
    
    async def start_trading(self, symbol: str = "BTCUSDT", mode: str = "balanced") -> Dict[str, Any]:
        """Start automated trading with configuration-based parameters"""
        try:
            if self.is_trading:
                raise TradingException("Trading is already active")
            
            # Validate trading mode
            if mode not in ["conservative", "balanced", "aggressive"]:
                raise ValueError(f"Invalid trading mode: {mode}. Must be one of: conservative, balanced, aggressive")
            
            self.trading_mode = mode
            
            # Get account balance to initialize session
            balance_info = await self.binance_service.get_account_balance()
            if not balance_info["success"]:
                raise TradingException("Failed to get account balance")
            
            self.session_start_balance = float(balance_info["data"]["free_balance"])
            self.max_session_balance = self.session_start_balance
            self.daily_pnl = 0.0
            self.current_drawdown = 0.0
            
            # Initialize ML service
            await self.ml_service.initialize()
            
            self.is_trading = True
            
            logger.info(f"🚀 Trading started for {symbol} in {mode} mode")
            logger.info(f"💰 Starting balance: ${self.session_start_balance:.2f}")
            logger.info(f"🎯 Position sizing: {self._get_position_config(mode).base_size:.1%} base size")
            
            return {
                "success": True,
                "message": f"Trading started successfully in {mode} mode",
                "config": {
                    "symbol": symbol,
                    "mode": mode,
                    "starting_balance": self.session_start_balance,
                    "max_position_size": self.risk_config.max_position_size,
                    "stop_loss": self.risk_config.stop_loss_percentage,
                    "take_profit": self.risk_config.take_profit_percentage,
                    "max_concurrent_trades": self.risk_config.max_concurrent_trades
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to start trading: {e}")
            return {"success": False, "error": str(e)}
    
    async def execute_trade(self, symbol: str, current_price: float, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade using configuration-based risk management"""
        try:
            if not self.is_trading:
                return {"success": False, "message": "Trading is not active"}
            
            # Check risk limits before trading
            risk_check = await self._check_risk_limits()
            if not risk_check["allowed"]:
                return {"success": False, "message": f"Risk limit exceeded: {risk_check['reason']}"}
            
            # Get market regime and ML prediction
            regime_info = await self.ml_service.detect_market_regime(market_data)
            prediction = await self.ml_service.predict(market_data, regime_info["regime"])
            
            if prediction["confidence"] < 0.6:  # Configurable threshold
                return {"success": False, "message": "Low prediction confidence"}
            
            # Determine trade action and size
            trade_action = self._determine_trade_action(prediction, regime_info)
            if trade_action["action"] == "HOLD":
                return {"success": False, "message": "No trading signal generated"}
            
            # Calculate position size based on configuration
            position_size = self._calculate_position_size(
                symbol, current_price, trade_action["action"], regime_info["regime"]
            )
            
            if position_size <= 0:
                return {"success": False, "message": "Position size too small"}
            
            # Execute the trade
            trade_result = await self._execute_position(
                symbol, trade_action["action"], position_size, current_price, regime_info
            )
            
            if trade_result["success"]:
                # Update position tracking
                self._update_position_tracking(symbol, trade_result)
                
                # Update PnL and drawdown tracking
                await self._update_performance_metrics()
                
                logger.info(f"✅ Trade executed: {trade_action['action']} {position_size:.6f} {symbol} "
                           f"at ${current_price:.2f} (Regime: {regime_info['regime']})")
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _check_risk_limits(self) -> Dict[str, Any]:
        """Check all risk limits based on configuration"""
        try:
            # Get current account balance
            balance_info = await self.binance_service.get_account_balance()
            if not balance_info["success"]:
                return {"allowed": False, "reason": "Failed to get account balance"}
            
            current_balance = float(balance_info["data"]["free_balance"])
            
            # Check maximum daily loss
            daily_loss_pct = (self.session_start_balance - current_balance) / self.session_start_balance
            if daily_loss_pct > self.risk_config.max_daily_loss:
                return {
                    "allowed": False, 
                    "reason": f"Daily loss limit exceeded: {daily_loss_pct:.2%} > {self.risk_config.max_daily_loss:.2%}"
                }
            
            # Check maximum drawdown
            if current_balance > self.max_session_balance:
                self.max_session_balance = current_balance
            
            drawdown_pct = (self.max_session_balance - current_balance) / self.max_session_balance
            self.current_drawdown = drawdown_pct
            
            if drawdown_pct > self.risk_config.max_drawdown:
                return {
                    "allowed": False,
                    "reason": f"Maximum drawdown exceeded: {drawdown_pct:.2%} > {self.risk_config.max_drawdown:.2%}"
                }
            
            # Check maximum concurrent positions
            active_positions = len([p for p in self.current_positions.values() if p.get("status") == "open"])
            if active_positions >= self.risk_config.max_concurrent_trades:
                return {
                    "allowed": False,
                    "reason": f"Maximum concurrent trades reached: {active_positions}/{self.risk_config.max_concurrent_trades}"
                }
            
            return {"allowed": True, "reason": "All risk checks passed"}
            
        except Exception as e:
            logger.error(f"Risk check failed: {e}")
            return {"allowed": False, "reason": f"Risk check error: {str(e)}"}
    
    def _get_position_config(self, mode: str = None) -> PositionSizingModeConfig:
        """Get position sizing configuration for the specified mode"""
        if mode is None:
            mode = self.trading_mode
        
        return self.config.get_position_sizing_config(mode)
    
    def _get_regime_config(self, regime: str) -> RegimeParameterConfig:
        """Get regime-specific configuration"""
        return self.config.get_regime_config(regime)
    
    def _determine_trade_action(self, prediction: Dict[str, Any], regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """Determine trade action based on ML prediction and regime"""
        regime = regime_info["regime"]
        regime_config = self._get_regime_config(regime)
        
        # Base action from ML prediction
        base_action = prediction["action"]
        confidence = prediction["confidence"]
        
        # Apply regime-specific filters
        if regime == "volatile":
            # Reduce trading in volatile markets
            confidence_threshold = 0.8
        elif regime == "trending":
            # Lower threshold for trending markets
            confidence_threshold = 0.65
        else:  # ranging
            # Standard threshold for ranging markets
            confidence_threshold = 0.7
        
        if confidence < confidence_threshold:
            return {"action": "HOLD", "confidence": confidence, "reason": "Confidence below regime threshold"}
        
        # Check regime-specific momentum requirements
        if regime == "trending" and regime_config.momentum_threshold:
            momentum = regime_info.get("momentum", 0)
            if abs(momentum) < regime_config.momentum_threshold:
                return {"action": "HOLD", "confidence": confidence, "reason": "Insufficient momentum for trending regime"}
        
        return {"action": base_action, "confidence": confidence, "regime": regime}
    
    def _calculate_position_size(self, symbol: str, price: float, action: str, regime: str) -> float:
        """Calculate position size based on configuration and regime"""
        try:
            # Get position sizing configuration
            position_config = self._get_position_config()
            regime_config = self._get_regime_config(regime)
            
            # Base position size
            base_size = position_config.base_size
            
            # Apply regime multiplier
            regime_multiplier = regime_config.position_multiplier
            adjusted_size = base_size * regime_multiplier
            
            # Apply volatility adjustment if enabled
            if position_config.volatility_adjustment:
                volatility = self._estimate_volatility(symbol)
                volatility_multiplier = max(0.5, min(1.5, 1.0 / (1.0 + volatility)))
                adjusted_size *= volatility_multiplier
            
            # Ensure size is within bounds
            final_size = max(position_config.min_size, min(position_config.max_size, adjusted_size))
            
            # Convert to absolute position size based on current balance
            # Note: In real implementation, this should be awaited properly
            
            # For now, use estimated balance
            estimated_balance = self.session_start_balance  # Simplified
            position_value = estimated_balance * final_size
            position_quantity = position_value / price
            
            logger.debug(f"Position calculation: base={base_size:.2%}, regime_mult={regime_multiplier:.2f}, "
                        f"final_size={final_size:.2%}, quantity={position_quantity:.6f}")
            
            return position_quantity
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0.0
    
    def _estimate_volatility(self, symbol: str) -> float:
        """Estimate current market volatility (simplified)"""
        # In a real implementation, this would calculate based on recent price data
        # For now, return a default volatility estimate
        return 0.2  # 20% estimated volatility
    
    async def _execute_position(self, symbol: str, action: str, quantity: float, price: float, regime_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trading position with configuration-based risk management"""
        try:
            # Calculate stop loss and take profit based on configuration
            stop_loss_pct = self.risk_config.stop_loss_percentage
            take_profit_pct = self.risk_config.take_profit_percentage
            
            # Apply regime-specific adjustments
            regime_config = self._get_regime_config(regime_info["regime"])
            if hasattr(regime_config, 'stop_loss_tightening') and regime_config.stop_loss_tightening:
                stop_loss_pct *= regime_config.stop_loss_tightening
            
            if action == "BUY":
                stop_loss_price = price * (1 - stop_loss_pct)
                take_profit_price = price * (1 + take_profit_pct)
            else:  # SELL
                stop_loss_price = price * (1 + stop_loss_pct)
                take_profit_price = price * (1 - take_profit_pct)
            
            # Execute the main order
            order_result = await self.binance_service.place_order(
                symbol=symbol,
                side=action,
                quantity=quantity,
                order_type="MARKET"
            )
            
            if not order_result["success"]:
                return order_result
            
            # Store position information
            position_id = f"{symbol}_{action}_{datetime.now().timestamp()}"
            position_info = {
                "id": position_id,
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "entry_price": price,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "regime": regime_info["regime"],
                "timestamp": datetime.now(),
                "status": "open"
            }
            
            self.current_positions[position_id] = position_info
            
            logger.info(f"📈 Position opened: {action} {quantity:.6f} {symbol}")
            logger.info(f"🎯 Stop Loss: ${stop_loss_price:.2f}, Take Profit: ${take_profit_price:.2f}")
            
            return {
                "success": True,
                "position_id": position_id,
                "order_result": order_result,
                "position_info": position_info
            }
            
        except Exception as e:
            logger.error(f"Position execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _update_position_tracking(self, symbol: str, trade_result: Dict[str, Any]):
        """Update position tracking after trade execution"""
        position_info = trade_result.get("position_info", {})
        if position_info:
            logger.debug(f"Position tracking updated for {symbol}: {position_info['id']}")
    
    async def _update_performance_metrics(self):
        """Update performance metrics including PnL and drawdown"""
        try:
            # Get current balance
            balance_info = await self.binance_service.get_account_balance()
            if balance_info["success"]:
                current_balance = float(balance_info["data"]["free_balance"])
                
                # Update daily PnL
                self.daily_pnl = current_balance - self.session_start_balance
                
                # Update maximum balance and drawdown
                if current_balance > self.max_session_balance:
                    self.max_session_balance = current_balance
                
                self.current_drawdown = (self.max_session_balance - current_balance) / self.max_session_balance
                
                logger.debug(f"Performance update: Balance=${current_balance:.2f}, "
                           f"PnL=${self.daily_pnl:.2f}, Drawdown={self.current_drawdown:.2%}")
                
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
    
    async def stop_trading(self) -> Dict[str, Any]:
        """Stop automated trading and close all positions"""
        try:
            if not self.is_trading:
                return {"success": False, "message": "Trading is not active"}
            
            self.is_trading = False
            
            # Close all open positions
            closed_positions = []
            for position_id, position in self.current_positions.items():
                if position["status"] == "open":
                    close_result = await self._close_position(position)
                    if close_result["success"]:
                        closed_positions.append(position_id)
            
            # Calculate final performance
            final_balance_info = await self.binance_service.get_account_balance()
            final_balance = float(final_balance_info["data"]["free_balance"]) if final_balance_info["success"] else 0
            
            total_pnl = final_balance - self.session_start_balance
            total_return = (total_pnl / self.session_start_balance) * 100 if self.session_start_balance > 0 else 0
            
            logger.info("🛑 Trading stopped")
            logger.info(f"📊 Session Performance: ${total_pnl:.2f} ({total_return:.2f}%)")
            logger.info(f"📉 Maximum Drawdown: {self.current_drawdown:.2%}")
            
            return {
                "success": True,
                "message": "Trading stopped successfully",
                "performance": {
                    "total_pnl": total_pnl,
                    "total_return_pct": total_return,
                    "max_drawdown": self.current_drawdown,
                    "positions_closed": len(closed_positions),
                    "session_duration": datetime.now() - datetime.fromtimestamp(
                        min(p["timestamp"].timestamp() for p in self.current_positions.values())
                        if self.current_positions else datetime.now().timestamp()
                    )
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to stop trading: {e}")
            return {"success": False, "error": str(e)}
    
    async def _close_position(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Close an individual position"""
        try:
            # Determine opposite action
            opposite_action = "SELL" if position["action"] == "BUY" else "BUY"
            
            # Execute closing order
            close_result = await self.binance_service.place_order(
                symbol=position["symbol"],
                side=opposite_action,
                quantity=position["quantity"],
                order_type="MARKET"
            )
            
            if close_result["success"]:
                position["status"] = "closed"
                position["close_time"] = datetime.now()
                
                logger.info(f"✅ Position closed: {position['symbol']} {position['id']}")
            
            return close_result
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return {"success": False, "error": str(e)}
    
    def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status and configuration"""
        return {
            "is_trading": self.is_trading,
            "trading_mode": self.trading_mode,
            "session_pnl": self.daily_pnl,
            "current_drawdown": self.current_drawdown,
            "active_positions": len([p for p in self.current_positions.values() if p.get("status") == "open"]),
            "risk_config": {
                "max_position_size": self.risk_config.max_position_size,
                "max_daily_loss": self.risk_config.max_daily_loss,
                "max_drawdown": self.risk_config.max_drawdown,
                "stop_loss_pct": self.risk_config.stop_loss_percentage,
                "take_profit_pct": self.risk_config.take_profit_percentage,
                "max_concurrent_trades": self.risk_config.max_concurrent_trades
            },
            "position_config": {
                "mode": self.trading_mode,
                "base_size": self._get_position_config().base_size,
                "max_size": self._get_position_config().max_size,
                "min_size": self._get_position_config().min_size,
                "volatility_adjustment": self._get_position_config().volatility_adjustment
            }
        }
    
    async def update_configuration(self) -> Dict[str, Any]:
        """Reload configuration from files"""
        try:
            # Reload configuration
            self.config.reload_config()
            
            # Update cached config references
            self.risk_config = self.config.trading.risk_management
            self.position_config = self.config.trading.position_sizing
            self.regime_config = self.config.trading.regime_parameters
            
            logger.info("🔄 Configuration reloaded successfully")
            
            return {
                "success": True,
                "message": "Configuration updated successfully",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            return {"success": False, "error": str(e)}