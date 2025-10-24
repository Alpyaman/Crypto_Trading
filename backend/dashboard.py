"""
AI Trading Dashboard - Real-time monitoring
"""
import requests
import time
from datetime import datetime

class TradingDashboard:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_trading_status(self):
        """Get current trading status"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/trading/status")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def get_current_price(self, symbol="BTCUSDT"):
        """Get current market price"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/market/price/{symbol}")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def get_trading_history(self):
        """Get trading history"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/trading/history")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def get_account_balance(self):
        """Get account balance"""
        try:
            response = self.session.get(f"{self.base_url}/api/v1/account/balance")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def stop_trading(self):
        """Stop trading"""
        try:
            response = self.session.post(f"{self.base_url}/api/v1/trading/stop")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None
    
    def display_dashboard(self):
        """Display comprehensive trading dashboard"""
        print("🤖 AI CRYPTO TRADING DASHBOARD")
        print("=" * 50)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Trading Status
        print("\n🚀 TRADING STATUS")
        print("-" * 30)
        status = self.get_trading_status()
        if status:
            is_trading = status.get('is_trading', False)
            total_trades = status.get('total_trades', 0)
            current_position = status.get('current_position')
            
            print(f"Status: {'🟢 ACTIVE' if is_trading else '🔴 STOPPED'}")
            print(f"Total Trades: {total_trades}")
            
            if current_position:
                print("Current Position:")
                print(f"  Symbol: {current_position.get('symbol')}")
                print(f"  Side: {current_position.get('side')}")
                print(f"  Quantity: {current_position.get('quantity')}")
                print(f"  Entry Price: ${current_position.get('entry_price', 0):,.2f}")
            else:
                print("Current Position: None")
        else:
            print("❌ Unable to get trading status")
        
        # Market Data
        print("\n📊 MARKET DATA")
        print("-" * 30)
        price_data = self.get_current_price()
        if price_data:
            price = price_data.get('price', 0)
            print(f"BTC/USDT Price: ${price:,.2f}")
        else:
            print("❌ Unable to get current price")
        
        # Account Balance
        print("\n💰 ACCOUNT BALANCE")
        print("-" * 30)
        balance = self.get_account_balance()
        if balance:
            if balance:
                print("Top balances:")
                for asset, info in list(balance.items())[:5]:
                    if info.get('total', 0) > 0:
                        print(f"  {asset}: {info['total']:.6f}")
            else:
                print("No significant balances")
        else:
            print("❌ Unable to get balance (normal for some testnet accounts)")
        
        # Recent Trading History
        print("\n📋 RECENT TRADES")
        print("-" * 30)
        history = self.get_trading_history()
        if history and history.get('history'):
            trades = history['history'][-5:]  # Last 5 trades
            for trade in trades:
                action = trade.get('action', 'UNKNOWN')
                symbol = trade.get('symbol', 'UNKNOWN')
                quantity = trade.get('quantity', 0)
                price = trade.get('price', 0)
                
                print(f"  {action} {quantity:.6f} {symbol} @ ${price:,.2f}")
        else:
            print("No trades yet (AI is analyzing market conditions)")
        
        print("\n" + "=" * 50)
    
    def monitor_live(self, interval=30):
        """Live monitoring with auto-refresh"""
        print("🔴 LIVE MONITORING MODE")
        print("Press Ctrl+C to stop monitoring")
        print("=" * 50)
        
        try:
            while True:
                self.display_dashboard()
                print(f"\n⏰ Refreshing in {interval} seconds...")
                print("🛑 Press Ctrl+C to stop monitoring")
                time.sleep(interval)
                
                # Clear screen (works on most terminals)
                print("\033[2J\033[H", end="")
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped by user")

def main():
    dashboard = TradingDashboard()
    
    print("🤖 AI Crypto Trading Dashboard")
    print("Choose an option:")
    print("1. Show current status")
    print("2. Start live monitoring") 
    print("3. Stop trading")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                dashboard.display_dashboard()
            
            elif choice == "2":
                dashboard.monitor_live()
                break
            
            elif choice == "3":
                print("🛑 Stopping trading...")
                result = dashboard.stop_trading()
                if result:
                    print("✅ Trading stopped successfully")
                else:
                    print("❌ Failed to stop trading")
            
            elif choice == "4":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()