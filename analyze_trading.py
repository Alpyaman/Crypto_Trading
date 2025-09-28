#!/usr/bin/env python3
"""
Trading Data Analysis Script
"""

import json
from datetime import datetime
from collections import Counter

def analyze_trading_data():
    # Load and analyze the trading data
    with open('live_trading_20250811.json', 'r') as f:
        data = json.load(f)

    metadata = data['metadata']
    trades = data['real_trades']
    portfolio_history = data['portfolio_history']

    print('📊 TRADING SESSION ANALYSIS')
    print('=' * 50)

    # Basic stats
    print(f'📅 Session Date: {datetime.fromisoformat(metadata["created"]).strftime("%Y-%m-%d %H:%M")}')
    print(f'💰 Initial Value: ${metadata["initial_value"]:.2f}')
    print(f'💰 Final Value: ${metadata["current_value"]:.2f}')
    print(f'📉 Total P&L: ${metadata["daily_pnl"]:.2f} ({(metadata["daily_pnl"]/metadata["initial_value"]*100):.2f}%)')
    print(f'🔄 Total Trades: {len(trades)}')
    print(f'📊 Portfolio Snapshots: {len(portfolio_history)}')

    # Trade analysis
    print(f'\n🎯 TRADE BREAKDOWN')
    print('=' * 30)

    buy_trades = [t for t in trades if t['side'] == 'buy']
    sell_trades = [t for t in trades if t['side'] == 'sell']

    print(f'📈 Buy Orders: {len(buy_trades)}')
    print(f'📉 Sell Orders: {len(sell_trades)}')

    # Symbol frequency
    symbol_counts = Counter([t['symbol'] for t in trades])
    print(f'\n🎯 Most Traded Symbols:')
    for symbol, count in symbol_counts.most_common(5):
        print(f'   {symbol}: {count} trades')

    # Fee analysis
    total_fees = sum(t['fee_usd'] for t in trades)
    avg_trade_size = sum(t['value_usd'] for t in trades) / len(trades)

    print(f'\n💳 FEE ANALYSIS')
    print('=' * 20)
    print(f'Total Fees Paid: ${total_fees:.4f}')
    print(f'Average Trade Size: ${avg_trade_size:.2f}')
    print(f'Fees as % of Portfolio: {(total_fees/metadata["initial_value"]*100):.3f}%')

    # Performance analysis
    print(f'\n📈 SYMBOL PERFORMANCE')
    print('=' * 25)
    
    # Group trades by symbol to calculate P&L
    symbol_positions = {}
    for trade in trades:
        symbol = trade['symbol']
        if symbol not in symbol_positions:
            symbol_positions[symbol] = []
        symbol_positions[symbol].append(trade)

    total_realized_pnl = 0
    for symbol, symbol_trades in symbol_positions.items():
        buy_volume = sum(t['amount'] for t in symbol_trades if t['side'] == 'buy')
        sell_volume = sum(t['amount'] for t in symbol_trades if t['side'] == 'sell')
        buy_value = sum(t['value_usd'] for t in symbol_trades if t['side'] == 'buy')
        sell_value = sum(t['value_usd'] for t in symbol_trades if t['side'] == 'sell')
        
        net_position = buy_volume - sell_volume
        realized_pnl = sell_value - buy_value if sell_volume > 0 else 0
        total_realized_pnl += realized_pnl
        
        print(f'{symbol}:')
        print(f'  Buys: {len([t for t in symbol_trades if t["side"] == "buy"])}')
        print(f'  Sells: {len([t for t in symbol_trades if t["side"] == "sell"])}')
        print(f'  Net Position: {net_position:.6f}')
        print(f'  Realized P&L: ${realized_pnl:.4f}')
        print()

    print(f'💰 TOTAL REALIZED P&L: ${total_realized_pnl:.4f}')
    print(f'💸 TOTAL FEES PAID: ${total_fees:.4f}')
    print(f'📊 NET AFTER FEES: ${total_realized_pnl - total_fees:.4f}')

if __name__ == "__main__":
    analyze_trading_data()
