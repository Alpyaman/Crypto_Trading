#!/usr/bin/env python3
"""
Script to check and update environment variables for live trading
"""
import os
from dotenv import load_dotenv

# Load any .env file if it exists
load_dotenv()

print("Current Binance Environment Variables:")
print(f"BINANCE_TESTNET: {os.getenv('BINANCE_TESTNET', 'Not set')}")
print(f"BINANCE_API_KEY: {os.getenv('BINANCE_API_KEY', 'Not set')}")
print(f"BINANCE_API_SECRET: {os.getenv('BINANCE_API_SECRET', 'Not set')}")
print(f"BINANCE_TESTNET_API: {os.getenv('BINANCE_TESTNET_API', 'Not set')}")
print(f"BINANCE_TESTNET_SECRET: {os.getenv('BINANCE_TESTNET_SECRET', 'Not set')}")
print(f"ENVIRONMENT: {os.getenv('ENVIRONMENT', 'Not set')}")

print("\nTo switch to LIVE trading, you need to:")
print("1. Set BINANCE_TESTNET=false")
print("2. Set your live BINANCE_API_KEY and BINANCE_API_SECRET")
print("3. Restart the application")

print("\n⚠️  WARNING: Live trading uses real money! Ensure you have proper risk management!")