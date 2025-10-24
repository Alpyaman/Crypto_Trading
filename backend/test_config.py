"""
Test configuration loading
"""
import os
import sys
from dotenv import load_dotenv

# Load .env explicitly
load_dotenv()

# Add the app directory to the path
sys.path.insert(0, os.path.dirname(__file__))

print("🔍 Testing Configuration Loading")
print("=" * 40)

# Test 1: Direct environment variable access
print("📋 Direct environment variables:")
api_key_direct = os.getenv("BINANCE_API_KEY")
api_secret_direct = os.getenv("BINANCE_API_SECRET")
testnet_direct = os.getenv("BINANCE_TESTNET")

print(f"  BINANCE_API_KEY: {bool(api_key_direct)} (length: {len(api_key_direct) if api_key_direct else 0})")
print(f"  BINANCE_API_SECRET: {bool(api_secret_direct)} (length: {len(api_secret_direct) if api_secret_direct else 0})")
print(f"  BINANCE_TESTNET: {testnet_direct}")

if api_key_direct:
    print(f"  API Key preview: {api_key_direct[:10]}...{api_key_direct[-10:]}")

# Test 2: Configuration module
print("\n⚙️ Configuration module:")
try:
    from app.config import config
    
    print(f"  Config API Key: {bool(config.binance_api_key)} (length: {len(config.binance_api_key) if config.binance_api_key else 0})")
    print(f"  Config API Secret: {bool(config.binance_api_secret)} (length: {len(config.binance_api_secret) if config.binance_api_secret else 0})")
    print(f"  Config Testnet: {config.binance_testnet}")
    print(f"  Config Environment: {config.environment}")
    print(f"  Config Debug: {config.debug}")
    
    if config.binance_api_key:
        print(f"  Config API Key preview: {config.binance_api_key[:10]}...{config.binance_api_key[-10:]}")
    
except Exception as e:
    print(f"  ❌ Error loading config: {e}")

# Test 3: Check .env file exists
print("\n📁 File checks:")
env_path = ".env"
if os.path.exists(env_path):
    print(f"  ✅ .env file exists at: {os.path.abspath(env_path)}")
    with open(env_path, 'r') as f:
        content = f.read()
        has_api_key = "BINANCE_API_KEY=" in content
        has_api_secret = "BINANCE_API_SECRET=" in content
        print(f"  Contains BINANCE_API_KEY: {has_api_key}")
        print(f"  Contains BINANCE_API_SECRET: {has_api_secret}")
else:
    print(f"  ❌ .env file not found at: {os.path.abspath(env_path)}")

print("\n" + "=" * 40)
if api_key_direct and api_secret_direct:
    print("✅ Configuration looks good!")
else:
    print("❌ Configuration has issues")
    print("💡 Try restarting your terminal and running the server again")