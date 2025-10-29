#!/usr/bin/env python3
"""
Environment Startup Script
Easy way to start the application with different environment configurations
"""
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

from app.utils.env_loader import load_environment, get_config_summary, validate_config


def main():
    parser = argparse.ArgumentParser(description="Crypto Trading AI - Environment Startup")
    parser.add_argument(
        "--env", 
        choices=["development", "staging", "production", "test"],
        default="development",
        help="Environment to run (default: development)"
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Override host address"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override port number"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Check configuration and exit"
    )
    
    args = parser.parse_args()
    
    # Set environment
    os.environ['ENVIRONMENT'] = args.env
    
    # Load environment configuration
    print(f"🚀 Starting Crypto Trading AI in {args.env.upper()} mode")
    
    if not load_environment(args.env):
        print("❌ Failed to load environment configuration")
        sys.exit(1)
    
    # Validate configuration
    missing = validate_config()
    if missing:
        print(f"❌ Missing required configuration: {', '.join(missing)}")
        print("Please check your .env files")
        sys.exit(1)
    
    # Show configuration summary
    config = get_config_summary()
    print("\n📊 Configuration Summary:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    if args.check_config:
        print("\n✅ Configuration is valid")
        return
    
    # Environment-specific warnings
    if args.env == "production":
        print("\n⚠️  PRODUCTION MODE - LIVE TRADING ENABLED")
        print("   Make sure you have:")
        print("   - Valid live API keys")
        print("   - Sufficient account balance")
        print("   - Risk management configured")
        confirm = input("\nContinue with production mode? (yes/no): ")
        if confirm.lower() != "yes":
            print("Aborted")
            sys.exit(0)
    
    elif args.env == "development":
        print("\n🔧 Development mode - Safe for testing")
        if os.getenv('BINANCE_TESTNET') != 'true':
            print("⚠️  Warning: BINANCE_TESTNET is not set to 'true'")
    
    # Prepare uvicorn command
    host = args.host or os.getenv('HOST', '127.0.0.1')
    port = args.port or int(os.getenv('PORT', 8000))
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "app.main:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if args.reload or args.env == "development":
        cmd.append("--reload")
    
    if args.env == "production":
        cmd.extend(["--workers", "4"])
    
    print(f"\n🌐 Starting server on http://{host}:{port}")
    print(f"📝 Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")


if __name__ == "__main__":
    main()