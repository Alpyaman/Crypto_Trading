"""
Credential Migration Tool
Helps migrate from plain text to encrypted credentials
"""
import os
import logging
from pathlib import Path

from app.security.credential_manager import SecureCredentialManager
from app.utils.env_loader import load_credentials

logger = logging.getLogger(__name__)


def migrate_credentials():
    """
    Migrate plain text credentials to encrypted format
    
    Steps:
    1. Load existing plain text credentials
    2. Initialize secure credential manager
    3. Encrypt and store credentials
    4. Verify encrypted credentials work
    5. Backup original .env file
    """
    print("🔐 CREDENTIAL MIGRATION TOOL")
    print("=" * 50)
    print()
    
    # Step 1: Load existing credentials
    print("1. Loading existing credentials...")
    try:
        credentials = load_credentials()
        api_key = credentials.get('api_key') or credentials.get('BINANCE_API_KEY')
        api_secret = credentials.get('api_secret') or credentials.get('BINANCE_API_SECRET')
        testnet = credentials.get('testnet', False)
        
        if not api_key or not api_secret:
            print("❌ No plain text credentials found to migrate")
            return False
        
        print(f"✅ Found credentials (API Key: {api_key[:8]}...)")
        
    except Exception as e:
        print(f"❌ Failed to load credentials: {e}")
        return False
    
    # Step 2: Initialize secure credential manager
    print("\n2. Initializing secure credential manager...")
    try:
        # Prompt for master password if not in environment
        master_password = os.getenv('CRYPTO_MASTER_PASSWORD')
        if not master_password:
            master_password = input("Enter master password for encryption (or press Enter for default): ").strip()
            if not master_password:
                master_password = None
        
        credential_manager = SecureCredentialManager(master_password)
        print("✅ Secure credential manager initialized")
        
    except Exception as e:
        print(f"❌ Failed to initialize credential manager: {e}")
        return False
    
    # Step 3: Encrypt and store credentials
    print("\n3. Encrypting and storing credentials...")
    try:
        credential_manager.store_binance_credentials(api_key, api_secret, testnet)
        print("✅ Credentials encrypted and stored")
        
    except Exception as e:
        print(f"❌ Failed to encrypt credentials: {e}")
        return False
    
    # Step 4: Verify encrypted credentials work
    print("\n4. Verifying encrypted credentials...")
    try:
        test_credentials = credential_manager.get_binance_credentials()
        if (test_credentials['api_key'] == api_key and 
            test_credentials['api_secret'] == api_secret):
            print("✅ Encrypted credentials verified successfully")
        else:
            print("❌ Encrypted credentials don't match original")
            return False
            
    except Exception as e:
        print(f"❌ Failed to verify encrypted credentials: {e}")
        return False
    
    # Step 5: Backup original .env file
    print("\n5. Backing up original .env file...")
    try:
        env_file = Path.cwd() / '.env'
        if env_file.exists():
            backup_file = Path.cwd() / '.env.backup'
            env_file.rename(backup_file)
            print(f"✅ Original .env backed up to {backup_file}")
        
    except Exception as e:
        print(f"⚠️  Failed to backup .env file: {e}")
    
    print("\n🎉 MIGRATION COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("✅ Your credentials are now encrypted and secure")
    print("✅ Plain text credentials have been replaced")
    print("✅ Original .env file backed up")
    print()
    print("NEXT STEPS:")
    print("1. Update your services to use SecureCredentialManager")
    print("2. Test the application with encrypted credentials")
    print("3. Delete the .env.backup file once everything works")
    print()
    
    return True


def verify_secure_setup():
    """Verify that secure credential setup is working"""
    print("🔍 VERIFYING SECURE CREDENTIAL SETUP")
    print("=" * 50)
    print()
    
    try:
        # Test credential manager
        credential_manager = SecureCredentialManager()
        
        if credential_manager.validate_credentials():
            credentials = credential_manager.get_binance_credentials()
            api_key = credentials['api_key']
            print(f"✅ Secure credentials loaded (API Key: {api_key[:8]}...)")
            
            # Test with Binance service
            from app.services.binance_service import BinanceService
            binance = BinanceService.create_secure()
            
            if binance.check_api_connectivity():
                print("✅ Binance API connectivity verified")
            else:
                print("⚠️  Binance API not accessible (check network/credentials)")
            
            return True
        else:
            print("❌ Secure credentials validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Secure setup verification failed: {e}")
        return False


def show_security_status():
    """Show current security status"""
    print("🛡️  SECURITY STATUS REPORT")
    print("=" * 50)
    print()
    
    # Check for encrypted credentials
    encrypted_key = os.getenv('BINANCE_API_KEY_ENCRYPTED')
    encrypted_secret = os.getenv('BINANCE_API_SECRET_ENCRYPTED')
    
    # Check for plain text credentials
    plain_key = os.getenv('BINANCE_API_KEY')
    plain_secret = os.getenv('BINANCE_API_SECRET')
    
    print("CREDENTIAL STATUS:")
    if encrypted_key and encrypted_secret:
        print("✅ Encrypted credentials: FOUND")
    else:
        print("❌ Encrypted credentials: NOT FOUND")
    
    if plain_key and plain_secret:
        print("⚠️  Plain text credentials: FOUND (should be removed)")
    else:
        print("✅ Plain text credentials: NOT FOUND")
    
    # Check for salt file
    salt_file = Path.cwd() / '.crypto_salt'
    if salt_file.exists():
        print("✅ Encryption salt file: FOUND")
    else:
        print("❌ Encryption salt file: NOT FOUND")
    
    # Check for master password
    master_password = os.getenv('CRYPTO_MASTER_PASSWORD')
    if master_password:
        print("✅ Master password: SET")
    else:
        print("⚠️  Master password: USING DEFAULT (change for production)")
    
    print()
    print("SECURITY RECOMMENDATIONS:")
    
    if plain_key and plain_secret:
        print("🔥 URGENT: Migrate to encrypted credentials immediately")
    
    if not master_password:
        print("⚠️  Set CRYPTO_MASTER_PASSWORD environment variable")
    
    if not (encrypted_key and encrypted_secret):
        print("⚠️  Run credential migration to encrypt your API keys")
    
    print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "migrate":
            migrate_credentials()
        elif command == "verify":
            verify_secure_setup()
        elif command == "status":
            show_security_status()
        else:
            print("Usage: python migrate_credentials.py [migrate|verify|status]")
    else:
        # Interactive mode
        print("🔐 CRYPTO TRADING SECURITY TOOL")
        print("=" * 50)
        print("1. Migrate to encrypted credentials")
        print("2. Verify secure setup")
        print("3. Show security status")
        print("4. Exit")
        print()
        
        while True:
            choice = input("Select option (1-4): ").strip()
            
            if choice == "1":
                migrate_credentials()
                break
            elif choice == "2":
                verify_secure_setup()
                break
            elif choice == "3":
                show_security_status()
                break
            elif choice == "4":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1-4.")