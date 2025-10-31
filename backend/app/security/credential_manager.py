"""
Secure Credential Manager
Handles encrypted storage and retrieval of sensitive API credentials
"""
import os
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class SecureCredentialManager:
    """
    Secure credential manager with encryption and key derivation
    
    Features:
    - Fernet encryption for credentials
    - PBKDF2 key derivation from master password
    - Secure key storage with fallback options
    - Environment-based configuration
    """
    
    def __init__(self, master_password: Optional[str] = None):
        self.master_password = master_password or self._get_master_password()
        self.cipher = self._initialize_cipher()
        self.credentials_cache = {}
        
    def _get_master_password(self) -> str:
        """
        Get master password from environment or prompt user
        Priority: ENV_VAR -> USER_INPUT -> DEFAULT_FALLBACK
        """
        # First try environment variable
        master_pass = os.getenv('CRYPTO_MASTER_PASSWORD')
        if master_pass:
            logger.info("Using master password from environment")
            return master_pass
            
        # Fallback for development - use a default key
        # In production, this should prompt for user input or use secure storage
        logger.warning("Using default master password - CHANGE FOR PRODUCTION!")
        return "crypto_trading_master_key_2025"  # Change this for production!
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,  # High iteration count for security
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def _get_or_create_salt(self) -> bytes:
        """Get existing salt or create new one"""
        salt_file = Path.cwd() / '.crypto_salt'
        
        if salt_file.exists():
            return salt_file.read_bytes()
        else:
            salt = os.urandom(16)
            salt_file.write_bytes(salt)
            # Hide the salt file on Windows
            if os.name == 'nt':
                import subprocess
                try:
                    subprocess.run(['attrib', '+H', str(salt_file)], check=True)
                except subprocess.CalledProcessError:
                    pass  # File hiding failed, continue anyway
            logger.info("Created new encryption salt")
            return salt
    
    def _initialize_cipher(self) -> Fernet:
        """Initialize Fernet cipher with derived key"""
        salt = self._get_or_create_salt()
        key = self._derive_key(self.master_password, salt)
        return Fernet(key)
    
    def encrypt_credential(self, value: str) -> str:
        """Encrypt a credential value"""
        if not value:
            return ""
        encrypted_bytes = self.cipher.encrypt(value.encode())
        return base64.urlsafe_b64encode(encrypted_bytes).decode()
    
    def decrypt_credential(self, encrypted_value: str) -> str:
        """Decrypt a credential value"""
        if not encrypted_value:
            return ""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
            return decrypted_bytes.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt credential: {e}")
            raise ValueError("Invalid or corrupted encrypted credential")
    
    def get_binance_credentials(self) -> Dict[str, Any]:
        """Get decrypted Binance API credentials"""
        credentials = {}
        
        # Try encrypted credentials first
        encrypted_key = os.getenv('BINANCE_API_KEY_ENCRYPTED')
        encrypted_secret = os.getenv('BINANCE_API_SECRET_ENCRYPTED')
        
        if encrypted_key and encrypted_secret:
            try:
                credentials['api_key'] = self.decrypt_credential(encrypted_key)
                credentials['api_secret'] = self.decrypt_credential(encrypted_secret)
                credentials['testnet'] = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
                logger.info("Using encrypted Binance credentials")
                return credentials
            except Exception as e:
                logger.error(f"Failed to decrypt Binance credentials: {e}")
        
        # Fallback to plain text (for migration period)
        plain_key = os.getenv('BINANCE_API_KEY')
        plain_secret = os.getenv('BINANCE_API_SECRET')
        
        if plain_key and plain_secret:
            logger.warning("Using plain text credentials - UPGRADE TO ENCRYPTED!")
            credentials['api_key'] = plain_key
            credentials['api_secret'] = plain_secret
            credentials['testnet'] = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'
            return credentials
        
        logger.error("No Binance credentials found")
        raise ValueError("No Binance API credentials available")
    
    def store_binance_credentials(self, api_key: str, api_secret: str, testnet: bool = False):
        """Store encrypted Binance credentials"""
        try:
            encrypted_key = self.encrypt_credential(api_key)
            encrypted_secret = self.encrypt_credential(api_secret)
            
            # Store in environment or .env file
            env_updates = {
                'BINANCE_API_KEY_ENCRYPTED': encrypted_key,
                'BINANCE_API_SECRET_ENCRYPTED': encrypted_secret,
                'BINANCE_TESTNET': str(testnet).lower()
            }
            
            # Update .env file
            env_file = Path.cwd() / '.env'
            existing_lines = []
            
            if env_file.exists():
                existing_lines = env_file.read_text().splitlines()
            
            # Update or add new values
            updated_lines = []
            keys_updated = set()
            
            for line in existing_lines:
                if '=' in line:
                    key = line.split('=')[0].strip()
                    if key in env_updates:
                        updated_lines.append(f"{key}={env_updates[key]}")
                        keys_updated.add(key)
                    elif key in ['BINANCE_API_KEY', 'BINANCE_API_SECRET']:
                        # Remove plain text versions
                        updated_lines.append(f"# {line}  # Replaced with encrypted version")
                    else:
                        updated_lines.append(line)
                else:
                    updated_lines.append(line)
            
            # Add new keys that weren't updated
            for key, value in env_updates.items():
                if key not in keys_updated:
                    updated_lines.append(f"{key}={value}")
            
            env_file.write_text('\n'.join(updated_lines))
            logger.info("Stored encrypted Binance credentials")
            
        except Exception as e:
            logger.error(f"Failed to store credentials: {e}")
            raise
    
    def validate_credentials(self) -> bool:
        """Validate that credentials can be retrieved and decrypted"""
        try:
            creds = self.get_binance_credentials()
            return bool(creds.get('api_key') and creds.get('api_secret'))
        except Exception:
            return False
    
    def rotate_encryption_key(self, new_master_password: str):
        """Rotate encryption key and re-encrypt all credentials"""
        # Get current credentials
        old_creds = self.get_binance_credentials()
        
        # Update master password and reinitialize cipher
        self.master_password = new_master_password
        
        # Remove old salt to force new key generation
        salt_file = Path.cwd() / '.crypto_salt'
        if salt_file.exists():
            salt_file.unlink()
        
        # Reinitialize with new key
        self.cipher = self._initialize_cipher()
        
        # Re-encrypt and store credentials
        self.store_binance_credentials(
            old_creds['api_key'],
            old_creds['api_secret'],
            old_creds['testnet']
        )
        
        logger.info("Encryption key rotated successfully")


# Global instance
_credential_manager = None


def get_credential_manager() -> SecureCredentialManager:
    """Get global credential manager instance"""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = SecureCredentialManager()
    return _credential_manager


def init_secure_credentials(master_password: Optional[str] = None) -> SecureCredentialManager:
    """Initialize secure credential manager"""
    global _credential_manager
    _credential_manager = SecureCredentialManager(master_password)
    return _credential_manager