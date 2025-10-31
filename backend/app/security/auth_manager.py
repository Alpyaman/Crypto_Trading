"""
JWT Authentication System
Provides secure token-based authentication for the trading API
"""
import os
# import jwt
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from pydantic import BaseModel
import secrets
# import hashlib

logger = logging.getLogger(__name__)


class UserModel(BaseModel):
    """User model for authentication"""
    username: str
    email: Optional[str] = None
    role: str = "trader"
    is_active: bool = True
    created_at: datetime
    last_login: Optional[datetime] = None


class TokenData(BaseModel):
    """Token data model"""
    username: str
    role: str
    exp: datetime
    iat: datetime


class JWTAuthManager:
    """
    JWT Authentication Manager
    
    Features:
    - JWT token generation and validation
    - Password hashing with bcrypt
    - Role-based access control
    - Token refresh mechanism
    - Rate limiting per user
    """
    
    def __init__(self):
        self.secret_key = self._get_or_create_secret_key()
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        # Simple in-memory user store (replace with database in production)
        self.users_db = {}
        self.refresh_tokens = {}  # Store valid refresh tokens
        self.failed_attempts = {}  # Track failed login attempts
        
        # Initialize default admin user
        self._create_default_user()
    
    def _get_or_create_secret_key(self) -> str:
        """Get JWT secret key from environment or create new one"""
        secret = os.getenv('JWT_SECRET_KEY')
        if secret:
            return secret
        
        # Generate new secret key
        new_secret = secrets.token_urlsafe(32)
        logger.warning(f"Generated new JWT secret key. Add to .env: JWT_SECRET_KEY={new_secret}")
        return new_secret
    
    def _create_default_user(self):
        """Create default admin user for initial access"""
        default_username = "admin"
        default_password = os.getenv('DEFAULT_ADMIN_PASSWORD', "admin123")  # Change this!
        
        if default_username not in self.users_db:
            hashed_password = self.hash_password(default_password)
            self.users_db[default_username] = {
                "username": default_username,
                "email": "admin@cryptotrading.local",
                "hashed_password": hashed_password,
                "role": "admin",
                "is_active": True,
                "created_at": datetime.now(timezone.utc),
                "last_login": None
            }
            logger.warning(f"Created default admin user. Username: {default_username}, Password: {default_password}")
            logger.warning("CHANGE DEFAULT PASSWORD IMMEDIATELY IN PRODUCTION!")
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user with username and password"""
        # Check for rate limiting
        if self._is_rate_limited(username):
            logger.warning(f"Rate limited login attempt for user: {username}")
            return None
        
        user = self.users_db.get(username)
        if not user:
            self._record_failed_attempt(username)
            return None
        
        if not user["is_active"]:
            self._record_failed_attempt(username)
            return None
        
        if not self.verify_password(password, user["hashed_password"]):
            self._record_failed_attempt(username)
            return None
        
        # Successful authentication
        self._clear_failed_attempts(username)
        user["last_login"] = datetime.now(timezone.utc)
        return user
    
    def _is_rate_limited(self, username: str) -> bool:
        """Check if user is rate limited due to failed attempts"""
        if username not in self.failed_attempts:
            return False
        
        attempts = self.failed_attempts[username]
        if len(attempts) < 5:  # Allow 5 attempts
            return False
        
        # Check if last attempt was within 15 minutes
        last_attempt = max(attempts)
        return (datetime.now(timezone.utc) - last_attempt) < timedelta(minutes=15)
    
    def _record_failed_attempt(self, username: str):
        """Record failed login attempt"""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = []
        
        self.failed_attempts[username].append(datetime.now(timezone.utc))
        
        # Keep only last 10 attempts
        self.failed_attempts[username] = self.failed_attempts[username][-10:]
    
    def _clear_failed_attempts(self, username: str):
        """Clear failed login attempts after successful login"""
        if username in self.failed_attempts:
            del self.failed_attempts[username]
    
    def create_access_token(self, username: str, role: str) -> str:
        """Create JWT access token"""
        expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
        
        payload = {
            "username": username,
            "role": role,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access"
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, username: str) -> str:
        """Create JWT refresh token"""
        expire = datetime.now(timezone.utc) + timedelta(days=self.refresh_token_expire_days)
        
        # Generate unique token ID
        token_id = secrets.token_urlsafe(16)
        
        payload = {
            "username": username,
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh",
            "jti": token_id  # JWT ID for token revocation
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Store refresh token for validation
        self.refresh_tokens[token_id] = {
            "username": username,
            "created_at": datetime.now(timezone.utc),
            "expires_at": expire
        }
        
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if user still exists and is active
            username = payload.get("username")
            if not username or username not in self.users_db:
                return None
            
            user = self.users_db[username]
            if not user["is_active"]:
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Create new access token from refresh token"""
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            
            if payload.get("type") != "refresh":
                return None
            
            # Check if refresh token is still valid
            token_id = payload.get("jti")
            if not token_id or token_id not in self.refresh_tokens:
                return None
            
            username = payload.get("username")
            if not username or username not in self.users_db:
                return None
            
            user = self.users_db[username]
            if not user["is_active"]:
                return None
            
            # Create new access token
            return self.create_access_token(username, user["role"])
            
        except jwt.InvalidTokenError:
            return None
    
    def revoke_refresh_token(self, refresh_token: str) -> bool:
        """Revoke refresh token"""
        try:
            payload = jwt.decode(refresh_token, self.secret_key, algorithms=[self.algorithm])
            token_id = payload.get("jti")
            
            if token_id and token_id in self.refresh_tokens:
                del self.refresh_tokens[token_id]
                return True
                
        except jwt.InvalidTokenError:
            pass
        
        return False
    
    def create_user(self, username: str, password: str, email: str, role: str = "trader") -> bool:
        """Create new user"""
        if username in self.users_db:
            return False
        
        hashed_password = self.hash_password(password)
        
        self.users_db[username] = {
            "username": username,
            "email": email,
            "hashed_password": hashed_password,
            "role": role,
            "is_active": True,
            "created_at": datetime.now(timezone.utc),
            "last_login": None
        }
        
        logger.info(f"Created new user: {username}")
        return True
    
    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change user password"""
        user = self.users_db.get(username)
        if not user:
            return False
        
        if not self.verify_password(old_password, user["hashed_password"]):
            return False
        
        user["hashed_password"] = self.hash_password(new_password)
        logger.info(f"Password changed for user: {username}")
        return True
    
    def deactivate_user(self, username: str) -> bool:
        """Deactivate user account"""
        user = self.users_db.get(username)
        if not user:
            return False
        
        user["is_active"] = False
        logger.info(f"Deactivated user: {username}")
        return True
    
    def has_permission(self, username: str, required_role: str) -> bool:
        """Check if user has required role/permission"""
        user = self.users_db.get(username)
        if not user or not user["is_active"]:
            return False
        
        user_role = user["role"]
        
        # Role hierarchy: admin > trader > viewer
        role_hierarchy = {
            "admin": 3,
            "trader": 2,
            "viewer": 1
        }
        
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level


# Global instance
_auth_manager = None


def get_auth_manager() -> JWTAuthManager:
    """Get global authentication manager instance"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = JWTAuthManager()
    return _auth_manager