"""
Security Middleware
Provides authentication, rate limiting, and request validation
"""
import os
import time
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Callable
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.security.auth_manager import get_auth_manager

logger = logging.getLogger(__name__)

# Rate limiting storage
rate_limit_storage = defaultdict(lambda: deque())
blocked_ips = {}


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware
    
    Features:
    - Per-IP rate limiting
    - Configurable limits per endpoint
    - Temporary IP blocking for abuse
    - Different limits for authenticated vs anonymous users
    """
    
    def __init__(self, app, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.block_duration = timedelta(minutes=15)  # Block abusive IPs for 15 minutes
        
    async def dispatch(self, request: Request, call_next: Callable):
        client_ip = self._get_client_ip(request)
        
        # Check if IP is temporarily blocked
        if self._is_ip_blocked(client_ip):
            return JSONResponse(
                status_code=429,
                content={"error": "IP temporarily blocked due to abuse"}
            )
        
        # Check rate limits
        if not self._check_rate_limit(client_ip, request):
            # Record potential abuse
            self._record_abuse(client_ip)
            
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {self.requests_per_minute} requests per minute allowed"
                }
            )
        
        response = await call_next(request)
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check for forwarded IP (if behind proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, ip: str, request: Request) -> bool:
        """Check if request is within rate limits"""
        now = time.time()
        minute_ago = now - 60
        hour_ago = now - 3600
        
        # Get request history for this IP
        requests = rate_limit_storage[ip]
        
        # Remove old requests
        while requests and requests[0] < hour_ago:
            requests.popleft()
        
        # Count recent requests
        minute_requests = sum(1 for timestamp in requests if timestamp > minute_ago)
        hour_requests = len(requests)
        
        # Check limits
        if minute_requests >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for IP {ip}: {minute_requests} requests in last minute")
            return False
        
        if hour_requests >= self.requests_per_hour:
            logger.warning(f"Rate limit exceeded for IP {ip}: {hour_requests} requests in last hour")
            return False
        
        # Record this request
        requests.append(now)
        
        return True
    
    def _is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is temporarily blocked"""
        if ip not in blocked_ips:
            return False
        
        block_time = blocked_ips[ip]
        if datetime.now() - block_time > self.block_duration:
            # Unblock IP
            del blocked_ips[ip]
            return False
        
        return True
    
    def _record_abuse(self, ip: str):
        """Record potential abuse and block if necessary"""
        # If rate limit exceeded multiple times in short period, block IP
        now = time.time()
        recent_limit = now - 300  # 5 minutes
        
        requests = rate_limit_storage[ip]
        recent_requests = sum(1 for timestamp in requests if timestamp > recent_limit)
        
        if recent_requests > self.requests_per_minute * 3:  # 3x rate limit in 5 minutes
            blocked_ips[ip] = datetime.now()
            logger.warning(f"Blocked IP {ip} for abuse: {recent_requests} requests in 5 minutes")


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self' wss: https:; "
            "font-src 'self'"
        )
        
        return response


# Authentication dependency
security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Dependency to get current authenticated user
    Validates JWT token and returns user info
    """
    auth_manager = get_auth_manager()
    token_data = auth_manager.verify_token(credentials.credentials)
    
    if not token_data:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return {
        "username": token_data["username"],
        "role": token_data["role"]
    }


async def get_admin_user(current_user: dict = Depends(get_current_user)):
    """Dependency to ensure user has admin role"""
    if current_user["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    return current_user


async def get_trader_user(current_user: dict = Depends(get_current_user)):
    """Dependency to ensure user has trader role or higher"""
    auth_manager = get_auth_manager()
    if not auth_manager.has_permission(current_user["username"], "trader"):
        raise HTTPException(
            status_code=403,
            detail="Trader access required"
        )
    return current_user


class APIKeyValidator:
    """Validate API keys for external integrations"""
    
    def __init__(self):
        self.valid_api_keys = set()
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load valid API keys from environment or database"""
        # In production, load from secure storage
        api_keys = os.getenv('VALID_API_KEYS', '').split(',')
        self.valid_api_keys = {key.strip() for key in api_keys if key.strip()}
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate external API key"""
        return api_key in self.valid_api_keys


def require_api_key(api_key: str = None) -> bool:
    """Dependency to validate API key for external access"""
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required"
        )
    
    validator = APIKeyValidator()
    if not validator.validate_api_key(api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return True