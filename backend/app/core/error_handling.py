"""
Enhanced API error handling and response models
"""
from fastapi import HTTPException, status
from pydantic import BaseModel
from typing import Any, Optional, Dict
import logging
import traceback
from datetime import datetime

logger = logging.getLogger(__name__)

class ErrorResponse(BaseModel):
    """Standard error response model"""
    success: bool = False
    error: str
    error_code: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None

class SuccessResponse(BaseModel):
    """Standard success response model"""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str

class APIResponse:
    """Utility class for standardized API responses"""
    
    @staticmethod
    def success(message: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a standardized success response"""
        return {
            "success": True,
            "message": message,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    @staticmethod
    def error(
        error_message: str, 
        error_code: str = "GENERAL_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ) -> HTTPException:
        """Create a standardized error response"""
        error_response = {
            "success": False,
            "error": error_message,
            "error_code": error_code,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details
        }
        
        # Log the error
        logger.error(f"API Error [{error_code}]: {error_message}")
        if details:
            logger.error(f"Error details: {details}")
        
        return HTTPException(status_code=status_code, detail=error_response)

class TradingAPIException(Exception):
    """Custom exception for trading-related errors"""
    def __init__(self, message: str, error_code: str = "TRADING_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details
        super().__init__(message)

class MLServiceException(Exception):
    """Custom exception for ML service errors"""
    def __init__(self, message: str, error_code: str = "ML_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details
        super().__init__(message)

class BinanceAPIException(Exception):
    """Custom exception for Binance API errors"""
    def __init__(self, message: str, error_code: str = "BINANCE_API_ERROR", details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details
        super().__init__(message)

def handle_trading_exception(e: Exception) -> HTTPException:
    """Handle trading-related exceptions"""
    if isinstance(e, TradingAPIException):
        return APIResponse.error(
            error_message=e.message,
            error_code=e.error_code,
            status_code=status.HTTP_400_BAD_REQUEST,
            details=e.details
        )
    elif isinstance(e, MLServiceException):
        return APIResponse.error(
            error_message=e.message,
            error_code=e.error_code,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=e.details
        )
    elif isinstance(e, BinanceAPIException):
        return APIResponse.error(
            error_message=e.message,
            error_code=e.error_code,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details=e.details
        )
    else:
        # Log unexpected errors with full traceback
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return APIResponse.error(
            error_message="An unexpected error occurred",
            error_code="INTERNAL_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={"original_error": str(e)}
        )

# Common error responses for documentation
COMMON_ERRORS = {
    "INVALID_CREDENTIALS": {
        "description": "Invalid API credentials",
        "status_code": 401,
        "example": {
            "success": False,
            "error": "Invalid API credentials provided",
            "error_code": "INVALID_CREDENTIALS",
            "timestamp": "2025-10-29T10:00:00Z"
        }
    },
    "INSUFFICIENT_BALANCE": {
        "description": "Insufficient account balance for trading",
        "status_code": 400,
        "example": {
            "success": False,
            "error": "Insufficient balance for the requested trade",
            "error_code": "INSUFFICIENT_BALANCE",
            "timestamp": "2025-10-29T10:00:00Z",
            "details": {"required": 100.0, "available": 50.0}
        }
    },
    "MODEL_NOT_READY": {
        "description": "ML model is not trained or ready",
        "status_code": 503,
        "example": {
            "success": False,
            "error": "ML model is not ready for trading",
            "error_code": "MODEL_NOT_READY",
            "timestamp": "2025-10-29T10:00:00Z"
        }
    },
    "MARKET_CLOSED": {
        "description": "Market is closed for trading",
        "status_code": 400,
        "example": {
            "success": False,
            "error": "Market is currently closed",
            "error_code": "MARKET_CLOSED",
            "timestamp": "2025-10-29T10:00:00Z"
        }
    }
}