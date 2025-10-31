"""
Secure Authentication Routes
Provides login, token management, and user authentication endpoints
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer
from pydantic import BaseModel, EmailStr

from app.security.auth_manager import get_auth_manager
from app.security.middleware import get_current_user, get_admin_user

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()


class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class RefreshRequest(BaseModel):
    """Token refresh request model"""
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    """Change password request model"""
    current_password: str
    new_password: str


class CreateUserRequest(BaseModel):
    """Create user request model"""
    username: str
    password: str
    email: EmailStr
    role: str = "trader"


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, http_request: Request):
    """
    Authenticate user and return JWT tokens
    
    Rate limited to prevent brute force attacks
    """
    auth_manager = get_auth_manager()
    
    # Authenticate user
    user = auth_manager.authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )
    
    # Create tokens
    access_token = auth_manager.create_access_token(user["username"], user["role"])
    refresh_token = auth_manager.create_refresh_token(user["username"])
    
    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=auth_manager.access_token_expire_minutes * 60,
        user={
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "last_login": user["last_login"].isoformat() if user["last_login"] else None
        }
    )


@router.post("/refresh")
async def refresh_token(request: RefreshRequest):
    """
    Refresh access token using refresh token
    """
    auth_manager = get_auth_manager()
    
    new_access_token = auth_manager.refresh_access_token(request.refresh_token)
    if not new_access_token:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired refresh token"
        )
    
    return {
        "access_token": new_access_token,
        "token_type": "bearer",
        "expires_in": auth_manager.access_token_expire_minutes * 60
    }


@router.post("/logout")
async def logout(request: RefreshRequest):
    """
    Logout user by revoking refresh token
    """
    auth_manager = get_auth_manager()
    
    success = auth_manager.revoke_refresh_token(request.refresh_token)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Invalid refresh token"
        )
    
    return {"message": "Successfully logged out"}


@router.get("/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    Get current user information
    Requires valid access token
    """
    auth_manager = get_auth_manager()
    user_data = auth_manager.users_db.get(current_user["username"])
    
    if not user_data:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "username": user_data["username"],
        "email": user_data["email"],
        "role": user_data["role"],
        "is_active": user_data["is_active"],
        "created_at": user_data["created_at"].isoformat(),
        "last_login": user_data["last_login"].isoformat() if user_data["last_login"] else None
    }


@router.post("/change-password")
async def change_password(
    request: ChangePasswordRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Change user password
    Requires current password for verification
    """
    auth_manager = get_auth_manager()
    
    success = auth_manager.change_password(
        current_user["username"],
        request.current_password,
        request.new_password
    )
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Current password is incorrect"
        )
    
    return {"message": "Password changed successfully"}


@router.post("/create-user")
async def create_user(
    request: CreateUserRequest,
    current_user: dict = Depends(get_admin_user)  # Only admin can create users
):
    """
    Create new user account
    Requires admin privileges
    """
    auth_manager = get_auth_manager()
    
    # Validate role
    valid_roles = ["viewer", "trader", "admin"]
    if request.role not in valid_roles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role. Must be one of: {valid_roles}"
        )
    
    success = auth_manager.create_user(
        request.username,
        request.password,
        request.email,
        request.role
    )
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Username already exists"
        )
    
    return {"message": f"User {request.username} created successfully"}


@router.post("/deactivate-user/{username}")
async def deactivate_user(
    username: str,
    current_user: dict = Depends(get_admin_user)  # Only admin can deactivate users
):
    """
    Deactivate user account
    Requires admin privileges
    """
    auth_manager = get_auth_manager()
    
    # Prevent admin from deactivating themselves
    if username == current_user["username"]:
        raise HTTPException(
            status_code=400,
            detail="Cannot deactivate your own account"
        )
    
    success = auth_manager.deactivate_user(username)
    if not success:
        raise HTTPException(
            status_code=404,
            detail="User not found"
        )
    
    return {"message": f"User {username} deactivated successfully"}


@router.get("/users")
async def list_users(current_user: dict = Depends(get_admin_user)):
    """
    List all users
    Requires admin privileges
    """
    auth_manager = get_auth_manager()
    
    users = []
    for username, user_data in auth_manager.users_db.items():
        users.append({
            "username": user_data["username"],
            "email": user_data["email"],
            "role": user_data["role"],
            "is_active": user_data["is_active"],
            "created_at": user_data["created_at"].isoformat(),
            "last_login": user_data["last_login"].isoformat() if user_data["last_login"] else None
        })
    
    return {"users": users}