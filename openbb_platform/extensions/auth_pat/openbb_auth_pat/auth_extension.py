"""PAT Authentication Extension for OpenBB Platform.

Implements the AuthService plugin interface for multi-session PAT authentication.
"""

import logging

from fastapi import APIRouter, HTTPException, Request, status
from openbb_core.app.model.user_settings import UserSettings

from .pat_utils import extract_pat_from_headers, sanitize_pat_for_logging
from .request_context import RequestContext
from .session_manager import session_manager

logger = logging.getLogger(__name__)

# Create router for auth-related endpoints
router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.get("/status")
async def auth_status(request: Request) -> dict:
    """Get authentication status for the current request."""
    pat = extract_pat_from_headers(request)
    is_authenticated = RequestContext.is_authenticated()

    return {
        "authenticated": is_authenticated,
        "pat_provided": pat is not None,
        "cache_stats": session_manager.get_cache_stats(),
    }


@router.post("/logout")
async def logout(request: Request) -> dict:
    """Logout the current PAT session."""
    pat = extract_pat_from_headers(request)
    if pat:
        session_manager.logout_pat(pat)
        RequestContext.clear()
        logger.info("PAT session logged out: %s", sanitize_pat_for_logging(pat))
        return {"message": "Logged out successfully"}
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST, detail="No PAT provided for logout"
    )


@router.delete("/sessions")
async def clear_all_sessions() -> dict:
    """Clear all cached sessions (admin function)."""
    session_manager.clear_all_sessions()
    logger.info("All PAT sessions cleared")
    return {"message": "All sessions cleared"}


async def auth_hook(request: Request) -> None:
    """Authenticate using the provided PAT and populate the request context, called by AuthService.

    Args:
        request: FastAPI request object

    Raises:
        HTTPException: If authentication fails
    """
    # Extract PAT from request headers
    pat = extract_pat_from_headers(request)

    if not pat:
        # No PAT provided - this might be OK for some endpoints
        logger.debug("No PAT provided in request headers")
        RequestContext.clear()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid PAT token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Authenticate the PAT
    user_settings = await session_manager.authenticate_pat(pat)

    if user_settings is None:
        logger.warning("PAT authentication failed: %s", sanitize_pat_for_logging(pat))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid PAT token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Set up request context
    RequestContext.set_user_settings(user_settings)
    RequestContext.set_pat(pat)

    logger.debug("PAT authentication successful: %s", sanitize_pat_for_logging(pat))


async def user_settings_hook(request: Request) -> UserSettings:
    """User settings hook called by AuthService.

    If no user settings are present in the per-request context, attempt
    to authenticate using the PAT from request headers and populate the
    context. This aligns with how the Platform API injects user settings
    into command endpoints via dependency injection.

    Args:
        request: FastAPI request object

    Returns:
        UserSettings for the authenticated user

    Raises:
        HTTPException: If authentication fails or settings cannot be obtained
    """
    # Return fast path if already authenticated in this request context
    user_settings = RequestContext.get_user_settings()
    if user_settings is not None:
        logger.debug("User settings retrieved from request context")
        return user_settings

    # No context set yet; perform on-demand authentication from headers
    pat = extract_pat_from_headers(request)
    if not pat:
        logger.debug("No PAT provided in request headers for user_settings_hook")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid PAT token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    authenticated_settings = await session_manager.authenticate_pat(pat)
    if authenticated_settings is None:
        logger.warning(
            "PAT authentication failed in user_settings_hook: %s",
            sanitize_pat_for_logging(pat),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid PAT token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Populate context for downstream code in the same request
    RequestContext.set_user_settings(authenticated_settings)
    RequestContext.set_pat(pat)
    logger.debug("User settings set in context via user_settings_hook authentication")
    return authenticated_settings


# Optional initialization function that can be called when the extension is loaded
def initialize_extension(base_app=None) -> None:
    """Initialize the PAT authentication extension.

    Args:
        base_app: Optional base app instance for proper integration
    """
    if base_app is not None:
        session_manager.set_base_app(base_app)
        logger.info("PAT auth extension initialized with base app")
        return
    logger.info("PAT auth extension initialized without base app")


# Export the required symbols for AuthService
__all__ = ["router", "auth_hook", "user_settings_hook", "initialize_extension"]
