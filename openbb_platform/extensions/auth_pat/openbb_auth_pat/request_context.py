"""Request context management for isolating user settings per request."""

import contextvars
import logging
from typing import Optional

from openbb_core.app.model.user_settings import UserSettings

logger = logging.getLogger(__name__)

# Context variables for request-scoped user settings
_user_settings_context: contextvars.ContextVar[Optional[UserSettings]] = (
    contextvars.ContextVar("user_settings", default=None)
)
_pat_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "pat", default=None
)


class RequestContext:
    """Manages request-scoped context for user authentication."""

    @staticmethod
    def set_user_settings(user_settings: UserSettings) -> None:
        """Set user settings for the current request context.

        Args:
            user_settings: User settings to set for this request
        """
        _user_settings_context.set(user_settings)
        logger.debug("User settings set in request context")

    @staticmethod
    def get_user_settings() -> Optional[UserSettings]:
        """Get user settings from the current request context.

        Returns:
            UserSettings if set, None otherwise
        """
        return _user_settings_context.get()

    @staticmethod
    def set_pat(pat: str) -> None:
        """Set PAT for the current request context.

        Args:
            pat: Personal Access Token
        """
        _pat_context.set(pat)
        logger.debug("PAT set in request context")

    @staticmethod
    def get_pat() -> Optional[str]:
        """Get PAT from the current request context.

        Returns:
            PAT if set, None otherwise
        """
        return _pat_context.get()

    @staticmethod
    def clear() -> None:
        """Clear all context variables for the current request."""
        _user_settings_context.set(None)
        _pat_context.set(None)
        logger.debug("Request context cleared")

    @staticmethod
    def is_authenticated() -> bool:
        """Check if the current request is authenticated.

        Returns:
            True if user settings are present, False otherwise
        """
        return _user_settings_context.get() is not None


class RequestContextManager:
    """Context manager for request-scoped authentication."""

    def __init__(
        self, user_settings: Optional[UserSettings] = None, pat: Optional[str] = None
    ):
        """Initialize the context manager.

        Args:
            user_settings: User settings to set for this context
            pat: PAT to set for this context
        """
        self.user_settings = user_settings
        self.pat = pat
        self._previous_user_settings = None
        self._previous_pat: Optional[str] = None

    def __enter__(self):
        """Enter the context manager."""
        # Save previous context
        self._previous_user_settings = RequestContext.get_user_settings()
        self._previous_pat = RequestContext.get_pat()

        # Set new context
        if self.user_settings is not None:
            RequestContext.set_user_settings(self.user_settings)
        if self.pat is not None:
            RequestContext.set_pat(self.pat)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        # Restore previous context
        if self._previous_user_settings is not None:
            RequestContext.set_user_settings(self._previous_user_settings)
        else:
            _user_settings_context.set(None)

        if self._previous_pat is not None:
            RequestContext.set_pat(self._previous_pat)
        else:
            _pat_context.set(None)
