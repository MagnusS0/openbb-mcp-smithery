"""Thread-safe PAT session manager with LRU cache and TTL."""

import logging
import threading
import time
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from openbb_core.app.model.user_settings import UserSettings
from openbb_core.app.service.hub_service import HubService
from openbb_core.app.static.account import Account

from .pat_utils import get_pat_hash, sanitize_pat_for_logging

if TYPE_CHECKING:  # Import types only for static checking to avoid circular imports
    from openbb_core.app.static.app_factory import BaseApp

logger = logging.getLogger(__name__)


class SessionCache:
    """LRU cache with TTL for authenticated sessions."""

    def __init__(
        self, max_size: int = 100, ttl_seconds: int = 1800
    ):  # 30 minutes default
        """Initialize the session cache.

        Args:
            max_size: Maximum number of cached sessions
            ttl_seconds: Time-to-live for cached sessions in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[UserSettings, float]] = {}
        self._access_order: Dict[str, float] = {}
        self._lock = threading.RLock()

    def get(self, pat_hash: str) -> Optional[UserSettings]:
        """Get a cached session if it exists and is not expired.

        Args:
            pat_hash: Hashed PAT token

        Returns:
            UserSettings if found and valid, None otherwise
        """
        with self._lock:
            if pat_hash not in self._cache:
                return None

            user_settings, timestamp = self._cache[pat_hash]
            current_time = time.time()

            # Check if expired
            if current_time - timestamp > self.ttl_seconds:
                self._remove_expired(pat_hash)
                return None

            # Update access time for LRU
            self._access_order[pat_hash] = current_time
            logger.debug("Session cache hit for PAT")
            return user_settings

    def put(self, pat_hash: str, user_settings: UserSettings) -> None:
        """Store a session in the cache.

        Args:
            pat_hash: Hashed PAT token
            user_settings: User settings to cache
        """
        with self._lock:
            current_time = time.time()

            # Remove expired entries
            self._cleanup_expired()

            # If at capacity, remove LRU item
            if len(self._cache) >= self.max_size and pat_hash not in self._cache:
                self._remove_lru()

            self._cache[pat_hash] = (user_settings, current_time)
            self._access_order[pat_hash] = current_time
            logger.debug("Session cached for PAT")

    def remove(self, pat_hash: str) -> None:
        """Remove a session from the cache.

        Args:
            pat_hash: Hashed PAT token to remove
        """
        with self._lock:
            self._remove_expired(pat_hash)

    def clear(self) -> None:
        """Clear all cached sessions."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            logger.info("Session cache cleared")

    def size(self) -> int:
        """Get the current cache size."""
        with self._lock:
            return len(self._cache)

    def _remove_expired(self, pat_hash: str) -> None:
        """Remove an expired entry (called with lock held)."""
        if pat_hash in self._cache:
            del self._cache[pat_hash]
        if pat_hash in self._access_order:
            del self._access_order[pat_hash]
            logger.debug("Removed session from cache")

    def _cleanup_expired(self) -> None:
        """Clean up all expired entries (called with lock held)."""
        current_time = time.time()
        expired_keys = []

        for pat_hash, (_, timestamp) in self._cache.items():
            if current_time - timestamp > self.ttl_seconds:
                expired_keys.append(pat_hash)

        for key in expired_keys:
            self._remove_expired(key)

        if expired_keys:
            logger.debug("Cleaned up %d expired sessions", len(expired_keys))

    def _remove_lru(self) -> None:
        """Remove the least recently used item (called with lock held)."""
        if not self._access_order:
            return

        lru_key = min(self._access_order.keys(), key=lambda k: self._access_order[k])
        self._remove_expired(lru_key)

    logger.debug("Removed LRU session from cache")


class PATSessionManager:
    """Thread-safe manager for PAT-based authentication sessions."""

    _instance: Optional["PATSessionManager"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, cache_size: int = 100, cache_ttl: int = 1800):
        """Initialize the session manager.

        Args:
            cache_size: Maximum number of cached sessions
            cache_ttl: Cache TTL in seconds (default 30 minutes)
        """
        if hasattr(self, "_initialized"):
            return

        self._cache = SessionCache(cache_size, cache_ttl)
        self._base_app: Optional[BaseApp] = None
        self._initialized = True
        logger.info(
            "PAT Session Manager initialized (cache_size=%d, ttl=%ds)",
            cache_size,
            cache_ttl,
        )

    def set_base_app(self, base_app: "BaseApp") -> None:
        """Set the base app instance for creating account services.

        Args:
            base_app: The OpenBB base app instance
        """
        self._base_app = base_app
        logger.debug("Base app set for PAT session manager")

    async def authenticate_pat(self, pat: str) -> Optional[UserSettings]:
        """Authenticate a PAT and return user settings.

        Args:
            pat: Personal Access Token

        Returns:
            UserSettings if authentication successful, None otherwise
        """
        if not pat or not pat.strip():
            logger.warning("Empty PAT provided for authentication")
            return None

        # Hash the PAT for cache key (deterministic SHA-256)
        pat_hash = get_pat_hash(pat)

        # Check cache first
        cached_settings = self._cache.get(pat_hash)
        if cached_settings is not None:
            return cached_settings

        # Authenticate with OpenBB Hub
        try:
            logger.debug(
                "Authenticating PAT with Hub: %s", sanitize_pat_for_logging(pat)
            )

            # Create a temporary account service for authentication
            if self._base_app is None:
                # If no base app, create a minimal HubService directly
                hub_service = HubService()
                hub_service.connect(pat=pat)
                user_settings = hub_service.pull()
            else:
                # Use the proper Account service
                account = Account(self._base_app)
                user_settings = account.login(pat=pat, return_settings=True)

            if user_settings is not None:
                # Cache the successful authentication
                self._cache.put(pat_hash, user_settings)
                logger.info(
                    "PAT authentication successful: %s", sanitize_pat_for_logging(pat)
                )
                return user_settings
            logger.warning(
                "PAT authentication failed: %s", sanitize_pat_for_logging(pat)
            )
            return None

        except Exception as e:
            logger.error("PAT authentication error: %s", e)
            return None

    def logout_pat(self, pat: str) -> None:
        """Remove a PAT session from the cache.

        Args:
            pat: Personal Access Token to logout
        """
        if not pat or not pat.strip():
            return

        pat_hash = get_pat_hash(pat)
        self._cache.remove(pat_hash)
        logger.debug("PAT session logged out: %s", sanitize_pat_for_logging(pat))

    def clear_all_sessions(self) -> None:
        """Clear all cached sessions."""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "size": self._cache.size(),
            "max_size": self._cache.max_size,
            "ttl_seconds": self._cache.ttl_seconds,
        }


# Global session manager instance
session_manager = PATSessionManager()
