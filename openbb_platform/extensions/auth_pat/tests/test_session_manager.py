"""Tests for PAT session manager."""

import time
from unittest.mock import Mock, patch

import pytest
from openbb_auth_pat.session_manager import PATSessionManager, SessionCache
from openbb_core.app.model.user_settings import UserSettings


class TestSessionCache:
    """Test the SessionCache class."""

    def test_cache_initialization(self):
        """Test cache initialization with custom parameters."""
        cache = SessionCache(max_size=50, ttl_seconds=900)
        assert cache.max_size == 50
        assert cache.ttl_seconds == 900
        assert cache.size() == 0

    def test_put_and_get(self):
        """Test basic put and get operations."""
        cache = SessionCache(max_size=10, ttl_seconds=3600)
        user_settings = Mock(spec=UserSettings)

        cache.put("test_hash", user_settings)
        assert cache.size() == 1

        retrieved = cache.get("test_hash")
        assert retrieved is user_settings

    def test_ttl_expiration(self):
        """Test that entries expire after TTL."""
        cache = SessionCache(max_size=10, ttl_seconds=1)  # 1 second TTL
        user_settings = Mock(spec=UserSettings)

        cache.put("test_hash", user_settings)
        assert cache.get("test_hash") is user_settings

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("test_hash") is None
        assert cache.size() == 0

    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = SessionCache(max_size=2, ttl_seconds=3600)
        user_settings1 = Mock(spec=UserSettings)
        user_settings2 = Mock(spec=UserSettings)
        user_settings3 = Mock(spec=UserSettings)

        cache.put("hash1", user_settings1)
        cache.put("hash2", user_settings2)
        assert cache.size() == 2

        # Access hash1 to make it more recently used
        cache.get("hash1")

        # Add a third item, should evict hash2 (least recently used)
        cache.put("hash3", user_settings3)
        assert cache.size() == 2
        assert cache.get("hash1") is user_settings1
        assert cache.get("hash2") is None  # Should be evicted
        assert cache.get("hash3") is user_settings3

    def test_clear(self):
        """Test cache clearing."""
        cache = SessionCache()
        user_settings = Mock(spec=UserSettings)

        cache.put("test_hash", user_settings)
        assert cache.size() == 1

        cache.clear()
        assert cache.size() == 0
        assert cache.get("test_hash") is None


class TestPATSessionManager:
    """Test the PATSessionManager class."""

    @pytest.fixture
    def session_manager(self):
        """Create a fresh session manager for each test."""
        # Reset the singleton for testing purposes (W0212 warning suppressed)
        # pylint: disable=protected-access
        PATSessionManager._instance = None
        return PATSessionManager(cache_size=10, cache_ttl=3600)

    def test_singleton_pattern(self):
        """Test that PATSessionManager follows singleton pattern."""
        manager1 = PATSessionManager()
        manager2 = PATSessionManager()
        assert manager1 is manager2

    @pytest.mark.asyncio
    async def test_authenticate_pat_invalid(self, session_manager):
        """Test authentication with invalid PAT."""
        result = await session_manager.authenticate_pat("")
        assert result is None

        result = await session_manager.authenticate_pat("invalid")
        assert result is None

    @pytest.mark.asyncio
    @patch("openbb_auth_pat.session_manager.HubService")
    async def test_authenticate_pat_success(self, mock_hub_service, session_manager):
        """Test successful PAT authentication."""
        # Mock successful authentication
        mock_user_settings = Mock(spec=UserSettings)
        mock_hub_instance = mock_hub_service.return_value
        mock_hub_instance.pull.return_value = mock_user_settings

        result = await session_manager.authenticate_pat("valid_pat_token")
        assert result is mock_user_settings

        # Test caching - second call should hit cache
        result2 = await session_manager.authenticate_pat("valid_pat_token")
        assert result2 is mock_user_settings

        # Hub service should only be called once due to caching
        mock_hub_service.assert_called_once()

    def test_logout_pat(self, session_manager):
        """Test PAT logout functionality."""
        # This mainly tests that logout doesn't crash
        session_manager.logout_pat("some_pat")
        session_manager.logout_pat("")

    def test_get_cache_stats(self, session_manager):
        """Test cache statistics retrieval."""
        stats = session_manager.get_cache_stats()
        assert "size" in stats
        assert "max_size" in stats
        assert "ttl_seconds" in stats
        assert stats["size"] == 0
        assert stats["max_size"] == 10
        assert stats["ttl_seconds"] == 3600
