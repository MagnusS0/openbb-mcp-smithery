"""In-memory LRU session store with optional TTL and cleanup."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class SessionEntry:
    key: str
    value: Any
    created_at: float
    last_used_at: float


class LRUSessionStore:
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[int] = None,
        on_evict: Optional[Callable[[SessionEntry], None]] = None,
    ) -> None:
        self._store: "OrderedDict[str, SessionEntry]" = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.on_evict = on_evict

    def _evict_if_needed(self) -> None:
        # TTL eviction
        if self.ttl_seconds:
            now = time.time()
            expired = [
                k
                for k, entry in list(self._store.items())
                if now - entry.last_used_at > self.ttl_seconds
            ]
            for k in expired:
                entry = self._store.pop(k)
                if self.on_evict:
                    try:
                        self.on_evict(entry)
                    except Exception:  # noqa: BLE001
                        pass

        # Size eviction (LRU)
        while len(self._store) > self.max_size:
            _, entry = self._store.popitem(last=False)
            if self.on_evict:
                try:
                    self.on_evict(entry)
                except Exception:  # noqa: BLE001
                    pass

    def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if not entry:
            return None
        entry.last_used_at = time.time()

        self._store.move_to_end(key)
        self._evict_if_needed()
        return entry.value

    def set(self, key: str, value: Any) -> None:
        now = time.time()
        entry = SessionEntry(key=key, value=value, created_at=now, last_used_at=now)
        self._store[key] = entry
        self._store.move_to_end(key)
        self._evict_if_needed()

    def get_or_create(self, key: str, factory: Callable[[], Any]) -> Any:
        existing = self.get(key)
        if existing is not None:
            return existing
        value = factory()
        self.set(key, value)
        return value

    def delete(self, key: str) -> bool:
        entry = self._store.pop(key, None)
        if entry and self.on_evict:
            try:
                self.on_evict(entry)
            except Exception:  # noqa: BLE001
                pass
        return entry is not None

    def stats(self) -> dict[str, Any]:
        return {
            "activeSessions": len(self._store),
            "maxSize": self.max_size,
            "ttlSeconds": self.ttl_seconds,
        }

    def sweep(self) -> None:
        """Run eviction checks without touching recency for TTL cleanup."""
        self._evict_if_needed()
