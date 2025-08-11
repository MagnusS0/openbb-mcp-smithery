"""Utilities for PAT extraction and validation."""

import hashlib
import logging
import re
from typing import Optional

from fastapi import Request

logger = logging.getLogger(__name__)

# PAT validation pattern
JWT_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_-]+$")


def _is__pat(token: str) -> bool:
    """Heuristic for accepting tokens during extraction: JWT-only."""
    if not token or not isinstance(token, str):
        return False
    token = token.strip()
    # Pre-check to avoid obviously bad inputs and log spam
    if len(token) < 20:
        return False
    if len(token) > 2000:
        return False
    return bool(JWT_PATTERN.fullmatch(token))


def extract_pat_from_headers(request: Request) -> Optional[str]:
    """Extract PAT from request headers.

    Supports multiple header formats:
    - Authorization: Bearer <token>
    - X-OpenBB-PAT: <token>
    - OpenBB-PAT: <token>

    Args:
        request: FastAPI request object

    Returns:
        PAT token if found and valid format, None otherwise
    """
    # Try Authorization: Bearer <token>
    auth_header = request.headers.get("Authorization", "").strip()
    if auth_header:
        if auth_header.lower().startswith("bearer "):
            token = auth_header[7:].strip()
            if _is__pat(token):
                logger.debug("PAT extracted from Authorization header")
                return token
            logger.warning("Invalid-looking PAT in Authorization header")
        else:
            logger.debug("Authorization header present but not Bearer type")

    # Try X-OpenBB-PAT header
    pat_header = request.headers.get("X-OpenBB-PAT", "").strip()
    if pat_header:
        if _is__pat(pat_header):
            logger.debug("PAT extracted from X-OpenBB-PAT header")
            return pat_header
        logger.warning("Invalid-looking PAT in X-OpenBB-PAT header")

    # Try OpenBB-PAT header (without X- prefix)
    pat_header_alt = request.headers.get("OpenBB-PAT", "").strip()
    if pat_header_alt:
        if _is__pat(pat_header_alt):
            logger.debug("PAT extracted from OpenBB-PAT header")
            return pat_header_alt
        logger.warning("Invalid-looking PAT in OpenBB-PAT header")

    logger.debug("No valid PAT found in request headers")
    return None


def sanitize_pat_for_logging(pat: str) -> str:
    """Sanitize PAT for safe logging.

    Args:
        pat: PAT token to sanitize

    Returns:
        Sanitized PAT string showing only first 8 characters
    """
    if not pat or len(pat) < 8:
        return "***"
    return f"{pat[:8]}..."


def get_pat_hash(pat: str) -> str:
    """Get a hash of the PAT for cache keys.

    Args:
        pat: PAT token

    Returns:
        Hash string suitable for use as cache key
    """
    if not pat:
        return ""

    # Normalize and handle whitespace-only tokens
    normalized = pat.strip()
    if not normalized:
        return ""

    # Deterministic hash for cache keys/log correlation
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
