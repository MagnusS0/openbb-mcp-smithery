"""Tests for PAT utilities."""

import base64
from unittest.mock import Mock

from openbb_auth_pat.pat_utils import (
    extract_pat_from_headers,
    get_pat_hash,
    sanitize_pat_for_logging,
)

# Valid JWT-like tokens
FAKE_PAT = base64.b64encode(b"unit-test-non-secret-1").decode()
ANOTHER_FAKE_PAT = base64.b64encode(b"unit-test-non-secret-2").decode()


def test_extract_pat_from_authorization_header():
    request = Mock()
    request.headers = {"Authorization": f"Bearer {FAKE_PAT}"}

    pat = extract_pat_from_headers(request)
    assert pat == FAKE_PAT


def test_extract_pat_from_openbb_headers():
    request = Mock()
    request.headers = {"X-OpenBB-PAT": FAKE_PAT}
    assert extract_pat_from_headers(request) == FAKE_PAT

    request.headers = {"OpenBB-PAT": ANOTHER_FAKE_PAT}
    assert extract_pat_from_headers(request) == ANOTHER_FAKE_PAT


def test_extract_pat_priority_authorization_wins():
    request = Mock()
    request.headers = {
        "Authorization": f"Bearer {FAKE_PAT}",
        "X-OpenBB-PAT": ANOTHER_FAKE_PAT,
        "OpenBB-PAT": ANOTHER_FAKE_PAT,
    }
    assert extract_pat_from_headers(request) == FAKE_PAT


def test_extract_pat_invalid_and_missing():
    request = Mock()
    request.headers = {}
    assert extract_pat_from_headers(request) is None

    request.headers = {"Authorization": "Basic user:pass"}
    assert extract_pat_from_headers(request) is None

    request.headers = {"X-OpenBB-PAT": "not-a-jwt"}  # invalid format
    assert extract_pat_from_headers(request) is None

    request.headers = {"Authorization": "Bearer header.payload"}  # missing signature
    assert extract_pat_from_headers(request) is None


def test_sanitize_pat_for_logging():
    assert sanitize_pat_for_logging("") == "***"
    assert sanitize_pat_for_logging("short") == "***"
    assert sanitize_pat_for_logging(FAKE_PAT).startswith(FAKE_PAT[:8])
    assert sanitize_pat_for_logging(FAKE_PAT).endswith("...")


def test_get_pat_hash():
    h1 = get_pat_hash(FAKE_PAT)
    h2 = get_pat_hash(ANOTHER_FAKE_PAT)
    assert h1 and h2 and h1 != h2
    assert get_pat_hash(FAKE_PAT) == h1
    assert get_pat_hash("") == ""
    assert get_pat_hash("   ") == ""
