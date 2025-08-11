"""OpenBB PAT Authentication Extension.

Multi-session Personal Access Token authentication for OpenBB Platform REST API.
"""

from .auth_extension import auth_hook, router, user_settings_hook

__all__ = ["router", "auth_hook", "user_settings_hook"]

# Export the auth extension entry point
auth_extension = {
    "router": router,
    "auth_hook": auth_hook,
    "user_settings_hook": user_settings_hook,
}
