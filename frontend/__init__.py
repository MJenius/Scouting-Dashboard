"""
frontend package - API client and UI utilities for Scouting Dashboard.
"""

from .api_client import (
    APIResponse,
    get_players,
    get_player_by_id,
    search_players,
    get_leagues,
    get_similar_players,
    get_feature_attribution,
    check_backend_health,
    is_backend_available,
    clear_api_cache,
    API_BASE_URL,
)

__all__ = [
    "APIResponse",
    "get_players",
    "get_player_by_id",
    "search_players",
    "get_leagues",
    "get_similar_players",
    "get_feature_attribution",
    "check_backend_health",
    "is_backend_available",
    "clear_api_cache",
    "API_BASE_URL",
]
