"""
api_client.py - Centralized API client for FastAPI backend.

Provides:
- Cached HTTP methods using @st.cache_data for snappy UI
- Automatic fallback to local utils on connection failure
- Error handling with user-friendly messages
- Type-safe response handling via APIResponse dataclass

Usage:
    from frontend.api_client import get_players, search_players, get_similar_players
    
    response = get_players(league="Premier League", min_age=18, max_age=25)
    if response.success:
        players = response.data
    else:
        st.error(response.error)
"""

import streamlit as st
import httpx
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import os

# Backend URL - configurable via environment variable for production deployment
API_BASE_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

# Default timeout for API calls (seconds)
API_TIMEOUT = 10.0


@dataclass
class APIResponse:
    """Standardized response wrapper for all API calls."""
    success: bool
    data: Any
    error: Optional[str] = None
    status_code: Optional[int] = None


def _make_request(
    method: str,
    endpoint: str,
    params: Optional[Dict] = None,
    json_body: Optional[Dict] = None,
    timeout: float = API_TIMEOUT
) -> APIResponse:
    """
    Internal helper for making HTTP requests.
    
    Args:
        method: HTTP method ('GET', 'POST', etc.)
        endpoint: API endpoint path (e.g., '/players/')
        params: Query parameters for GET requests
        json_body: JSON body for POST requests
        timeout: Request timeout in seconds
        
    Returns:
        APIResponse with success status, data, and optional error
    """
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        with httpx.Client(timeout=timeout) as client:
            if method.upper() == "GET":
                response = client.get(url, params=params)
            elif method.upper() == "POST":
                response = client.post(url, json=json_body)
            else:
                return APIResponse(
                    success=False,
                    data=None,
                    error=f"Unsupported HTTP method: {method}"
                )
            
            # Check for successful status codes
            if response.status_code >= 200 and response.status_code < 300:
                return APIResponse(
                    success=True,
                    data=response.json(),
                    status_code=response.status_code
                )
            else:
                return APIResponse(
                    success=False,
                    data=None,
                    error=f"API returned status {response.status_code}: {response.text}",
                    status_code=response.status_code
                )
                
    except httpx.ConnectError:
        return APIResponse(
            success=False,
            data=None,
            error="Cannot connect to backend. Is the FastAPI server running?"
        )
    except httpx.TimeoutException:
        return APIResponse(
            success=False,
            data=None,
            error=f"Request timed out after {timeout} seconds"
        )
    except Exception as e:
        return APIResponse(
            success=False,
            data=None,
            error=f"Unexpected error: {str(e)}"
        )


# =============================================================================
# HEALTH CHECK (Not cached - always check live status)
# =============================================================================

def check_backend_health() -> Tuple[bool, Dict]:
    """
    Check if backend is reachable and healthy.
    
    Returns:
        Tuple of (is_healthy: bool, health_data: dict)
    """
    response = _make_request("GET", "/health")
    if response.success:
        return True, response.data
    return False, {"error": response.error}


# =============================================================================
# PLAYER ENDPOINTS (Cached)
# =============================================================================

@st.cache_data(ttl=300, show_spinner=False)  # 5 minute cache
def get_players(
    skip: int = 0,
    limit: int = 100,
    league: Optional[str] = None,
    position: Optional[str] = None,
    search: Optional[str] = None,
    min_age: Optional[int] = None,
    max_age: Optional[int] = None,
    min_goals: Optional[float] = None,
    min_assists: Optional[float] = None,
    archetype: Optional[str] = None,
    sort_by: Optional[str] = None,
) -> APIResponse:
    """
    Fetch players with optional filters from backend.
    
    Args:
        skip: Pagination offset
        limit: Max results to return
        league: Filter by league name
        position: Filter by position (FW, MF, DF, GK)
        search: Search by player name (partial match)
        min_age: Minimum age filter
        max_age: Maximum age filter
        min_goals: Minimum Gls/90 filter
        min_assists: Minimum Ast/90 filter
        archetype: Filter by player archetype
        sort_by: Sort by stat field (e.g., "Gls/90")
        
    Returns:
        APIResponse with list of players or error
    """
    params = {"skip": skip, "limit": limit}
    
    # Only add non-None params
    if league and league.lower() != "all":
        params["league"] = league
    if position and position.lower() != "all":
        params["position"] = position
    if search:
        params["search"] = search
    if min_age is not None:
        params["min_age"] = min_age
    if max_age is not None:
        params["max_age"] = max_age
    if min_goals is not None:
        params["min_goals"] = min_goals
    if min_assists is not None:
        params["min_assists"] = min_assists
    if archetype:
        params["archetype"] = archetype
    if sort_by:
        params["sort_by"] = sort_by
    
    return _make_request("GET", "/players/", params=params)


@st.cache_data(ttl=60, show_spinner=False)  # 1 minute cache
def get_player_by_id(player_id: int) -> APIResponse:
    """
    Fetch a single player by database ID.
    
    Args:
        player_id: Database ID of the player
        
    Returns:
        APIResponse with player data or error
    """
    return _make_request("GET", f"/players/{player_id}")


@st.cache_data(ttl=300, show_spinner=False)  # 5 minute cache
def search_players(
    query: str,
    league: str = "all",
    limit: int = 10
) -> APIResponse:
    """
    Autocomplete search for player names.
    
    Args:
        query: Search query string
        league: Filter by league ("all" for no filter)
        limit: Max suggestions to return
        
    Returns:
        APIResponse with list of (name, score) tuples or error
    """
    params = {
        "query": query,
        "limit": limit
    }
    if league and league.lower() != "all":
        params["league"] = league
    
    return _make_request("GET", "/players/search", params=params)


@st.cache_data(ttl=300, show_spinner=False)  # 5 minute cache
def get_leagues() -> APIResponse:
    """
    Get list of all leagues with player counts.
    
    Returns:
        APIResponse with list of {"league": str, "player_count": int}
    """
    return _make_request("GET", "/leagues/")


# =============================================================================
# SIMILARITY SEARCH (Cached)
# =============================================================================

@st.cache_data(ttl=120, show_spinner=False)  # 2 minute cache
def get_similar_players(
    player_id: int,
    league: str = "all",
    top_n: int = 5,
    use_position_weights: bool = True,
    scouting_priority: str = "Standard",
    target_league_tier: Optional[str] = None,
) -> APIResponse:
    """
    Server-side similarity search.
    
    Offloads the heavy cosine similarity calculation to the backend.
    
    Args:
        player_id: Database ID of target player
        league: Filter results by league ("all" for all leagues)
        top_n: Number of similar players to return
        use_position_weights: Apply position-specific weighting
        scouting_priority: Scouting priority profile name
        target_league_tier: Target league tier for metric availability
        
    Returns:
        APIResponse with similarity results or error
    """
    request_body = {
        "player_id": player_id,
        "league": league,
        "top_n": top_n,
        "use_position_weights": use_position_weights,
        "scouting_priority": scouting_priority,
    }
    
    if target_league_tier:
        request_body["target_league_tier"] = target_league_tier
    
    return _make_request("POST", "/analysis/similarity", json_body=request_body)


@st.cache_data(ttl=120, show_spinner=False)  # 2 minute cache
def get_feature_attribution(
    player1_id: int,
    player2_id: int,
    use_position_weights: bool = True
) -> APIResponse:
    """
    Get detailed feature attribution between two players.
    
    Args:
        player1_id: Database ID of first player (target)
        player2_id: Database ID of second player (comparison)
        use_position_weights: Apply position-specific weighting
        
    Returns:
        APIResponse with feature attribution data or error
    """
    request_body = {
        "player1_id": player1_id,
        "player2_id": player2_id,
        "use_position_weights": use_position_weights,
    }
    
    return _make_request("POST", "/analysis/attribution", json_body=request_body)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clear_api_cache():
    """Clear all cached API responses. Call when data changes."""
    get_players.clear()
    get_player_by_id.clear()
    search_players.clear()
    get_leagues.clear()
    get_similar_players.clear()
    get_feature_attribution.clear()


def is_backend_available() -> bool:
    """Quick check if backend is reachable (cached result)."""
    healthy, _ = check_backend_health()
    return healthy
