"""
schemas.py - Pydantic schemas for API request/response validation.

Provides type-safe serialization for FastAPI endpoints.
"""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List, Dict, Any


# =============================================================================
# PLAYER SCHEMAS
# =============================================================================

class PlayerBase(BaseModel):
    """Base schema with core player fields."""
    name: str
    squad: str
    league: str
    position: Optional[str] = None
    age: Optional[int] = None
    nation: Optional[str] = None
    stats: dict = {}


class PlayerCreate(PlayerBase):
    """Schema for creating new players (used by ETL)."""
    pass


class PlayerOut(PlayerBase):
    """
    Schema for API responses.
    
    Includes database metadata (id, timestamps).
    Uses from_attributes=True for ORM model compatibility.
    """
    id: int
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


# =============================================================================
# HEALTH CHECK
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    database: str
    player_count: int
    engine_loaded: bool = False


# =============================================================================
# SIMILARITY SEARCH SCHEMAS
# =============================================================================

class SimilarityRequest(BaseModel):
    """Request body for similarity search."""
    player_id: int = Field(..., description="Database ID of target player")
    league: str = Field("all", description="Filter results by league")
    top_n: int = Field(5, ge=1, le=20, description="Number of similar players to return")
    use_position_weights: bool = Field(True, description="Apply position-specific weighting")
    scouting_priority: str = Field("Standard", description="Scouting priority profile")
    target_league_tier: Optional[str] = Field(None, description="Target league tier for metric availability")


class SimilarMatch(BaseModel):
    """A single similar player result."""
    id: Optional[int] = Field(None, description="Database ID of matched player")
    name: str = Field(..., description="Player name")
    squad: str = Field(..., description="Squad/team name")
    league: str = Field(..., description="League name")
    position: str = Field(..., description="Primary position")
    match_score: float = Field(..., ge=0, le=100, description="Similarity score (0-100)")
    primary_drivers: str = Field("", description="Text describing similarity drivers")
    stats: Dict[str, Any] = Field(default_factory=dict, description="Key stats for display")


class SimilarityResponse(BaseModel):
    """Response for similarity search."""
    target: PlayerOut = Field(..., description="Target player info")
    matches: List[SimilarMatch] = Field(default_factory=list, description="Similar players")
    proxy_warnings: Optional[str] = Field(None, description="Warnings about metric proxies used")


# =============================================================================
# FEATURE ATTRIBUTION SCHEMAS
# =============================================================================

class AttributionRequest(BaseModel):
    """Request body for feature attribution calculation."""
    player1_id: int = Field(..., description="Database ID of target player")
    player2_id: int = Field(..., description="Database ID of comparison player")
    use_position_weights: bool = Field(True, description="Apply position-specific weighting")


class AttributionResponse(BaseModel):
    """Response for feature attribution."""
    attribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature distances sorted by target's priorities"
    )
    error: Optional[str] = Field(None, description="Error message if calculation failed")


# =============================================================================
# SEARCH SCHEMAS
# =============================================================================

class PlayerSearchResult(BaseModel):
    """Single result from player autocomplete search."""
    name: str = Field(..., description="Display name (Player (Squad))")
    squad: str = Field(..., description="Squad name")
    id: Optional[int] = Field(None, description="Database ID")
    score: int = Field(..., description="Match score from fuzzy search")


class LeagueInfo(BaseModel):
    """League information with player count."""
    league: str
    player_count: int
