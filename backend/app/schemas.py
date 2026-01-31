"""
schemas.py - Pydantic schemas for API request/response validation.

Provides type-safe serialization for FastAPI endpoints.
"""

from pydantic import BaseModel
from datetime import datetime
from typing import Optional


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


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    database: str
    player_count: int
