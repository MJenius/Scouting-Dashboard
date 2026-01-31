"""
models.py - SQLAlchemy ORM models for Scouting Dashboard.

Implements a Hybrid Schema approach:
- Fixed columns for core player identity (name, squad, league, position, age)
- Flexible JSON column for all statistical metrics (handles league tier variance)

This design allows:
- Premier League players to have xG, xA, xGChain, xGBuildup
- National League players to have only Gls/90, Ast/90
- No null-column overhead or future schema migrations
"""

from sqlalchemy import Column, Integer, String, DateTime, Index, JSON, UniqueConstraint
from sqlalchemy.sql import func
from .database import Base


class Player(Base):
    """
    Player entity with hybrid schema for cross-league compatibility.
    
    The `stats` JSON field stores all per-90 metrics, percentiles, and 
    derived features. This approach handles schema variance between leagues
    (e.g., xG exists in Premier League but not in National League).
    
    Indexes:
    - name: For player search
    - league: For league filtering
    - squad: For team filtering
    - Composite (name, league): For cross-league player search
    
    Unique Constraint:
    - (name, squad, league): Ensures no duplicate players per team/league
    """
    __tablename__ = "players"
    
    # Primary Key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Core Identity Fields
    name = Column(String(255), nullable=False, index=True)
    squad = Column(String(255), nullable=False, index=True)
    league = Column(String(100), nullable=False, index=True)
    position = Column(String(50), nullable=True)  # Primary position: FW, MF, DF, GK
    age = Column(Integer, nullable=True)
    nation = Column(String(100), nullable=True)
    
    # Hybrid Schema: All statistics stored as JSON
    # Example: {"Gls/90": 0.5, "xG90": 0.4, "Gls/90_pct": 85.0, ...}
    stats = Column(JSON, nullable=True, default=dict)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Composite index for cross-league player search
    __table_args__ = (
        Index('ix_player_name_league', 'name', 'league'),
        UniqueConstraint('name', 'squad', 'league', name='uq_player_identity'),
    )
    
    def __repr__(self):
        return f"<Player(id={self.id}, name='{self.name}', squad='{self.squad}', league='{self.league}')>"
