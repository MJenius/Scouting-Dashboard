"""
main.py - FastAPI application for Scouting Dashboard.

Provides REST API endpoints:
- GET /health: System status check
- GET /players/: Paginated player list with optional filters
"""

from fastapi import FastAPI, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional

from . import models, schemas
from .database import engine, get_db

# Create database tables on startup
models.Base.metadata.create_all(bind=engine)

# Initialize FastAPI app
app = FastAPI(
    title="Scouting Dashboard API",
    description="REST API for accessing player statistics and scouting data",
    version="1.0.0"
)


@app.get("/health", response_model=schemas.HealthResponse)
def health_check(db: Session = Depends(get_db)):
    """
    System health check endpoint.
    
    Returns:
    - status: "healthy" if API is running
    - database: "connected" if SQLite is accessible
    - player_count: Total number of players in database
    """
    try:
        player_count = db.query(models.Player).count()
        return {
            "status": "healthy",
            "database": "connected",
            "player_count": player_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/players/", response_model=List[schemas.PlayerOut])
def get_players(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    league: Optional[str] = Query(None, description="Filter by league name"),
    position: Optional[str] = Query(None, description="Filter by position (FW, MF, DF, GK)"),
    search: Optional[str] = Query(None, description="Search by player name"),
    db: Session = Depends(get_db)
):
    """
    Retrieve players with pagination and optional filters.
    
    Query Parameters:
    - skip: Offset for pagination (default: 0)
    - limit: Max results per page (default: 100, max: 1000)
    - league: Filter by league name (exact match)
    - position: Filter by position (FW, MF, DF, GK)
    - search: Search by player name (case-insensitive partial match)
    
    Returns:
        List of PlayerOut objects with full stats JSON
    """
    query = db.query(models.Player)
    
    # Apply filters
    if league:
        query = query.filter(models.Player.league == league)
    
    if position:
        query = query.filter(models.Player.position == position)
    
    if search:
        # Case-insensitive partial search using SQLite LIKE
        query = query.filter(models.Player.name.ilike(f"%{search}%"))
    
    # Apply pagination
    players = query.offset(skip).limit(limit).all()
    
    return players


@app.get("/players/{player_id}", response_model=schemas.PlayerOut)
def get_player(player_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a single player by ID.
    
    Args:
        player_id: Database ID of the player
        
    Returns:
        PlayerOut object with full stats JSON
        
    Raises:
        404: Player not found
    """
    player = db.query(models.Player).filter(models.Player.id == player_id).first()
    
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    return player


@app.get("/leagues/")
def get_leagues(db: Session = Depends(get_db)):
    """
    Get list of all leagues in the database.
    
    Returns:
        List of unique league names with player counts
    """
    from sqlalchemy import func
    
    results = db.query(
        models.Player.league,
        func.count(models.Player.id).label("player_count")
    ).group_by(models.Player.league).all()
    
    return [{"league": r[0], "player_count": r[1]} for r in results]
