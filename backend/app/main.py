"""
main.py - FastAPI application for Scouting Dashboard.

Provides REST API endpoints:
- GET /health: System status check
- GET /players/: Paginated player list with dynamic filters
- GET /players/search: Autocomplete player search
- GET /players/{id}: Single player by ID
- GET /leagues/: List all leagues
- POST /analysis/similarity: Server-side similarity search
- POST /analysis/attribution: Feature attribution between players

Key Features:
- Stateful SimilarityEngine loaded once at startup (lifespan handler)
- CORS middleware for cross-origin requests (Streamlit on 8501, API on 8000)
- SQL JSON extraction helpers for stats-based filtering
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import logging

from . import models, schemas
from .database import engine, get_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# STATEFUL ENGINE STORAGE (Loaded once at startup)
# =============================================================================

class AppState:
    """Global application state for stateful resources."""
    engine: Any = None  # SimilarityEngine instance
    df: pd.DataFrame = None  # Full player dataframe
    scaled_features: np.ndarray = None  # Scaled feature matrix
    scalers: Dict = None  # Fitted scalers
    is_loaded: bool = False


app_state = AppState()


# =============================================================================
# STAT KEY MAPPING (UI Display Name -> JSON Key)
# =============================================================================

# Maps frontend-friendly names to exact JSON keys in stats column
STAT_KEY_MAP = {
    # UI Name -> JSON Key
    "Goals per 90": "Gls/90",
    "Assists per 90": "Ast/90",
    "Shots per 90": "Sh/90",
    "Shots on Target per 90": "SoT/90",
    "Crosses per 90": "Crs/90",
    "Interceptions per 90": "Int/90",
    "Tackles Won per 90": "TklW/90",
    "Fouls per 90": "Fls/90",
    "Fouls Drawn per 90": "Fld/90",
    "xG per 90": "xG90",
    "xA per 90": "xAG90",
    "xG Chain per 90": "xGChain90",
    "xG Buildup per 90": "xGBuildup90",
    # Direct keys (for programmatic use)
    "Gls/90": "Gls/90",
    "Ast/90": "Ast/90",
    "Sh/90": "Sh/90",
    "SoT/90": "SoT/90",
    "Crs/90": "Crs/90",
    "Int/90": "Int/90",
    "TklW/90": "TklW/90",
    "Fls/90": "Fls/90",
    "Fld/90": "Fld/90",
    "xG90": "xG90",
    "xAG90": "xAG90",
    "xGChain90": "xGChain90",
    "xGBuildup90": "xGBuildup90",
    # GK Stats
    "GA90": "GA90",
    "Save%": "Save%",
    "CS%": "CS%",
}


def get_stat_json_key(display_name: str) -> str:
    """
    Convert UI display name to exact JSON key for database queries.
    
    Args:
        display_name: Frontend-friendly stat name (e.g., "Goals per 90")
        
    Returns:
        JSON key as stored in database (e.g., "Gls/90")
        
    Raises:
        KeyError: If display_name is not recognized
    """
    if display_name in STAT_KEY_MAP:
        return STAT_KEY_MAP[display_name]
    raise KeyError(f"Unknown stat name: '{display_name}'. Valid options: {list(STAT_KEY_MAP.keys())}")


# =============================================================================
# LIFESPAN HANDLER (Load engine once at startup)
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler for startup/shutdown events.
    
    On startup:
    - Creates database tables
    - Loads player data from database into pandas DataFrame
    - Initializes SimilarityEngine (expensive operation - done once)
    
    On shutdown:
    - Cleanup (if needed)
    """
    # === STARTUP ===
    logger.info("🚀 Starting Scouting Dashboard API...")
    
    # Create tables
    models.Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created/verified")
    
    # Load data into memory for similarity engine
    try:
        # Import utils (these are in the parent project)
        import sys
        import os
        
        # Add project root to path for utils import
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        from utils import process_all_data, SimilarityEngine
        
        logger.info("📊 Loading player data for similarity engine...")
        
        # Load and process data (this is the expensive operation)
        result = process_all_data('english_football_pyramid_master.csv', min_90s=0.5)
        
        app_state.df = result['dataframe']
        app_state.scaled_features = result['scaled_features']
        app_state.scalers = result['scalers']
        
        # Initialize similarity engine
        app_state.engine = SimilarityEngine(
            app_state.df,
            app_state.scaled_features,
            app_state.scalers
        )
        
        app_state.is_loaded = True
        logger.info(f"✅ SimilarityEngine loaded with {len(app_state.df)} players")
        
    except Exception as e:
        logger.error(f"❌ Failed to load SimilarityEngine: {e}")
        logger.warning("⚠️ API will run without similarity features")
        app_state.is_loaded = False
    
    yield  # Application runs here
    
    # === SHUTDOWN ===
    logger.info("👋 Shutting down Scouting Dashboard API...")
    app_state.engine = None
    app_state.df = None
    app_state.scaled_features = None


# =============================================================================
# FASTAPI APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Scouting Dashboard API",
    description="REST API for accessing player statistics, similarity search, and scouting data",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware (required for Streamlit on port 8501 to call API on port 8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit default
        "http://127.0.0.1:8501",
        "http://localhost:3000",  # Dev server
        "*",  # Allow all for development (restrict in production)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# HEALTH CHECK
# =============================================================================

@app.get("/health", response_model=schemas.HealthResponse)
def health_check(db: Session = Depends(get_db)):
    """
    System health check endpoint.
    
    Returns:
    - status: "healthy" if API is running
    - database: "connected" if SQLite is accessible
    - player_count: Total number of players in database
    - engine_loaded: Whether SimilarityEngine is ready
    """
    try:
        player_count = db.query(models.Player).count()
        return {
            "status": "healthy",
            "database": "connected",
            "player_count": player_count,
            "engine_loaded": app_state.is_loaded
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


# =============================================================================
# PLAYER ENDPOINTS
# =============================================================================

@app.get("/players/", response_model=List[schemas.PlayerOut])
def get_players(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Max records to return"),
    league: Optional[str] = Query(None, description="Filter by league name"),
    position: Optional[str] = Query(None, description="Filter by position (FW, MF, DF, GK)"),
    search: Optional[str] = Query(None, description="Search by player name"),
    min_age: Optional[int] = Query(None, ge=15, le=50, description="Minimum age"),
    max_age: Optional[int] = Query(None, ge=15, le=50, description="Maximum age"),
    min_goals: Optional[float] = Query(None, ge=0, description="Minimum Gls/90"),
    min_assists: Optional[float] = Query(None, ge=0, description="Minimum Ast/90"),
    archetype: Optional[str] = Query(None, description="Filter by archetype"),
    sort_by: Optional[str] = Query(None, description="Sort by stat (e.g., 'Gls/90')"),
    sort_desc: bool = Query(True, description="Sort descending (default: True)"),
    db: Session = Depends(get_db)
):
    """
    Retrieve players with pagination and dynamic filters.
    
    Supports filtering by:
    - Core fields: league, position, age range, name search
    - Stats (JSON): min_goals, min_assists, archetype
    - Sorting by any stat field
    
    Returns:
        List of PlayerOut objects with full stats JSON
    """
    query = db.query(models.Player)
    
    # === Core Field Filters ===
    if league:
        query = query.filter(models.Player.league == league)
    
    if position:
        query = query.filter(models.Player.position == position)
    
    if search:
        query = query.filter(models.Player.name.ilike(f"%{search}%"))
    
    if min_age is not None:
        query = query.filter(models.Player.age >= min_age)
    
    if max_age is not None:
        query = query.filter(models.Player.age <= max_age)
    
    # === JSON Stats Filters (using SQLite json_extract) ===
    if min_goals is not None:
        # SQLite: json_extract(stats, '$."Gls/90"') >= value
        query = query.filter(
            func.json_extract(models.Player.stats, '$."Gls/90"') >= min_goals
        )
    
    if min_assists is not None:
        query = query.filter(
            func.json_extract(models.Player.stats, '$."Ast/90"') >= min_assists
        )
    
    if archetype:
        query = query.filter(
            func.json_extract(models.Player.stats, '$.Archetype') == archetype
        )
    
    # === Sorting ===
    if sort_by:
        try:
            # Resolve stat key (handles both UI names and direct keys)
            json_key = get_stat_json_key(sort_by) if sort_by in STAT_KEY_MAP else sort_by
            sort_expr = func.json_extract(models.Player.stats, f'$."{json_key}"')
            
            if sort_desc:
                query = query.order_by(sort_expr.desc().nullslast())
            else:
                query = query.order_by(sort_expr.asc().nullsfirst())
        except KeyError:
            # Invalid sort key - just ignore
            pass
    
    # Apply pagination
    players = query.offset(skip).limit(limit).all()
    
    return players


@app.get("/players/search")
def search_players(
    query: str = Query(..., min_length=1, description="Search query"),
    league: Optional[str] = Query(None, description="Filter by league"),
    limit: int = Query(10, ge=1, le=50, description="Max results"),
    db: Session = Depends(get_db)
):
    """
    Autocomplete search for player names.
    
    Uses the in-memory SimilarityEngine for fuzzy matching if available,
    otherwise falls back to SQL LIKE search.
    
    Returns:
        List of {"name": str, "squad": str, "id": int, "score": int}
    """
    if app_state.is_loaded and app_state.engine is not None:
        # Use SimilarityEngine's fuzzy search
        suggestions = app_state.engine.get_player_suggestions(
            query,
            league=league if league else 'all',
            limit=limit
        )
        
        # Convert to API format with database IDs
        results = []
        for display_name, score in suggestions:
            # Parse "Player (Squad)" format
            if " (" in display_name and display_name.endswith(")"):
                name_part = display_name.split(" (")[0]
                squad_part = display_name.split(" (")[1].replace(")", "")
                
                # Find in database
                player = db.query(models.Player).filter(
                    models.Player.name.ilike(f"%{name_part}%"),
                    models.Player.squad.ilike(f"%{squad_part}%")
                ).first()
                
                if player:
                    results.append({
                        "name": display_name,
                        "squad": player.squad,
                        "id": player.id,
                        "score": score
                    })
            else:
                results.append({
                    "name": display_name,
                    "squad": "",
                    "id": None,
                    "score": score
                })
        
        return results
    else:
        # Fallback: SQL LIKE search
        players = db.query(models.Player).filter(
            models.Player.name.ilike(f"%{query}%")
        ).limit(limit).all()
        
        return [
            {
                "name": f"{p.name} ({p.squad})",
                "squad": p.squad,
                "id": p.id,
                "score": 100
            }
            for p in players
        ]


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
        List of {"league": str, "player_count": int}
    """
    results = db.query(
        models.Player.league,
        func.count(models.Player.id).label("player_count")
    ).group_by(models.Player.league).all()
    
    return [{"league": r[0], "player_count": r[1]} for r in results]


# =============================================================================
# SIMILARITY ANALYSIS ENDPOINTS
# =============================================================================

@app.post("/analysis/similarity", response_model=schemas.SimilarityResponse)
def find_similar_players(
    request: schemas.SimilarityRequest,
    db: Session = Depends(get_db)
):
    """
    Server-side similarity search.
    
    Offloads heavy cosine similarity calculation to the backend.
    Uses the pre-loaded SimilarityEngine for fast computation.
    
    Request Body:
    - player_id: Database ID of target player
    - league: Filter results by league ("all" for all leagues)
    - top_n: Number of similar players to return (default: 5)
    - use_position_weights: Apply position-specific weighting
    - scouting_priority: Priority profile ("Standard", "Goals", etc.)
    
    Returns:
        SimilarityResponse with target player and matches
    """
    if not app_state.is_loaded or app_state.engine is None:
        raise HTTPException(
            status_code=503,
            detail="SimilarityEngine not loaded. Server may still be starting."
        )
    
    # Get target player from database
    target_player = db.query(models.Player).filter(
        models.Player.id == request.player_id
    ).first()
    
    if not target_player:
        raise HTTPException(status_code=404, detail="Target player not found")
    
    # Find player in engine's DataFrame using name
    player_name = f"{target_player.name} ({target_player.squad})"
    
    try:
        similar_df = app_state.engine.find_similar_players(
            target_player=player_name,
            league=request.league if request.league != "all" else "all",
            top_n=request.top_n,
            use_position_weights=request.use_position_weights,
            scouting_priority=request.scouting_priority,
            target_league_tier=request.target_league_tier
        )
        
        if similar_df is None or len(similar_df) == 0:
            return schemas.SimilarityResponse(
                target=target_player,
                matches=[],
                proxy_warnings=None
            )
        
        # Convert DataFrame rows to SimilarMatch objects
        matches = []
        for _, row in similar_df.iterrows():
            # Look up database ID for each match
            match_player = db.query(models.Player).filter(
                models.Player.name == row['Player'],
                models.Player.squad == row['Squad']
            ).first()
            
            match_id = match_player.id if match_player else None
            
            matches.append(schemas.SimilarMatch(
                id=match_id,
                name=row['Player'],
                squad=row['Squad'],
                league=row['League'],
                position=row['Primary_Pos'],
                match_score=float(row['Match_Score']),
                primary_drivers=row.get('Primary_Drivers', ''),
                stats={
                    "Gls/90": float(row.get('Gls/90', 0)),
                    "Ast/90": float(row.get('Ast/90', 0)),
                    "Age": int(row.get('Age', 0)),
                }
            ))
        
        # Get proxy warnings if any
        proxy_warnings = similar_df.iloc[0].get('Proxy_Warnings', '') if len(similar_df) > 0 else None
        
        return schemas.SimilarityResponse(
            target=target_player,
            matches=matches,
            proxy_warnings=proxy_warnings if proxy_warnings else None
        )
        
    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Similarity search failed: {str(e)}"
        )


@app.post("/analysis/attribution")
def get_feature_attribution(
    request: schemas.AttributionRequest,
    db: Session = Depends(get_db)
):
    """
    Get detailed feature attribution between two players.
    
    Shows which stats drive similarity/difference between players.
    
    Returns:
        Dict with feature distances sorted by target's priorities
    """
    if not app_state.is_loaded or app_state.engine is None:
        raise HTTPException(
            status_code=503,
            detail="SimilarityEngine not loaded."
        )
    
    # Get both players
    player1 = db.query(models.Player).filter(
        models.Player.id == request.player1_id
    ).first()
    player2 = db.query(models.Player).filter(
        models.Player.id == request.player2_id
    ).first()
    
    if not player1:
        raise HTTPException(status_code=404, detail="Player 1 not found")
    if not player2:
        raise HTTPException(status_code=404, detail="Player 2 not found")
    
    # Get attribution
    player1_name = f"{player1.name} ({player1.squad})"
    player2_name = f"{player2.name} ({player2.squad})"
    
    try:
        attribution = app_state.engine.calculate_feature_attribution(
            target_player=player1_name,
            comparison_player=player2_name,
            use_position_weights=request.use_position_weights
        )
        
        if attribution is None:
            return {"attribution": {}, "error": "Could not calculate attribution"}
        
        return {"attribution": attribution}
        
    except Exception as e:
        logger.error(f"Attribution calculation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Attribution calculation failed: {str(e)}"
        )
