
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.database import Base, get_db
from backend.app.main import app, app_state
from backend.app import models

# In-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)

TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def db_session():
    """Create a fresh in-memory database for each test."""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.fixture(scope="function")
def client(db_session):
    """TestClient with database dependency override."""
    def override_get_db():
        try:
            yield db_session
        finally:
            db_session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture(autouse=True)
def mock_similarity_engine(monkeypatch):
    """Mock the heavy SimilarityEngine logic."""
    class MockEngine:
        def find_similar_players(self, target_player, league, top_n, use_position_weights, scouting_priority, target_league_tier):
             # Return a minimal pandas DataFrame-like structure or list of dicts if that's what the endpoint expects
             import pandas as pd
             return pd.DataFrame([{
                 'Player': 'Sim Player',
                 'Squad': 'Sim Squad',
                 'League': 'Sim League',
                 'Primary_Pos': 'FW',
                 'Match_Score': 95.5,
                 'Primary_Drivers': 'Gls/90',
                 'Gls/90': 0.8,
                 'Ast/90': 0.2,
                 'Age': 24,
                 'Proxy_Warnings': None
             }])

        def calculate_feature_attribution(self, target_player, comparison_player, use_position_weights):
            return {"Gls/90": 0.1, "Ast/90": -0.05}
    
    # Mock the app_state to appear loaded
    app_state.is_loaded = True
    app_state.engine = MockEngine()
    yield

@pytest.fixture
def valid_player_payload(db_session):
    """Insert a valid player into the test database."""
    stats = {
        "Gls/90": 0.5,
        "Ast/90": 0.3,
        "Archetype": "Finisher",
        "90s": 10.0
    }
    player = models.Player(
        name="Test Player",
        squad="Test Squad",
        league="Test League",
        position="FW",
        age=25,
        stats=stats
    )
    db_session.add(player)
    db_session.commit()
    db_session.refresh(player)
    return player
