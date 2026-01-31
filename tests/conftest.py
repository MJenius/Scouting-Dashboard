
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
import utils
import utils.data_engine
import pandas as pd
import numpy as np

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
    """Mock the heavy SimilarityEngine logic and data processing."""
    
    # 1. Mock process_all_data so lifespan doesn't load real CSVs
    # We return dummy data that looks enough like the real result to not crash
    def mock_process_all_data(*args, **kwargs):
        return {
            'dataframe': pd.DataFrame(),
            'scaled_features': np.zeros((0, 22)),
            'scalers': {}
        }
    monkeypatch.setattr(utils.data_engine, "process_all_data", mock_process_all_data)

    class MockEngine:
        def __init__(self, *args, **kwargs):
            pass
        
        def find_similar_players(self, target_player, league='all', top_n=5, use_position_weights=True, scouting_priority='Standard', target_league_tier=None):
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

        def calculate_feature_attribution(self, target_player, comparison_player, use_position_weights=True):
            return {"Gls/90": 0.1, "Ast/90": -0.05}
    
    # Mock the class in utils
    monkeypatch.setattr(utils, "SimilarityEngine", MockEngine)
    
    # Also set it on app_state directly to be safe
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
    
    # Add the simulated player that the mock engine returns
    sim_player = models.Player(
        name="Sim Player",
        squad="Sim Squad",
        league="Sim League",
        position="FW",
        age=24,
        stats={"Gls/90": 0.8, "Ast/90": 0.2}
    )
    db_session.add(sim_player)
    
    db_session.commit()
    db_session.refresh(player)
    return player
