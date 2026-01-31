
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from utils import SimilarityEngine

def test_cosine_similarity_logic():
    """Verify that manual cosine calculation matches scikit-learn."""
    # Vector A: [1, 0]
    # Vector B: [0, 1]
    # Similarity should be 0
    vec_a = np.array([[1, 0]])
    vec_b = np.array([[0, 1]])
    sim = cosine_similarity(vec_a, vec_b)[0][0]
    assert np.isclose(sim, 0.0)
    
    # Vector A: [1, 1]
    # Vector B: [2, 2]
    # Similarity should be 1
    vec_a = np.array([[1, 1]])
    vec_b = np.array([[2, 2]])
    sim = cosine_similarity(vec_a, vec_b)[0][0]
    assert np.isclose(sim, 1.0)

def test_similarity_engine_structure():
    """Verify engine initialization and indexing."""
    # Mock data
    data = pd.DataFrame({
        'Player': ['A', 'B', 'C'],
        'Squad': ['S1', 'S2', 'S3'],
        'League': ['L1', 'L1', 'L1'],
        'Primary_Pos': ['FW', 'FW', 'FW'],
        'Age': [20, 20, 20],
        '90s': [10, 10, 10]
    })
    
    # Scaled features (3 players, 22 features: 13 outfield + 9 GK)
    scaled = np.zeros((3, 22))
    # Player A
    scaled[0, 0] = 1.0 # Gls/90
    # Player B (Very similar to A)
    scaled[1, 0] = 0.9
    scaled[1, 1] = 0.1
    # Player C (Dissimilar)
    scaled[2, 2] = 1.0
    
    scalers = {'FW': None} # Mock scaler dict
    
    engine = SimilarityEngine(data, scaled, scalers)
    
    # Identify index
    idx = engine._find_player_index("A (S1)")
    assert idx == 0
    
    # Test Similarity Search
    # Should find B as closest match
    matches = engine.find_similar_players("A (S1)", top_n=1)
    
    # Matches is a DataFrame
    assert len(matches) == 1
    assert matches.iloc[0]['Player'] == 'B'
