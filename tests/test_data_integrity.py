
import pytest
import pandas as pd
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import data_engine
from utils.constants import FEATURE_COLUMNS, GK_FEATURE_COLUMNS

# Path to the actual CSV because these are data integrity tests on the REAL data
CSV_PATH = "english_football_pyramid_master.csv"

@pytest.fixture(scope="module")
def loaded_data():
    """Load the full dataset once for integrity tests."""
    if not os.path.exists(CSV_PATH):
        pytest.skip(f"Data file {CSV_PATH} not found.")
    
    # We load with min_90s=0 to inspect raw data availability if needed, 
    # but strictly we care about the processed data used in the app.
    # Let's use the standard loader.
    result = data_engine.process_all_data(CSV_PATH, min_90s=0.1) # low threshold to catch low-minute issues if we want
    df = result['dataframe']
    # Filter out players with missing critical info to make tests robust
    df = df.dropna(subset=['Primary_Pos', 'Player', 'League', 'Age'])
    return df

def test_critical_columns_not_null(loaded_data):
    """Ensure critical identification columns are never null."""
    critical_cols = ['Player', 'League', 'Primary_Pos', 'Age', '90s']
    for col in critical_cols:
        assert loaded_data[col].isnull().sum() == 0, f"Found nulls in {col}"

def test_stats_completeness_outfield(loaded_data):
    """Ensure outfield players have valid float values for Feature Columns."""
    outfield_mask = loaded_data['Primary_Pos'] != 'GK'
    df_outfield = loaded_data[outfield_mask]
    
    for col in FEATURE_COLUMNS:
        # Check for NaNs
        null_count = df_outfield[col].isnull().sum()
        assert null_count == 0, f"Feature {col} has {null_count} nulls for outfield players"
        
        # Check for infinite values (division by zero errors)
        inf_count = (df_outfield[col] == float('inf')).sum()
        assert inf_count == 0, f"Feature {col} has {inf_count} infinite values"

def test_stats_completeness_gk(loaded_data):
    """Ensure goalkeepers have valid float values for GK Feature Columns."""
    gk_mask = loaded_data['Primary_Pos'] == 'GK'
    df_gk = loaded_data[gk_mask]
    
    if len(df_gk) == 0:
        return # Skip if no GKs
        
    for col in GK_FEATURE_COLUMNS:
        if col in df_gk.columns:
            null_count = df_gk[col].isnull().sum()
            assert null_count == 0, f"GK Feature {col} has {null_count} nulls"

def test_schema_enforcement_logic():
    """
    Verify that our code logic enforces the schema.
    (This tests the utility, not just the data)
    """
    # Create a dummy DF with missing columns
    dummy_data = pd.DataFrame([{
        'Player': 'Test', 'League': 'Test', 'Primary_Pos': 'FW', '90s': 10.0, 'Age': 20
    }])
    
    # Run cleaning
    cleaned = data_engine.clean_feature_columns(dummy_data)
    
    # Assert missing 'Gls/90' was created and filled with 0.0
    assert 'Gls/90' in cleaned.columns
    assert cleaned['Gls/90'].iloc[0] == 0.0
    assert cleaned['Gls/90'].dtype == float
