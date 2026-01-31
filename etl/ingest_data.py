"""
ingest_data.py - ETL script to load CSV data into SQLite database.

This script bridges the existing data_engine.py processing pipeline 
to the new SQLite database backend.

Features:
- Imports process_all_data() from utils.data_engine
- Converts NumPy types to Python native types for JSON serialization
- Implements Upsert (Update or Insert) logic for idempotency
- Packs all statistical columns into the stats JSON field
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from backend.app.database import engine, SessionLocal
from backend.app.models import Base, Player
from utils.data_engine import process_all_data
from utils.constants import FEATURE_COLUMNS, GK_FEATURE_COLUMNS


def convert_numpy_types(obj):
    """
    Recursively convert NumPy types to Python native types.
    
    Required because SQLAlchemy JSON column cannot serialize np.int64, np.float64, etc.
    
    Args:
        obj: Any Python object that may contain NumPy types
        
    Returns:
        Object with all NumPy types converted to native Python types
    """
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle NaN and Inf values
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.str_, np.bytes_)):
        return str(obj)
    else:
        return obj


def extract_stats(row, columns):
    """
    Extract statistical columns from a DataFrame row into a dictionary.
    
    Args:
        row: pandas Series (single row from DataFrame)
        columns: List of column names to extract
        
    Returns:
        Dictionary of {column_name: value} with NumPy types converted
    """
    stats = {}
    
    for col in columns:
        if col in row.index:
            value = row[col]
            # Skip NaN values
            if isinstance(value, float) and np.isnan(value):
                continue
            stats[col] = convert_numpy_types(value)
    
    # Also include percentile columns
    pct_columns = [col for col in row.index if '_pct' in col]
    for col in pct_columns:
        value = row[col]
        if isinstance(value, float) and np.isnan(value):
            continue
        stats[col] = convert_numpy_types(value)
    
    # Include derived metrics (efficiency, z-scores, PCA)
    derived_cols = [
        'Finishing_Efficiency', 'Creative_Efficiency', 
        'Age_Z_Score_GA90', 'G_plus_A',
        'PCA_X', 'PCA_Y', 'Completeness_Score', 'Completeness_Core',
        '90s', 'Min'
    ]
    for col in derived_cols:
        if col in row.index:
            value = row[col]
            if isinstance(value, float) and np.isnan(value):
                continue
            stats[col] = convert_numpy_types(value)
    
    return stats


def upsert_players(session, players_data):
    """
    Perform upsert (update or insert) for a batch of players.
    
    Uses SQLite's ON CONFLICT DO UPDATE for idempotency.
    Unique constraint is on (name, squad, league).
    
    Args:
        session: SQLAlchemy session
        players_data: List of dictionaries with player data
    """
    if not players_data:
        return
    
    for player_data in players_data:
        # Create insert statement
        stmt = sqlite_insert(Player).values(**player_data)
        
        # On conflict (name, squad, league), update all fields
        stmt = stmt.on_conflict_do_update(
            index_elements=['name', 'squad', 'league'],
            set_={
                'position': stmt.excluded.position,
                'age': stmt.excluded.age,
                'nation': stmt.excluded.nation,
                'stats': stmt.excluded.stats,
                'updated_at': datetime.utcnow()
            }
        )
        
        session.execute(stmt)


def run_ingestion():
    """
    Main ETL function: Load CSV → Process → Ingest to SQLite.
    
    Steps:
    1. Create database tables if they don't exist
    2. Call process_all_data() from data_engine
    3. Extract fixed fields and pack stats as JSON
    4. Upsert into SQLite with batch commits
    """
    print("=" * 80)
    print("🚀 STARTING ETL INGESTION")
    print("=" * 80)
    
    # Step 1: Create database tables
    print("\n[1/4] Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Tables ready")
    
    # Step 2: Load and process data using existing data_engine
    print("\n[2/4] Processing CSV data...")
    csv_path = os.path.join(project_root, "english_football_pyramid_master.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: CSV file not found at {csv_path}")
        return
    
    result = process_all_data(csv_path)
    df = result['dataframe']
    print(f"✓ Processed {len(df)} players from {df['League'].nunique()} leagues")
    
    # Step 3: Transform and prepare for database insertion
    print("\n[3/4] Transforming data for database...")
    
    session = SessionLocal()
    batch_size = 500
    players_batch = []
    total_processed = 0
    
    try:
        for idx, row in df.iterrows():
            # Determine which feature columns to use based on position
            position = row.get('Primary_Pos', None)
            
            if position == 'GK':
                stat_cols = GK_FEATURE_COLUMNS
            else:
                stat_cols = FEATURE_COLUMNS
            
            # Extract stats as JSON
            stats = extract_stats(row, stat_cols)
            
            # Prepare player data
            player_data = {
                'name': str(row.get('Player', '')).strip(),
                'squad': str(row.get('Squad', '')).strip(),
                'league': str(row.get('League', '')).strip(),
                'position': str(position) if position else None,
                'age': int(row['Age']) if not np.isnan(row.get('Age', np.nan)) else None,
                'nation': str(row.get('Nation', '')).strip() if row.get('Nation') else None,
                'stats': stats
            }
            
            players_batch.append(player_data)
            
            # Batch commit for performance
            if len(players_batch) >= batch_size:
                upsert_players(session, players_batch)
                session.commit()
                total_processed += len(players_batch)
                print(f"  → Processed {total_processed} players...")
                players_batch = []
        
        # Final batch
        if players_batch:
            upsert_players(session, players_batch)
            session.commit()
            total_processed += len(players_batch)
        
        print(f"✓ Transformed {total_processed} players")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error during ingestion: {e}")
        raise
    finally:
        session.close()
    
    # Step 4: Summary
    print("\n[4/4] Verifying ingestion...")
    session = SessionLocal()
    try:
        player_count = session.query(Player).count()
        league_counts = session.query(Player.league).distinct().count()
        print(f"✓ Database contains {player_count} players across {league_counts} leagues")
    finally:
        session.close()
    
    print("\n" + "=" * 80)
    print(f"✅ Ingested {total_processed} players into SQLite.")
    print("=" * 80)


if __name__ == "__main__":
    run_ingestion()
