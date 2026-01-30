"""
data_engine.py - Core data loading, processing, and feature engineering pipeline.

This module provides the foundation for the Scouting Dashboard:
- Load and validate the master dataset (english_football_pyramid_master.csv)
- Type conversion and cleaning for per-90 metrics
- League + Position-specific percentile ranking with quality flags
- Data completeness scoring (league-aware)
- Feature scaling for similarity calculations
- Data validation and error reporting

Main entry point: process_all_data(csv_path) - orchestrates all processing steps
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import warnings

from .constants import (
    FEATURE_COLUMNS,
    GK_FEATURE_COLUMNS,
    OFFENSIVE_FEATURES,
    POSSESSION_FEATURES,
    LEAGUES,
    PRIMARY_POSITIONS,
    LEAGUE_METRIC_MAP,
    MIN_MINUTES_PLAYED,
    MIN_PLAYERS_PER_GROUP,
    PERCENTILE_QUALITY_THRESHOLDS,
    LOW_DATA_LEAGUES,
)


def clean_player_name(name: any) -> str:
    """
    Remove quotes, extra spaces, and handle NaN for merging consistency.
    """
    if pd.isna(name):
        return ""
    return str(name).replace('"', '').replace("'", "").strip()


# ============================================================================
# DATA LOADING & CLEANING
# ============================================================================

def load_data(file_path: str, low_memory: bool = False) -> pd.DataFrame:
    """
    Load the master dataset and merge advanced metrics from individual files.
    """
    try:
        df = pd.read_csv(file_path, low_memory=low_memory)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    
    # Clean player names for consistent merging
    df['Player'] = df['Player'].apply(clean_player_name)
    
    # Extract primary position (first position if comma-separated)
    if 'Pos' in df.columns:
        df['Primary_Pos'] = df['Pos'].str.split(',').str[0].str.strip()
    
    # Ensure numerical types for critical columns
    for col in ['90s', 'Age']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Initialize advanced metrics if not present
    advanced_metrics = ['xG90', 'xA90', 'xGChain90', 'xGBuildup90']
    for col in advanced_metrics:
        if col not in df.columns:
            df[col] = 0.0

    # Load and merge advanced files
    advanced_files = {
        'Premier League': 'data/Premier League Advanced Stats.csv',
        'Bundesliga': 'data/Bundesliga Advanced Stats.csv',
        'La Liga': 'data/La Liga Advanced Stats.csv',
        'Serie A': 'data/Serie A Advanced Stats.csv',
        'Ligue 1': 'data/Ligue 1 Advanced Stats.csv'
    }

    for league, adv_path in advanced_files.items():
        try:
            # Use ; as delimiter as requested
            adv_df = pd.read_csv(adv_path, sep=';', low_memory=False)
            
            # Normalize column names to match main dataframe (e.g., 'player' -> 'Player')
            # The CSVs from this source often use lowercase or specific headers
            column_map = {
                'player': 'Player',
                'team': 'Squad',
                'goals': 'Gls',
                'a': 'Ast',
                'min': 'Min',
            }
            # Add any other case-insensitive mappings if needed
            adv_df = adv_df.rename(columns={k: v for k, v in column_map.items() if k in adv_df.columns})
            
            # If 'Player' still missing, look for it case-insensitively
            if 'Player' not in adv_df.columns:
                actual_cols = {c.lower(): c for c in adv_df.columns}
                if 'player' in actual_cols:
                    adv_df = adv_df.rename(columns={actual_cols['player']: 'Player'})
            
            if 'Player' not in adv_df.columns:
                print(f"⚠️  Could not find 'Player' column in {adv_path}. Skipping.")
                continue

            # Clean player names for consistent merging
            adv_df['Player'] = adv_df['Player'].apply(clean_player_name)
            
            # Identify players from this league in the main dataframe
            league_mask = df['League'] == league
            league_players = df[league_mask].copy()
            
            # Sub-select only needed columns to avoid conflicts
            # Decision (A): Only merge the specific advanced metrics we need
            merge_cols = ['Player'] + [c for c in advanced_metrics if c in adv_df.columns]
            adv_subset = adv_df[merge_cols].drop_duplicates(subset=['Player'])
            
            # Remove existing advanced columns from league_players to avoid suffixes
            league_players = league_players.drop(columns=[c for c in advanced_metrics if c in league_players.columns])
            
            # Merge
            merged = pd.merge(league_players, adv_subset, on='Player', how='left')
            
            # Fill missing values with 0
            for col in advanced_metrics:
                if col in merged.columns:
                    merged[col] = merged[col].fillna(0.0)
                else:
                    merged[col] = 0.0
            
            # Update the main dataframe
            df = pd.concat([df[~league_mask], merged], axis=0, ignore_index=True)
            print(f"📊 Merged advanced stats for {league} ({len(merged)} players)")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not merge advanced stats for {league} ({adv_path}): {e}")

    # Final validation of required columns
    required_columns = ['Player', 'Squad', 'League', 'Pos', 'Age', '90s'] + FEATURE_COLUMNS
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        # Some columns might still be missing if not in FEATURE_COLUMNS yet or not loaded
        # We'll just warn instead of failing if it's not critical
        print(f"⚠️  Missing columns in merged dataset: {missing_columns}")
    
    return df


def clean_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce all feature columns to float and handle missing values.
    
    Strategy:
    - Coerce to float (invalid values → NaN)
    - Fill NaN with 0 ONLY for features (zeros are valid for "no goal", "no assist")
    - Keep NaN meaningful where it indicates "not tracked" (National League defensive stats)
    
    Args:
        df (pd.DataFrame): Dataset with feature columns
        
    Returns:
        pd.DataFrame: Cleaned dataset with float features
    """
    df = df.copy()
    
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Fill NaN with 0 for missing stats (safe for per-90 metrics)
            df[col] = df[col].fillna(0)
    
    return df


# ============================================================================
# FILTERING & VALIDATION
# ============================================================================

def filter_by_minutes_played(df: pd.DataFrame, min_90s: int = MIN_MINUTES_PLAYED) -> pd.DataFrame:
    """
    Filter players to ensure statistical reliability.
    
    Rationale: Players with <10 90s are unreliable for percentile comparisons
    because small sample sizes produce volatile per-90 metrics.
    
    Args:
        df (pd.DataFrame): Raw dataset
        min_90s (int): Minimum 90-minute equivalents for inclusion (default: 10)
        
    Returns:
        pd.DataFrame: Filtered dataset
    """
    initial_count = len(df)
    df_filtered = df[df['90s'] >= min_90s].copy()
    filtered_count = len(df_filtered)
    
    removed_count = initial_count - filtered_count
    if removed_count > 0:
        print(f"ℹ️  Filtered {removed_count} players with <{min_90s} 90s matches. "
              f"Retained {filtered_count} players.")
    
    return df_filtered


def validate_data(df: pd.DataFrame) -> Dict[str, any]:
    """
    Comprehensive data validation and quality checks.
    
    Returns:
        Dict with validation results and warnings
    """
    validation_report = {
        'total_players': len(df),
        'total_leagues': df['League'].nunique(),
        'total_positions': df['Primary_Pos'].nunique(),
        'warnings': [],
        'league_stats': {},
    }
    
    # Per-league quality check
    for league in df['League'].unique():
        league_df = df[df['League'] == league]
        player_count = len(league_df)
        
        stats = {
            'player_count': player_count,
            'avg_90s': league_df['90s'].mean(),
            'avg_age': league_df['Age'].mean(),
        }
        
        # Check for missing defensive data (Low Data Leagues)
        if league in LOW_DATA_LEAGUES:
            defensive_metrics = ['Crs/90', 'Int/90', 'TklW/90', 'Fls/90', 'Fld/90']
            null_counts = {
                metric: league_df[metric].isna().sum() 
                for metric in defensive_metrics
            }
            stats['null_defensive_stats'] = null_counts
            validation_report['warnings'].append(
                f"⚠️  {league}: Defensive stats may be unavailable in this data tier (as expected)"
            )
        
        validation_report['league_stats'][league] = stats
    
    # Check for position-league combinations with too few players
    for league in df['League'].unique():
        for pos in df['Primary_Pos'].unique():
            if pd.isna(pos):
                continue
            group_size = len(df[(df['League'] == league) & (df['Primary_Pos'] == pos)])
            if group_size < MIN_PLAYERS_PER_GROUP:
                validation_report['warnings'].append(
                    f"⚠️  Low sample: {league} {pos} has only {group_size} players"
                )
    
    return validation_report


# ============================================================================
# PERCENTILE RANKING
# ============================================================================

def calculate_position_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate League + Position-specific percentiles for all features.
    
    Strategy:
    - Group players by (League, Primary_Pos)
    - Rank each feature within group using percentile rank (0-100)
    - Add "Percentile_Quality" flag (High/Medium/Low) based on group size
    - Skip groups with <5 players (percentiles unreliable)
    
    Output columns:
    - {feature}_pct: Percentile rank within (League, Position) group
    - {feature}_pct_quality: Quality flag (High/Medium/Low)
    
    Args:
        df (pd.DataFrame): Cleaned dataset with features
        
    Returns:
        pd.DataFrame: Dataset with percentile columns added
    """
    df = df.copy()
    
    leagues = df['League'].dropna().unique()
    positions = df['Primary_Pos'].dropna().unique()
    
    for league in leagues:
        for pos in positions:
            # Define position-league group
            mask = (df['League'] == league) & (df['Primary_Pos'] == pos)
            group_size = mask.sum()
            
            # Skip small groups (percentiles unreliable)
            if group_size < MIN_PLAYERS_PER_GROUP:
                continue
            
            # Determine quality flag
            if group_size >= PERCENTILE_QUALITY_THRESHOLDS['High']:
                quality = 'High'
            elif group_size >= PERCENTILE_QUALITY_THRESHOLDS['Medium']:
                quality = 'Medium'
            else:
                quality = 'Low'
            
            # Calculate percentile for each feature within group
            for feature in FEATURE_COLUMNS:
                pct_col = f'{feature}_pct'
                quality_col = f'{feature}_pct_quality'
                
                # Rank within group: converts to 0-100 percentile
                # na_option='bottom' places NaN values at 0th percentile
                df.loc[mask, pct_col] = (
                    df.loc[mask, feature].rank(pct=True, na_option='bottom') * 100
                )
                df.loc[mask, quality_col] = quality
            
            # Also calculate percentiles for goalkeeper-specific metrics
            if pos == 'GK':
                for feature in GK_FEATURE_COLUMNS:
                    # Skip if column doesn't exist
                    if feature not in df.columns:
                        continue
                    
                    pct_col = f'{feature}_pct'
                    quality_col = f'{feature}_pct_quality'
                    
                    # For goalkeeper metrics, we need to handle direction properly
                    # Lower is better for GA90 and L (losses), higher is better for others
                    if feature in ['GA90', 'L']:
                        # Lower is better - invert the ranking
                        df.loc[mask, pct_col] = (
                            100 - (df.loc[mask, feature].rank(pct=True, na_option='bottom') * 100)
                        )
                    else:
                        # Higher is better
                        df.loc[mask, pct_col] = (
                            df.loc[mask, feature].rank(pct=True, na_option='bottom') * 100
                        )
                    df.loc[mask, quality_col] = quality
    
    return df


# ============================================================================
# DATA COMPLETENESS SCORING
# ============================================================================

def calculate_data_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """
    League-aware completeness scoring.
    
    Decision (B): Adjust expected metric set based on league data availability.
    - Premier League, Championship, League One, League Two: Score on all 9 features
    - National League: Score ONLY on Gls/90, Ast/90 (defensive stats not tracked)
    
    Returns two scores:
    - Completeness_Score: Main score (league-aware)
    - Completeness_Core: Always based on Gls/90 and Ast/90 only
    
    Args:
        df (pd.DataFrame): Dataset with feature columns
        
    Returns:
        pd.DataFrame: Dataset with completeness scores
    """
    df = df.copy()
    
    def calculate_score(row: pd.Series, score_type: str = 'main') -> float:
        """
        Calculate completeness score for a single player.
        
        Args:
            row: Player row
            score_type: 'main' (league-aware) or 'core' (always Gls/Ast only)
            
        Returns:
            float: Score 0-100
        """
        if score_type == 'core':
            # Always use only attacking metrics
            expected_metrics = ['Gls/90', 'Ast/90']
        else:
            # Main score: use league-specific metric set
            league = row['League']
            expected_metrics = LEAGUE_METRIC_MAP.get(league, FEATURE_COLUMNS)
        
        # Count non-null AND non-zero values (0 is valid data)
        found = sum(
            1 for metric in expected_metrics 
            if metric in row.index and pd.notna(row[metric]) and row[metric] > 0
        )
        
        completeness = (found / len(expected_metrics)) * 100
        return completeness
    
    # Apply both scoring methods
    df['Completeness_Score'] = df.apply(
        lambda row: calculate_score(row, score_type='main'), 
        axis=1
    )
    df['Completeness_Core'] = df.apply(
        lambda row: calculate_score(row, score_type='core'), 
        axis=1
    )
    
    return df


# ============================================================================
# FEATURE SCALING & NORMALIZATION
# ============================================================================

def scale_features(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize feature columns using StandardScaler.
    
    Strategy:
    - Fit scaler on entire filtered dataset
    - Transforms: (value - mean) / std for each feature
    - Output: Zero mean, unit variance per feature
    - Suitable for cosine similarity and clustering
    
    Args:
        df (pd.DataFrame): Dataset with feature columns
        
    Returns:
        Tuple[np.ndarray, StandardScaler]: Scaled features and fitted scaler object
    """
    scaler = StandardScaler()
    
    # Extract feature columns (fill any remaining NaN with 0)
    features = df[FEATURE_COLUMNS].fillna(0)
    
    # Fit and transform
    scaled = scaler.fit_transform(features)
    
    return scaled, scaler


def get_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for all features across dataset.
    
    Useful for:
    - Understanding feature distributions
    - Identifying outliers
    - Documenting baseline statistics
    
    Args:
        df (pd.DataFrame): Dataset with features
        
    Returns:
        pd.DataFrame: Statistics (mean, std, min, max, 25th, 50th, 75th percentile)
    """
    return df[FEATURE_COLUMNS].describe().T


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def process_all_data(csv_path: str, min_90s: int = MIN_MINUTES_PLAYED) -> Dict[str, any]:
    """
    Full data processing pipeline: Load → Validate → Filter → Percentiles → Completeness → Scale
    
    This is the main entry point for the Streamlit app (wrapped in @st.cache_data).
    
    Processing steps:
    1. Load CSV and extract primary positions
    2. Clean feature columns (coerce to float)
    3. Filter by minimum minutes played (90s >= 10)
    4. Validate data quality
    5. Calculate position-specific percentiles (League + Position groups)
    6. Calculate data completeness scores (league-aware)
    7. Scale features for similarity/clustering
    
    Args:
        csv_path (str): Path to english_football_pyramid_master.csv
        min_90s (int): Minimum 90-minute equivalents for inclusion
        
    Returns:
        Dict containing:
        - 'dataframe': Processed pandas DataFrame
        - 'scaled_features': Numpy array of scaled features
        - 'scaler': Fitted StandardScaler object (for inverse transforms)
        - 'validation_report': Data quality checks
        - 'feature_stats': Descriptive statistics for all features
        - 'processing_info': Metadata about processing
        
    Raises:
        FileNotFoundError: If CSV not found
        ValueError: If critical columns missing
    """
    print("=" * 80)
    print("🔄 STARTING DATA PROCESSING PIPELINE")
    print("=" * 80)
    
    # Step 1: Load and initial cleaning
    print("\n[1/7] Loading CSV...")
    df = load_data(csv_path)
    print(f"✓ Loaded {len(df)} total records")
    
    # Step 2: Clean features
    print("\n[2/7] Cleaning feature columns...")
    df = clean_feature_columns(df)
    print(f"✓ Coerced all features to float")
    
    # Step 3: Filter by minutes
    print(f"\n[3/7] Filtering by minimum {min_90s} 90s...")
    df_filtered = filter_by_minutes_played(df, min_90s=min_90s)
    
    # Step 4: Validation
    print("\n[4/7] Validating data quality...")
    validation_report = validate_data(df_filtered)
    print(f"✓ Validation complete")
    if validation_report['warnings']:
        for warning in validation_report['warnings']:
            print(f"  {warning}")
    
    # Step 5: Calculate percentiles
    print("\n[5/7] Calculating position-specific percentiles...")
    df_filtered = calculate_position_percentiles(df_filtered)
    pct_cols = [col for col in df_filtered.columns if '_pct' in col]
    print(f"✓ Added {len(pct_cols)} percentile columns")
    
    # Step 6: Completeness scoring
    print("\n[6/7] Calculating data completeness scores...")
    df_filtered = calculate_data_completeness(df_filtered)
    avg_completeness = df_filtered['Completeness_Score'].mean()
    print(f"✓ Average completeness: {avg_completeness:.1f}%")
    
    # Step 7: Feature scaling
    print("\n[7/7] Scaling features for similarity/clustering...")
    scaled_features, scaler = scale_features(df_filtered)
    print(f"✓ Scaled {scaled_features.shape[1]} features for {scaled_features.shape[0]} players")
    
    # Compute statistics
    feature_stats = get_feature_statistics(df_filtered)
    
    # Summary
    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Final dataset: {len(df_filtered)} players across {df_filtered['League'].nunique()} leagues")
    print(f"Data dropped: {len(df) - len(df_filtered)} players (statistical reliability filtering)")
    
    return {
        'dataframe': df_filtered,
        'scaled_features': scaled_features,
        'scaler': scaler,
        'validation_report': validation_report,
        'feature_stats': feature_stats,
        'processing_info': {
            'csv_path': csv_path,
            'min_90s': min_90s,
            'total_players': len(df_filtered),
            'total_leagues': df_filtered['League'].nunique(),
            'total_positions': df_filtered['Primary_Pos'].nunique(),
            'percentile_columns': pct_cols,
            'feature_count': len(FEATURE_COLUMNS),
        },
    }


# ============================================================================
# HELPER FUNCTIONS FOR STREAMLIT APP
# ============================================================================

def get_player_by_name(df: pd.DataFrame, player_name: str) -> Optional[pd.Series]:
    """
    Retrieve a single player row by name (exact match).
    
    Args:
        df (pd.DataFrame): Processed dataset
        player_name (str): Exact player name
        
    Returns:
        pd.Series: Player row, or None if not found
    """
    matches = df[df['Player'] == player_name]
    if len(matches) > 0:
        return matches.iloc[0]
    return None


def get_players_in_league(df: pd.DataFrame, league: str) -> pd.DataFrame:
    """
    Get all players in a specific league.
    
    Args:
        df (pd.DataFrame): Processed dataset
        league (str): League name
        
    Returns:
        pd.DataFrame: Filtered dataset for league
    """
    return df[df['League'] == league].copy()


def get_players_in_position(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """
    Get all players in a specific primary position.
    
    Args:
        df (pd.DataFrame): Processed dataset
        position (str): Position code (e.g., 'FW', 'MF', 'DF')
        
    Returns:
        pd.DataFrame: Filtered dataset for position
    """
    return df[df['Primary_Pos'] == position].copy()


def get_league_position_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get count of players per league and position.
    
    Useful for understanding data distribution and detecting small groups.
    
    Args:
        df (pd.DataFrame): Processed dataset
        
    Returns:
        pd.DataFrame: Pivot table with league-position combinations
    """
    return pd.crosstab(df['League'], df['Primary_Pos'], margins=True)


def get_percentile_for_player(df: pd.DataFrame, player_name: str) -> Dict[str, float]:
    """
    Get all percentile scores for a specific player.
    
    Args:
        df (pd.DataFrame): Processed dataset
        player_name (str): Player name
        
    Returns:
        Dict mapping feature name to percentile rank (0-100)
    """
    player = get_player_by_name(df, player_name)
    if player is None:
        return {}
    
    percentiles = {}
    for feature in FEATURE_COLUMNS:
        pct_col = f'{feature}_pct'
        if pct_col in player.index:
            percentiles[feature] = player[pct_col]
    
    return percentiles
