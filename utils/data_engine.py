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
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
import glob

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
import unicodedata
from rapidfuzz import process, fuzz


def normalize_name(name: any) -> str:
    """
    Lowercase and remove accents/quotes/extra spaces for matching.
    """
    if pd.isna(name):
        return ""
    name = str(name).replace('"', '').replace("'", "").strip().lower()
    return unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')

def clean_player_name(name: any) -> str:
    """
    Preserve display name but remove quotes.
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
    (Data Fix: Corrected file swap for Bundesliga/Serie A - Cache Bump)
    """
    try:
        # robust delimiter detection
        try:
            df = pd.read_csv(file_path, low_memory=low_memory)
            if len(df.columns) <= 1:
                # Try semicolon
                df = pd.read_csv(file_path, sep=';', low_memory=low_memory)
        except:
             df = pd.read_csv(file_path, sep=';', low_memory=low_memory)
            
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

    # Dynamic Advanced Stats Loading
    data_dir = os.path.dirname(file_path) # Assuming advanced stats are in same/similar dir
    if not data_dir: data_dir = 'data'
    
    # Look for any file containing "Advanced Stats"
    advanced_files = glob.glob(os.path.join(data_dir, "*Advanced Stats.csv"))
    
    print(f"ℹ️  Found {len(advanced_files)} advanced stats files.")

    for adv_path in advanced_files:
        try:
            filename = os.path.basename(adv_path)
            # Infer league from filename
            current_league = None
            for league_name in LEAGUES:
                # Handle "Ligue Un" vs "Ligue 1"
                check_name = league_name
                if league_name == 'Ligue 1':
                    check_name = 'Ligue Un'
                
                if check_name.lower() in filename.lower():
                    current_league = league_name
                    break
            
            if not current_league:
                print(f"⚠️  Skipping {filename}: Could not infer league from filename.")
                continue

            # Load with semicolon delimiter
            adv_df = pd.read_csv(adv_path, sep=';', low_memory=False)
            
            # Normalize column names to match main dataframe
            column_map = {
                'player': 'Player',
                'team': 'Squad',
                'goals': 'Gls',
                'a': 'Ast',
                'min': 'Min',
                'xg90': 'xG90',
                'xa90': 'xA90',
                'xgchain90': 'xGChain90',
                'xgbuildup90': 'xGBuildup90'
            }
            # Add case-insensitive mappings
            current_cols = {c.lower(): c for c in adv_df.columns}
            rename_dict = {}
            for k, v in column_map.items():
                if k.lower() in current_cols:
                    rename_dict[current_cols[k.lower()]] = v
            
            adv_df = adv_df.rename(columns=rename_dict)
            
            if 'Player' not in adv_df.columns:
                print(f"⚠️  Could not find 'Player' column in {adv_path}. Skipping.")
                continue

            # Prepare for merge with lowercase keys
            adv_df['MergeKey_Player'] = adv_df['Player'].apply(normalize_name)
            if 'Squad' in adv_df.columns:
                adv_df['MergeKey_Squad'] = adv_df['Squad'].apply(normalize_name)
            
            # Identify players from this league in the main dataframe
            league_mask = df['League'] == current_league
            if not league_mask.any():
                print(f"ℹ️  No players found for {current_league} in master file (checking '{current_league}').")
                continue

            league_players = df[league_mask].copy()
            league_players['MergeKey_Player'] = league_players['Player'].apply(normalize_name)
            league_players['MergeKey_Squad'] = league_players['Squad'].apply(normalize_name)
            
            # Normalize squad names in master
            squad_fixes = {
                'rb leipzig': 'rasenballsport leipzig',
                'paris saint-germain': 'paris saint germain',
                'inter': 'inter milan',
                'milan': 'ac milan',
                'leverkusen': 'bayer leverkusen',
                'gladbach': 'borussia monchengladbach',
                'real madrid': 'real madrid',
                'manchester city': 'manchester city',
                'bayern munich': 'bayern munich',
                'athletic club': 'athletic bilbao',
                'manchester utd': 'manchester united',
                'nott\'m forest': 'nottingham forest',
                'wolves': 'wolverhampton wanderers',
                'west ham united': 'west ham',
                'tottenham hotspur': 'tottenham',
                'leicester city': 'leicester',
                'ipswich town': 'ipswich',
                'newcastle utd': 'newcastle united'
            }
            league_players['MergeKey_Squad'] = league_players['MergeKey_Squad'].replace(squad_fixes)
            
            # Sub-select advanced data
            merge_cols_adv = ['MergeKey_Player', 'MergeKey_Squad'] + [c for c in advanced_metrics if c in adv_df.columns]
            # Ensure we have at least one metric to merge
            if len(merge_cols_adv) <= 2:
                continue

            adv_subset = adv_df[merge_cols_adv].drop_duplicates(subset=['MergeKey_Player', 'MergeKey_Squad'])
            
            # Remove existing advanced columns from league_players to avoid suffix creation
            league_players = league_players.drop(columns=[c for c in advanced_metrics if c in league_players.columns], errors='ignore')
            
            # Merge on keys
            merged = pd.merge(league_players, adv_subset, on=['MergeKey_Player', 'MergeKey_Squad'], how='left')
            
            # Fuzzy fallback
            unmatched_mask = merged['xG90'].isna() | (merged['xG90'] == 0)
            if unmatched_mask.any():
                idx_to_update = merged[unmatched_mask].index
                for idx in idx_to_update:
                    row = merged.loc[idx]
                    squad_key = row['MergeKey_Squad']
                    candidates = adv_subset[adv_subset['MergeKey_Squad'] == squad_key]
                    if not candidates.empty:
                        best = process.extractOne(row['MergeKey_Player'], candidates['MergeKey_Player'], scorer=fuzz.token_sort_ratio)
                        if best and best[1] >= 75:
                            match_data = candidates[candidates['MergeKey_Player'] == best[0]].iloc[0]
                            for col in advanced_metrics:
                                if col in match_data:
                                    merged.at[idx, col] = match_data[col]

            # Cleanup
            merged = merged.drop(columns=['MergeKey_Player', 'MergeKey_Squad'], errors='ignore')
            for col in advanced_metrics:
                if col in merged.columns:
                    merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0.0)
                else:
                    merged[col] = 0.0
            
            # Update the main dataframe
            # Drop old rows for this league and append new merged rows
            df = df[df['League'] != current_league]
            df = pd.concat([df, merged], axis=0, ignore_index=True)
            print(f"📊 Merged advanced stats for {current_league} ({len(merged)} players)")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not merge advanced stats for {adv_path}: {e}")

    # Final validation of required columns
    required_columns = ['Player', 'Squad', 'League', 'Pos', 'Age', '90s'] + FEATURE_COLUMNS
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"⚠️  Missing columns in merged dataset: {missing_columns}")
    
    return df



def clean_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce all feature columns to float and handle missing values.
    
    Strategy (Anti-Bias):
    - Coerce to float
    - For NL: Keep defensive metrics (non Gls/Ast) as NaN to prevent "Defensive Failure" bias.
    - For others: Fill NaN with 0.
    """
    df = df.copy()
    
    attacking_metrics = ['Gls/90', 'Ast/90']
    
    # Valid numeric coercion
    for col in FEATURE_COLUMNS + GK_FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create Finishing Efficiency (Gls - xG) - New metric for Hidden Gems
    if 'Gls/90' in df.columns and 'xG90' in df.columns:
         df['Finishing_Efficiency'] = df['Gls/90'] - df['xG90']
    else:
         df['Finishing_Efficiency'] = 0.0

    # -------------------------------------------------------------------------
    # NATIONAL LEAGUE IMPUTATION (Median from League Two)
    # -------------------------------------------------------------------------
    print("ℹ️  Performing National League defensive imputation...")
    defensive_metrics = ['Crs/90', 'Int/90', 'TklW/90', 'Fls/90', 'Fld/90']
    
    # Calculate medians from League Two by position
    league_two_mask = df['League'] == 'League Two'
    if league_two_mask.any():
        l2_data = df[league_two_mask]
        l2_medians = l2_data.groupby('Primary_Pos')[defensive_metrics].median()
        
        # Apply to National League
        nl_mask = df['League'] == 'National League'
        if nl_mask.any():
            # Vectorized approach for better performance
            for metric in defensive_metrics:
                if metric in df.columns:
                    # Create a mapping series
                    mapping = df.loc[nl_mask, 'Primary_Pos'].map(l2_medians[metric])
                    # Fill NaNs where we have a mapping
                    df.loc[nl_mask, metric] = df.loc[nl_mask, metric].fillna(mapping)

    # Impute missing values with Position-Specific League Averages (or Global Fallback)
    # This prevents "Limited Data" players from looking like failures (0s)
    # Strategy: 
    # 1. Fill with League+Position Mean
    # 2. Fill remaining with Global Position Mean
    # 3. Fill remaining with 0 (safe fallback)
    
    print("ℹ️  Imputing missing data with league/position averages...")
    
    # We do this for FEATURE_COLUMNS (Outfield) and GK_FEATURE_COLUMNS
    all_metrics = list(set(FEATURE_COLUMNS + GK_FEATURE_COLUMNS))
    
    for col in all_metrics:
        if col not in df.columns:
            continue
            
        # 1. League + Position Mean
        df[col] = df[col].fillna(
            df.groupby(['League', 'Primary_Pos'])[col].transform('mean')
        )
        
        # 2. Global Position Mean
        df[col] = df[col].fillna(
            df.groupby(['Primary_Pos'])[col].transform('mean')
        )
        
        # 3. Zero Fallback
        df[col] = df[col].fillna(0.0)
            
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
            # Outfield players: Use FEATURE_COLUMNS
            if pos != 'GK':
                for feature in FEATURE_COLUMNS:
                    pct_col = f'{feature}_pct'
                    quality_col = f'{feature}_pct_quality'
                    
                    df.loc[mask, pct_col] = (
                        df.loc[mask, feature].rank(pct=True, na_option='bottom') * 100
                    )
                    df.loc[mask, quality_col] = quality
            
            # Goalkeepers: Use GK_FEATURE_COLUMNS
            else:
                for feature in GK_FEATURE_COLUMNS:
                    # Skip if column doesn't exist
                    if feature not in df.columns:
                        continue
                    
                    pct_col = f'{feature}_pct'
                    quality_col = f'{feature}_pct_quality'
                    
                    # For goalkeeper metrics, handle direction properly
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
    
    
    # Explicitly mask outfield percentiles for GKs to prevent contamination
    # (And vice versa if needed, though usually GKs are the issue)
    gk_mask = df['Primary_Pos'] == 'GK'
    if gk_mask.any():
        outfield_pct_cols = [f'{c}_pct' for c in FEATURE_COLUMNS if f'{c}_pct' in df.columns]
        if outfield_pct_cols:
            df.loc[gk_mask, outfield_pct_cols] = np.nan
        
        outfield_qual_cols = [f'{c}_pct_quality' for c in FEATURE_COLUMNS if f'{c}_pct_quality' in df.columns]
        if outfield_qual_cols:
            df.loc[gk_mask, outfield_qual_cols] = np.nan
            
    return df


# ============================================================================
# DATA COMPLETENESS SCORING
# ============================================================================

def calculate_data_completeness(df: pd.DataFrame) -> pd.DataFrame:
    """
    League-aware completeness scoring (updated for tier-aware requirements).
    
    - Tier 1 (PL/Top 5): Requires Advanced 4 (xG/xA/Chain/Buildup) + core for 100%
    - National League: Only requires core 2 (Gls/Ast) for 100%
    - Others: Core 9 metrics
    """
    df = df.copy()
    
    # Standard Tier 1 leagues that have advanced stats
    ADVANCED_LEAGUES = ['Premier League', 'Bundesliga', 'La Liga', 'Serie A', 'Ligue 1']
    
    def calculate_score(row: pd.Series, score_type: str = 'main') -> float:
        if score_type == 'core':
            expected_metrics = ['Gls/90', 'Ast/90']
        else:
            league = row['League']
            if league == 'National League':
                expected_metrics = ['Gls/90', 'Ast/90']
            elif league in ADVANCED_LEAGUES:
                expected_metrics = FEATURE_COLUMNS # All 13
            else:
                # Core 9 for Championship, L1, L2
                expected_metrics = ['Gls/90', 'Ast/90', 'Sh/90', 'SoT/90', 'Crs/90', 'Int/90', 'TklW/90', 'Fls/90', 'Fld/90']
        
        found = sum(
            1 for metric in expected_metrics 
            if metric in row.index and pd.notna(row[metric]) and row[metric] > 0
        )
        
        return (found / len(expected_metrics)) * 100
    
    df['Completeness_Score'] = df.apply(lambda row: calculate_score(row, 'main'), axis=1)
    df['Completeness_Core'] = df.apply(lambda row: calculate_score(row, 'core'), axis=1)
    
    return df


# ============================================================================
# FEATURE SCALING & NORMALIZATION
# ============================================================================

def scale_features(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, StandardScaler]]:
    """
    Positional Scaling (The "Anti-Bias" Fix).
    Fit and transform features separately for FW, MF, DF groups.
    GKs are handled separately.
    
    Returns:
        scaled_matrix: (N_samples, N_features + N_gk_features)
        scalers: Dict mapping position group ('FW', 'MF', 'DF', 'GK') to its fitted scaler.
                 This ensures we can re-scale correctly even if filtered.
    """
    # Initialize combined scaled matrix
    # Columns: [FEATURE_COLUMNS (13) | GK_FEATURE_COLUMNS (9)]
    # Total width: 22
    total_width = len(FEATURE_COLUMNS) + len(GK_FEATURE_COLUMNS)
    scaled_matrix = np.zeros((len(df), total_width))
    scalers = {}
    
    # Outfield groups
    # Explicit boolean mask to exclude Goalkeepers (GK) from all outfield metric calculations
    outfield_mask = df['Primary_Pos'] != 'GK'
    
    for pos in ['FW', 'MF', 'DF']:
        # Combine position check with explicit outfield mask
        mask = (df['Primary_Pos'] == pos) & outfield_mask
        if mask.any():
            scaler = StandardScaler()
            # Fill NaN with 0 ONLY for scaling
            pos_data = df.loc[mask, FEATURE_COLUMNS].fillna(0)
            scaled_outfield = scaler.fit_transform(pos_data)
            scaled_matrix[mask, :len(FEATURE_COLUMNS)] = scaled_outfield
            scalers[pos] = scaler
        else:
            # Fallback if a position is missing entirely
            scalers[pos] = StandardScaler() # unused but prevents key error
            
    # Goalkeeper group
    gk_mask = df['Primary_Pos'] == 'GK'
    if gk_mask.any():
        gk_scaler = StandardScaler()
        gk_data = df.loc[gk_mask, GK_FEATURE_COLUMNS].fillna(0)
        scaled_gk = gk_scaler.fit_transform(gk_data)
        scaled_matrix[gk_mask, len(FEATURE_COLUMNS):] = scaled_gk
        scalers['GK'] = gk_scaler
    else:
        scalers['GK'] = StandardScaler()
    
    return scaled_matrix, scalers


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
    print("\n[7/7] Scaling features and calculating PCA...")
    scaled_features, scalers = scale_features(df_filtered)
    print(f"✓ Scaled features for {scaled_features.shape[0]} players")
    
    # Step 8: PCA Calculation (Phase 3 Prep)
    # We calculate PCA coordinates on the Scaled Features
    # Note: We use the first 13 columns (Outfield features) for PCA as it's primarily for style analysis
    # GKs will have their own or 0s
    try:
        pca = PCA(n_components=2)
        # We only take the Outfield columns for the Galaxy view to prevent GK outliers distorting the map
        outfield_matrix = scaled_features[:, :len(FEATURE_COLUMNS)]
        
        # Handle the fact that GKs are 0 in these columns - this clusters them at origin, which is fine
        pca_coords = pca.fit_transform(outfield_matrix)
        
        df_filtered['PCA_X'] = pca_coords[:, 0]
        df_filtered['PCA_Y'] = pca_coords[:, 1]
        print(f"✓ Calculated PCA Coordinates (Explained Variance: {pca.explained_variance_ratio_})")
    except Exception as e:
        print(f"⚠️ PCA calculation failed: {e}")
        df_filtered['PCA_X'] = 0.0
        df_filtered['PCA_Y'] = 0.0
    
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
        'scalers': scalers,
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
