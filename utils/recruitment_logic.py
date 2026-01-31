"""
recruitment_logic.py - Logic for projecting player performance across league tiers.

This module provides:
- League coefficient mappings
- Performance projection logic (step-up analysis)
- Identification of 'ready-now' vs 'development' prospects
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

# League Coefficients (Approximate relative strength to Premier League)
# Based on global power rankings and transfer success rates
LEAGUE_COEFFICIENTS = {
    'Premier League': 1.0,
    'Bundesliga': 0.92,
    'La Liga': 0.92,
    'Serie A': 0.90,
    'Ligue 1': 0.85,  # Slightly lower
    'Championship': 0.70, # The "Step Up" gap
    'League One': 0.55,
    'League Two': 0.45,
    'National League': 0.35,
    'Eredivisie': 0.80,
    'Primeira Liga': 0.78,
}

DEFAULT_COEFFICIENT = 0.60

def get_league_coefficient(league: str) -> float:
    """Get the strength coefficient for a league."""
    return LEAGUE_COEFFICIENTS.get(league, DEFAULT_COEFFICIENT)

def project_to_tier(
    df: pd.DataFrame,
    target_tier: str = 'Premier League',
    strict_mode: bool = False
) -> pd.DataFrame:
    """
    Project player stats to a target tier using league coefficients.
    
    Args:
        df: Player DataFrame
        target_tier: Target league name
        strict_mode: If True, uses more aggressive penalties for lower leagues
        
    Returns:
        DataFrame with added 'Projected_{Stat}' columns
    """
    df_proj = df.copy()
    
    # Target coefficient
    target_coeff = get_league_coefficient(target_tier)
    
    # Identify numeric stat columns to project (excluding PCT and generic info)
    # We focus on per-90 metrics
    stat_cols = [
        c for c in df.columns 
        if c not in ['Age', '90s'] 
        and df[c].dtype in [np.float64, np.float32]
        and not c.endswith('_pct')
        and not c.startswith('PCA_')
        and not c.startswith('Projected_')
    ]
    
    # Metrics that shouldn't be discounted (rates/percentages might be different, but let's strictly discount volume stats)
    # Actually, almost all stats suffer from league quality differences.
    # Exceptions might be physical attributes (if we had them), but we don't.
    
    print(f"ℹ️  Projecting performance to {target_tier} (Coeff: {target_coeff})")
    
    for idx, row in df_proj.iterrows():
        # Source league coefficient
        source_league = row['League']
        source_coeff = get_league_coefficient(source_league)
        
        # Calculate Step-Up Factor
        # If moving UP: Factor < 1.0 (Discount)
        # If moving DOWN: Factor > 1.0 (Premium)
        # Factor = Source / Target
        
        if strict_mode and source_coeff < target_coeff:
            # Apply extra penalty for big jumps
            step_factor = (source_coeff / target_coeff) * 0.9
        else:
            step_factor = source_coeff / target_coeff
            
        # Apply to stats
        for col in stat_cols:
            if pd.notna(row[col]):
                df_proj.at[idx, f'Projected_{col}'] = row[col] * step_factor
                
    # Add a flag for 'Step Up Ready'
    # Logic: If Projected Gls/90 or Ast/90 is still above PL median? 
    # For now, just the columns are enough.
    
    return df_proj
