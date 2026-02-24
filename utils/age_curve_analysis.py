"""
age_curve_analysis.py - Advanced age-curve analysis for identifying high-ceiling prospects.

This module provides a robust framework for:
- Hierarchical Cohort Analysis (comparing players to league, position, and age peers)
- Age-Curve Normalization (handling small sample sizes with fallback logic)
- "Ahead of the Curve" Anomaly Detection (Z-score and Percentile based)
- Global vs. Local Ranking (comparing a player to their league vs. the entire database)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AgeCurveAnomaly:
    """Enhanced Data class for age-curve analysis results."""
    player_name: str
    age: int
    position: str
    league: str
    metric: str
    player_value: float
    age_mean: float
    age_std: float
    z_score: float
    percentile_rank: float
    is_high_ceiling: bool
    confidence_score: float  # 0.0 to 1.0 based on sample size
    cohort_type: str        # 'League-Pos', 'Position', or 'Global'
    global_z_score: float   # Z-score compared to ALL players of same age (cross-league)
    global_percentile: float
    local_cohort_size: int   # Number of players in the primary comparison group
    global_cohort_size: int  # Number of players in the global age group

class AgeCurveAnalyzer:
    """
    Overhauled analyzer for identifying players performing above their age cohort.
    
    Features:
    - Robust handling of missing data and small sample sizes.
    - Hierarchical fallback: If league-specific data is sparse, fall back to position-wide data.
    - Global Benchmarking: Every player is compared to both their immediate peers (Local) 
      and the entire database's same-age cohort (Global).
    """
    
    HIGH_CEILING_THRESHOLD = 2.0  # Z-score threshold for "Ahead of Curve"
    ELITE_THRESHOLD = 3.0         # Z-score threshold for "Elite Prospect"
    MIN_SAMPLE_SIZE_LOCAL = 5     # Minimum players for league-specific analysis
    MIN_SAMPLE_SIZE_GLOBAL = 10   # Minimum players for global age analysis
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer with robust data cleaning.
        
        Args:
            df: Player DataFrame
        """
        self.df = df.copy()
        
        # Ensure critical columns are numeric and handle NaNs
        if 'Age' in self.df.columns:
            self.df['Age'] = pd.to_numeric(self.df['Age'], errors='coerce')
            self.df = self.df.dropna(subset=['Age'])
            # Cast to float for grouping but keep as displayable int later
            self.df['Age_Group'] = self.df['Age'].round()
        else:
            logger.error("DataFrame passed to AgeCurveAnalyzer missing 'Age' column")
            
        self._cache = {}

    def _get_stats(self, df_view: pd.DataFrame, metric: str, age: float) -> Dict[str, Any]:
        """Internal helper to calculate stats for a specific age in a view."""
        # Filter for the specific age (with some tolerance for rounding if float)
        cohort = df_view[df_view['Age_Group'] == age][metric].dropna()
        
        if len(cohort) == 0:
            return {'mean': 0, 'std': 0, 'count': 0, 'median': 0}
            
        return {
            'mean': float(cohort.mean()),
            'std': float(cohort.std()) if len(cohort) > 1 else 0.0,
            'count': int(len(cohort)),
            'median': float(cohort.median())
        }

    def calculate_age_curves(
        self,
        metric: str,
        position: Optional[str] = None,
        league: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Calculate age-curve trends with optional filters.
        """
        df_filtered = self.df.copy()
        if position:
            df_filtered = df_filtered[df_filtered['Primary_Pos'] == position]
        if league:
            df_filtered = df_filtered[df_filtered['League'] == league]
            
        if len(df_filtered) == 0:
            return pd.DataFrame(columns=['Age', 'Mean', 'Std', 'Count', 'Position', 'League', 'Metric'])
            
        # Group by Age_Group
        age_stats = df_filtered.groupby('Age_Group')[metric].agg([
            ('Mean', 'mean'),
            ('Std', 'std'),
            ('Count', 'count'),
            ('Median', 'median')
        ]).reset_index()
        
        age_stats.rename(columns={'Age_Group': 'Age'}, inplace=True)
        age_stats['Std'] = age_stats['Std'].fillna(0.0)
        age_stats['Position'] = position or 'All'
        age_stats['League'] = league or 'All'
        age_stats['Metric'] = metric
        
        return age_stats

    def get_player_age_curve_status(
        self,
        player_name: str,
        metric: str,
    ) -> Optional[AgeCurveAnomaly]:
        """
        Detailed analysis of a player relative to their age group using hierarchical fallback.
        """
        # Handle "Player (Squad)" format for robustness with UI selectboxes
        if " (" in player_name and player_name.endswith(")"):
            try:
                name_part = player_name.split(" (")[0].strip()
                squad_part = player_name.split(" (")[1].replace(")", "").strip()
                player_data = self.df[
                    (self.df['Player'] == name_part) & 
                    (self.df['Squad'] == squad_part)
                ]
            except Exception:
                player_data = self.df[self.df['Player'] == player_name]
        else:
            player_data = self.df[self.df['Player'] == player_name]
            
        if len(player_data) == 0:
            return None
            
        row = player_data.iloc[0]
        age = row['Age_Group']
        pos = row['Primary_Pos']
        league = row['League']
        val = row.get(metric, 0.0)
        
        if pd.isna(val) or pd.isna(age):
            return None

        # 1. HIERARCHICAL LOCAL COHORT SELECTION
        # Strategy: Prefer League-Position-Age, then Position-Age, then Global-Age
        
        # Local (League + Position)
        local_df = self.df[(self.df['League'] == league) & (self.df['Primary_Pos'] == pos)]
        local_stats = self._get_stats(local_df, metric, age)
        
        cohort_type = "League-Pos"
        stats_to_use = local_stats
        
        # Fallback to Position-only if sample too small
        if local_stats['count'] < self.MIN_SAMPLE_SIZE_LOCAL:
            pos_df = self.df[self.df['Primary_Pos'] == pos]
            pos_stats = self._get_stats(pos_df, metric, age)
            
            if pos_stats['count'] >= self.MIN_SAMPLE_SIZE_LOCAL:
                stats_to_use = pos_stats
                cohort_type = "Position"
            else:
                # Absolute fallback to Global Age cohort
                global_age_stats = self._get_stats(self.df, metric, age)
                stats_to_use = global_age_stats
                cohort_type = "Global"

        # 2. GLOBAL BENCHMARKING (Compare to all players of same age across ALL leagues)
        global_stats = self._get_stats(self.df, metric, age)
        global_cohort = self.df[self.df['Age_Group'] == age][metric].dropna()
        
        # 3. PERCENTILE CALCULATION
        local_cohort = None
        if cohort_type == "League-Pos":
            local_cohort = local_df[local_df['Age_Group'] == age][metric].dropna()
        elif cohort_type == "Position":
            local_cohort = self.df[(self.df['Primary_Pos'] == pos) & (self.df['Age_Group'] == age)][metric].dropna()
        else:
            local_cohort = global_cohort
            
        if len(local_cohort) > 0:
            pct_rank = (local_cohort < val).mean() * 100.0
        else:
            pct_rank = 50.0
            
        # Global Percentile
        if len(global_cohort) > 0:
            global_pct = (global_cohort < val).mean() * 100.0
        else:
            global_pct = 50.0

        # 4. Z-SCORE CALCULATION
        mean, std = stats_to_use['mean'], stats_to_use['std']
        z_score = (val - mean) / std if std > 0 else 0.0
        
        g_mean, g_std = global_stats['mean'], global_stats['std']
        global_z = (val - g_mean) / g_std if g_std > 0 else 0.0

        # Confidence Score based on cohort size
        confidence = min(stats_to_use['count'] / 50.0, 1.0)

        return AgeCurveAnomaly(
            player_name=player_name,
            age=int(age),
            position=pos,
            league=league,
            metric=metric,
            player_value=float(val),
            age_mean=mean,
            age_std=std,
            z_score=float(z_score),
            percentile_rank=float(pct_rank),
            is_high_ceiling=(z_score >= self.HIGH_CEILING_THRESHOLD),
            confidence_score=confidence,
            cohort_type=cohort_type,
            global_z_score=float(global_z),
            global_percentile=float(global_pct),
            local_cohort_size=int(stats_to_use['count']),
            global_cohort_size=int(global_stats['count'])
        )

    def get_high_ceiling_prospects(
        self,
        max_age: int = 23,
        min_z_score: float = HIGH_CEILING_THRESHOLD,
        top_n: int = 25,
        min_90s: float = 5.0
    ) -> pd.DataFrame:
        """
        Aggregated report of top prospects performing ahead of their age curve across all relevant metrics.
        """
        all_results = []
        metrics_to_test = ['Gls/90', 'Ast/90', 'Sh/90', 'SoT/90', 'TklW/90', 'Int/90', 'xG90', 'xA90']
        
        # Filter for young players with enough minutes
        young_df = self.df[(self.df['Age_Group'] <= max_age) & (self.df['90s'] >= min_90s)].copy()
        
        for metric in metrics_to_test:
            if metric not in young_df.columns:
                continue
            
            # Group by Age and Position for local z-scores in bulk
            # This is faster than calling get_player_age_curve_status for thousands of rows
            for pos in young_df['Primary_Pos'].unique():
                pos_mask = young_df['Primary_Pos'] == pos
                pos_df = young_df[pos_mask]
                
                if len(pos_df) < 5: continue
                
                # Calculate mean/std per age group for this position
                stats_df = pos_df.groupby('Age_Group')[metric].transform(['mean', 'std', 'count'])
                
                # Z-score within Position-Age cohort
                z_col = (pos_df[metric] - stats_df['mean']) / stats_df['std']
                
                # Filter for anomalies
                anomalies = pos_df[z_col >= min_z_score].copy()
                if not anomalies.empty:
                    anomalies['Z_Score'] = z_col[z_col >= min_z_score]
                    anomalies['Test_Metric'] = metric
                    all_results.append(anomalies[['Player', 'Age', 'Primary_Pos', 'League', 'Test_Metric', metric, 'Z_Score']])

        if not all_results:
            return pd.DataFrame()
            
        merged = pd.concat(all_results)
        
        # Final summary by player
        summary = merged.groupby('Player').agg({
            'Z_Score': 'mean',
            'Test_Metric': 'count',
            'Age': 'first',
            'Primary_Pos': 'first',
            'League': 'first'
        }).reset_index()
        
        summary.columns = ['Player', 'Avg_Anomaly_Z', 'High_Ceiling_Metric_Count', 'Age', 'Pos', 'League']
        summary = summary.sort_values(['High_Ceiling_Metric_Count', 'Avg_Anomaly_Z'], ascending=False)
        
        return summary.head(top_n)

def format_age_curve_badge(anomaly: AgeCurveAnomaly) -> str:
    """
    Format a visually stunning badge/narrative for the UI.
    Uses HTML styling for Premium impact.
    """
    is_young = anomaly.age <= 23
    
    if is_young:
        if anomaly.z_score >= 3.0:
            return f"🌟 ELITE GENERATIONAL PROSPECT ({anomaly.z_score:.1f}σ Above Age Average)"
        elif anomaly.z_score >= 2.5:
            return f"💎 HIGH-CEILING PROSPECT ({anomaly.z_score:.1f}σ Above Age Average)"
        elif anomaly.z_score >= 2.0:
            return f"🚀 AHEAD OF THE CURVE ({anomaly.z_score:.1f}σ Above Age Average)"
        elif anomaly.global_z_score >= 1.5:
            return f"📈 RISING TALENT (Global Top Tier for Age {anomaly.age})"
    
    # For older players or lower z-scores
    if anomaly.z_score >= 1.5:
        return f"✅ PERFORMING ABOVE AGE COHORT ({anomaly.z_score:.1f}σ Above Average)"
    elif anomaly.z_score >= 0.5:
        return "👍 Solid Performance for Age"
    elif anomaly.z_score <= -1.5:
        return "⚠️ Underperforming Age Expectation"
    else:
        return "📊 Meeting Age-Group Expectations"
