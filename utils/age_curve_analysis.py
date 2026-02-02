"""
age_curve_analysis.py - Age-curve anomaly detection for identifying high-ceiling prospects.

This module provides:
- Age-curve calculation (mean performance by age for each position/league)
- Z-score anomaly detection (players >2 std deviations above age average)
- "Ahead of the Curve" prospect identification
- High-ceiling prospect badges for UI display
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .constants import FEATURE_COLUMNS, PRIMARY_POSITIONS


@dataclass
class AgeCurveAnomaly:
    """Data class for age-curve anomaly results."""
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


class AgeCurveAnalyzer:
    """
    Analyzes age curves and identifies players performing above their age cohort.
    
    Strategy:
    - For each position and league, calculate mean/std performance by age
    - Compute Z-scores for each player relative to their age group
    - Flag players with Z-score > 2.0 as "High-Ceiling Prospects"
    - Provide age-curve visualizations and anomaly reports
    """
    
    HIGH_CEILING_THRESHOLD = 2.0  # Z-score threshold for "Ahead of Curve"
    MIN_SAMPLE_SIZE = 5  # Minimum players per age group for reliable stats
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize analyzer.
        
        Args:
            df: Player DataFrame with Age, Primary_Pos, League, and metric columns
        """
        self.df = df.copy()
        self.age_curves = {}  # Cache for age curve statistics
        self.anomalies = []  # List of detected anomalies
    
    def calculate_age_curves(
        self,
        metric: str,
        position: Optional[str] = None,
        league: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Calculate age-curve statistics (mean, std, count) for a metric.
        
        Args:
            metric: Metric to analyze (e.g., 'Gls/90')
            position: Filter by position (None for all)
            league: Filter by league (None for all)
            
        Returns:
            DataFrame with columns: Age, Mean, Std, Count, Position, League
        """
        # Filter data
        df_filtered = self.df.copy()
        
        if position:
            df_filtered = df_filtered[df_filtered['Primary_Pos'] == position]
        
        if league:
            df_filtered = df_filtered[df_filtered['League'] == league]
        
        # Group by age and calculate statistics
        age_stats = df_filtered.groupby('Age')[metric].agg([
            ('Mean', 'mean'),
            ('Std', 'std'),
            ('Count', 'count'),
            ('Median', 'median'),
            ('Min', 'min'),
            ('Max', 'max'),
        ]).reset_index()
        
        # Filter out age groups with insufficient sample size
        age_stats = age_stats[age_stats['Count'] >= self.MIN_SAMPLE_SIZE]
        
        # Add context
        age_stats['Position'] = position if position else 'All'
        age_stats['League'] = league if league else 'All'
        age_stats['Metric'] = metric
        
        return age_stats
    
    def detect_anomalies(
        self,
        metric: str,
        position: Optional[str] = None,
        league: Optional[str] = None,
        z_threshold: float = HIGH_CEILING_THRESHOLD,
    ) -> List[AgeCurveAnomaly]:
        """
        Detect players performing significantly above their age cohort.
        
        Args:
            metric: Metric to analyze
            position: Filter by position
            league: Filter by league
            z_threshold: Z-score threshold for anomaly detection
            
        Returns:
            List of AgeCurveAnomaly objects for high-ceiling prospects
        """
        # Calculate age curves
        age_curves = self.calculate_age_curves(metric, position, league)
        
        # Filter data
        df_filtered = self.df.copy()
        
        if position:
            df_filtered = df_filtered[df_filtered['Primary_Pos'] == position]
        
        if league:
            df_filtered = df_filtered[df_filtered['League'] == league]
        
        # Merge age statistics
        df_with_age_stats = df_filtered.merge(
            age_curves[['Age', 'Mean', 'Std']],
            on='Age',
            how='left'
        )
        
        # Calculate Z-scores
        df_with_age_stats['Z_Score'] = (
            (df_with_age_stats[metric] - df_with_age_stats['Mean']) / 
            df_with_age_stats['Std']
        )
        
        # Get percentile rank within age group
        df_with_age_stats['Age_Percentile'] = df_with_age_stats.groupby('Age')[metric].rank(pct=True) * 100
        
        # Filter anomalies
        anomalies_df = df_with_age_stats[
            (df_with_age_stats['Z_Score'] >= z_threshold) &
            (df_with_age_stats['Std'].notna())  # Exclude ages with insufficient data
        ].copy()
        
        # Convert to AgeCurveAnomaly objects
        anomalies = []
        for _, row in anomalies_df.iterrows():
            anomaly = AgeCurveAnomaly(
                player_name=row['Player'],
                age=int(row['Age']),
                position=row['Primary_Pos'],
                league=row['League'],
                metric=metric,
                player_value=float(row[metric]),
                age_mean=float(row['Mean']),
                age_std=float(row['Std']),
                z_score=float(row['Z_Score']),
                percentile_rank=float(row['Age_Percentile']),
                is_high_ceiling=True,
            )
            anomalies.append(anomaly)
        
        # Sort by Z-score descending
        anomalies.sort(key=lambda x: x.z_score, reverse=True)
        
        return anomalies
    
    def get_player_age_curve_status(
        self,
        player_name: str,
        metric: str,
    ) -> Optional[AgeCurveAnomaly]:
        """
        Get age-curve status for a specific player.
        
        Args:
            player_name: Player name
            metric: Metric to analyze
            
        Returns:
            AgeCurveAnomaly object or None if player not found
        """
        player_data = self.df[self.df['Player'] == player_name]
        
        if len(player_data) == 0:
            return None
        
        player_row = player_data.iloc[0]
        position = player_row['Primary_Pos']
        league = player_row['League']
        age = int(player_row['Age'])
        
        # Get age curve for this position/league
        age_curves = self.calculate_age_curves(metric, position, league)
        
        # Find age group stats
        age_stats = age_curves[age_curves['Age'] == age]
        
        if len(age_stats) == 0:
            return None
        
        age_mean = float(age_stats.iloc[0]['Mean'])
        age_std = float(age_stats.iloc[0]['Std'])
        player_value = float(player_row[metric])
        
        # Calculate Z-score
        z_score = (player_value - age_mean) / age_std if age_std > 0 else 0
        
        # Get percentile within age group
        age_cohort = self.df[
            (self.df['Age'] == age) &
            (self.df['Primary_Pos'] == position) &
            (self.df['League'] == league)
        ]
        percentile_rank = (age_cohort[metric] < player_value).sum() / len(age_cohort) * 100
        
        return AgeCurveAnomaly(
            player_name=player_name,
            age=age,
            position=position,
            league=league,
            metric=metric,
            player_value=player_value,
            age_mean=age_mean,
            age_std=age_std,
            z_score=z_score,
            percentile_rank=percentile_rank,
            is_high_ceiling=(z_score >= self.HIGH_CEILING_THRESHOLD),
        )
    
    def get_high_ceiling_prospects(
        self,
        max_age: int = 23,
        min_z_score: float = HIGH_CEILING_THRESHOLD,
        top_n: int = 50,
    ) -> pd.DataFrame:
        """
        Get comprehensive list of high-ceiling prospects across all metrics.
        
        Args:
            max_age: Maximum age for prospects
            min_z_score: Minimum Z-score threshold
            top_n: Number of prospects to return
            
        Returns:
            DataFrame of top prospects with anomaly scores
        """
        all_anomalies = []
        
        # Analyze key metrics
        key_metrics = ['Gls/90', 'Ast/90', 'Sh/90', 'Int/90', 'TklW/90']
        
        for metric in key_metrics:
            if metric not in self.df.columns:
                continue
            
            anomalies = self.detect_anomalies(metric, z_threshold=min_z_score)
            
            # Filter by age
            young_anomalies = [a for a in anomalies if a.age <= max_age]
            all_anomalies.extend(young_anomalies)
        
        if len(all_anomalies) == 0:
            return pd.DataFrame()
        
        # Convert to DataFrame
        prospects_df = pd.DataFrame([
            {
                'Player': a.player_name,
                'Age': a.age,
                'Position': a.position,
                'League': a.league,
                'Metric': a.metric,
                'Value': a.player_value,
                'Age_Mean': a.age_mean,
                'Z_Score': a.z_score,
                'Age_Percentile': a.percentile_rank,
            }
            for a in all_anomalies
        ])
        
        # Aggregate by player (count of high-ceiling metrics)
        player_summary = prospects_df.groupby('Player').agg({
            'Z_Score': 'mean',
            'Metric': 'count',
            'Age': 'first',
            'Position': 'first',
            'League': 'first',
        }).reset_index()
        
        player_summary.columns = ['Player', 'Avg_Z_Score', 'High_Ceiling_Metrics', 'Age', 'Position', 'League']
        player_summary = player_summary.sort_values('Avg_Z_Score', ascending=False)
        
        return player_summary.head(top_n)


def format_age_curve_badge(anomaly: AgeCurveAnomaly) -> str:
    """
    Format a UI badge for age-curve anomaly.
    
    Args:
        anomaly: AgeCurveAnomaly object
        
    Returns:
        HTML/markdown badge string
    """
    if anomaly.z_score >= 3.0:
        return "Elite Prospect (3 standard deviations Above Age Cohort)"
    elif anomaly.z_score >= 2.5:
        return "High-Ceiling Prospect (2.5 standard deviations Above Age Cohort)"
    elif anomaly.z_score >= 2.0:
        return "Ahead of Curve (2 standard deviations Above Age Cohort)"
    else:
        return "Above Average for Age"
