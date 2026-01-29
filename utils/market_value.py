"""
market_value.py - Transfer market value estimation and price vs performance analysis.

This module provides:
- Estimated market value calculation based on age, league, performance percentiles
- Price vs Performance outlier detection (undervalued/overvalued players)
- Transfer value modeling using multiple factors
- Market inefficiency identification

Key components:
- MarketValueEstimator: Calculate estimated transfer values
- PricePerformanceAnalyzer: Find bargains and overpriced players
- Value tier classification (Bargain, Good Value, Fair, Expensive, Overpriced)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .constants import LEAGUE_TIERS, FEATURE_COLUMNS


class MarketValueEstimator:
    """
    Estimates player market values using a multi-factor model.
    
    Factors considered:
    - Age (peak value at 24-26)
    - League tier (Premier League premium)
    - Performance percentiles (weighted average)
    - Archetype (position-specific adjustments)
    - Playing time (90s as reliability indicator)
    - Output metrics (goals, assists weighted heavily)
    
    Value ranges by league:
    - Premier League: £0.5M - £80M
    - Championship: £0.2M - £15M
    - League One: £0.05M - £3M
    - League Two: £0.02M - £0.8M
    - National League: £0.01M - £0.3M
    """
    
    # Base value ranges by league (in millions £)
    LEAGUE_BASE_VALUES = {
        'Premier League': {'min': 0.5, 'median': 8.0, 'max': 80.0},
        'Championship': {'min': 0.2, 'median': 1.5, 'max': 15.0},
        'League One': {'min': 0.05, 'median': 0.3, 'max': 3.0},
        'League Two': {'min': 0.02, 'median': 0.15, 'max': 0.8},
        'National League': {'min': 0.01, 'median': 0.08, 'max': 0.3},
    }
    
    # Age multipliers (peak at 24-26)
    AGE_MULTIPLIERS = {
        (16, 19): 1.3,   # Young prospect premium
        (20, 22): 1.5,   # High potential
        (23, 26): 1.8,   # Peak years
        (27, 29): 1.3,   # Experienced
        (30, 32): 0.8,   # Decline phase
        (33, 40): 0.4,   # Veteran discount
    }
    
    # Position multipliers (attackers generally more valuable)
    POSITION_MULTIPLIERS = {
        'FW': 1.4,
        'MF': 1.0,
        'DF': 0.8,
        'GK': 0.7,
    }
    
    def __init__(self):
        """Initialize the market value estimator."""
        pass
    
    def estimate_value(self, player_data: pd.Series) -> Dict[str, float]:
        """
        Estimate market value for a player.
        
        Args:
            player_data: Player row from DataFrame with all stats
            
        Returns:
            Dict with value estimates:
                - estimated_value: Primary estimate in £M
                - min_value: Conservative estimate
                - max_value: Optimistic estimate
                - confidence: Confidence score (0-100)
        """
        # Extract key data with fallbacks for missing values
        try:
            age = int(player_data['Age']) if not pd.isna(player_data['Age']) else 25
        except (ValueError, KeyError):
            age = 25  # Default age if missing
        
        league = player_data.get('League', 'National League')
        position = player_data.get('Primary_Pos', 'MF')
        minutes = player_data.get('90s', 0)
        
        # Get base value from league
        base_ranges = self.LEAGUE_BASE_VALUES.get(league, self.LEAGUE_BASE_VALUES['National League'])
        base_value = base_ranges['median']
        
        # Apply age multiplier
        age_mult = self._get_age_multiplier(age)
        
        # Apply position multiplier
        pos_mult = self.POSITION_MULTIPLIERS.get(position, 1.0)
        
        # Calculate performance multiplier from percentiles
        perf_mult = self._calculate_performance_multiplier(player_data, position)
        
        # Calculate playing time factor (reliability)
        playing_time_mult = self._calculate_playing_time_multiplier(minutes)
        
        # Calculate output bonus (goals + assists are premium)
        output_bonus = self._calculate_output_bonus(player_data)
        
        # Combine all factors
        estimated_value = (
            base_value * 
            age_mult * 
            pos_mult * 
            perf_mult * 
            playing_time_mult * 
            (1 + output_bonus)
        )
        
        # Apply bounds
        estimated_value = max(base_ranges['min'], min(estimated_value, base_ranges['max']))
        
        # Calculate range
        min_value = estimated_value * 0.7  # 30% lower
        max_value = estimated_value * 1.5  # 50% higher
        
        # Confidence score based on data completeness
        confidence = self._calculate_confidence(player_data, minutes)
        
        return {
            'estimated_value': round(estimated_value, 2),
            'min_value': round(min_value, 2),
            'max_value': round(max_value, 2),
            'confidence': round(confidence, 1),
        }
    
    def _get_age_multiplier(self, age: int) -> float:
        """Get age-based multiplier."""
        for (min_age, max_age), mult in self.AGE_MULTIPLIERS.items():
            if min_age <= age <= max_age:
                return mult
        return 0.3  # Outside normal age ranges
    
    def _calculate_performance_multiplier(self, player_data: pd.Series, position: str) -> float:
        """
        Calculate performance multiplier from percentile rankings.
        
        Uses weighted average of relevant percentiles:
        - 90th+ percentile: 2.5x
        - 80-89th percentile: 2.0x
        - 70-79th percentile: 1.5x
        - 60-69th percentile: 1.2x
        - 50-59th percentile: 1.0x
        - Below 50th: 0.5-0.9x
        """
        percentiles = []
        
        # Weight key metrics by position
        key_metrics = self._get_key_metrics_for_position(position)
        
        for metric in key_metrics:
            pct_col = f'{metric}_pct'
            if pct_col in player_data.index and not pd.isna(player_data[pct_col]):
                percentiles.append(player_data[pct_col])
        
        if not percentiles:
            return 1.0
        
        avg_pct = np.mean(percentiles)
        
        # Convert percentile to multiplier
        if avg_pct >= 90:
            return 2.5
        elif avg_pct >= 80:
            return 2.0
        elif avg_pct >= 70:
            return 1.5
        elif avg_pct >= 60:
            return 1.2
        elif avg_pct >= 50:
            return 1.0
        else:
            return 0.5 + (avg_pct / 100)  # 0.5 to 1.0 range
    
    def _get_key_metrics_for_position(self, position: str) -> List[str]:
        """Get the most valuable metrics for market value by position."""
        key_metrics = {
            'FW': ['Gls/90', 'Ast/90', 'Sh/90', 'SoT/90'],
            'MF': ['Ast/90', 'Gls/90', 'Crs/90', 'Int/90', 'TklW/90'],
            'DF': ['Int/90', 'TklW/90', 'Crs/90'],
            'GK': [],  # Would use Save%, GA90 if available
        }
        return key_metrics.get(position, FEATURE_COLUMNS)
    
    def _calculate_playing_time_multiplier(self, minutes: float) -> float:
        """
        Playing time reliability multiplier.
        
        - 25+ matches: 1.2x (proven starter)
        - 20-24 matches: 1.1x (regular)
        - 15-19 matches: 1.0x (rotation)
        - 10-14 matches: 0.9x (squad player)
        - < 10 matches: 0.7x (limited sample)
        """
        if minutes >= 25:
            return 1.2
        elif minutes >= 20:
            return 1.1
        elif minutes >= 15:
            return 1.0
        elif minutes >= 10:
            return 0.9
        else:
            return 0.7
    
    def _calculate_output_bonus(self, player_data: pd.Series) -> float:
        """
        Calculate bonus for direct goal contributions.
        
        Goals and assists command premium in transfer market.
        """
        goals = player_data.get('Gls/90', 0)
        assists = player_data.get('Ast/90', 0)
        
        # Calculate total output
        total_output = goals + (assists * 0.7)  # Goals worth slightly more
        
        # Bonus scaling
        if total_output >= 0.8:
            return 0.5  # 50% bonus for elite output
        elif total_output >= 0.5:
            return 0.3  # 30% bonus for good output
        elif total_output >= 0.3:
            return 0.15  # 15% bonus for decent output
        else:
            return 0.0  # No bonus
    
    def _calculate_confidence(self, player_data: pd.Series, minutes: float) -> float:
        """
        Calculate confidence in the estimate (0-100).
        
        Higher confidence with:
        - More playing time
        - Better data completeness
        - More relevant percentile data
        """
        # Base confidence from playing time
        if minutes >= 20:
            time_conf = 90
        elif minutes >= 15:
            time_conf = 80
        elif minutes >= 10:
            time_conf = 70
        else:
            time_conf = 50
        
        # Data completeness bonus
        completeness = player_data.get('Completeness_Score', 50)
        data_conf = completeness * 0.3  # Max 30 points
        
        # Combine
        total_conf = min(100, time_conf + data_conf)
        
        return total_conf


class PricePerformanceAnalyzer:
    """
    Analyzes price vs performance to find market inefficiencies.
    
    Identifies:
    - Bargain players (undervalued relative to performance)
    - Overpriced players (high value, low performance)
    - Fair value players (price matches performance)
    - Value tier classification
    
    Value Score = Performance Percentile / (Estimated Value ^ 0.5)
    Higher score = better value
    """
    
    def __init__(self, estimator: Optional[MarketValueEstimator] = None):
        """
        Initialize analyzer.
        
        Args:
            estimator: MarketValueEstimator instance (creates one if None)
        """
        self.estimator = estimator or MarketValueEstimator()
    
    def analyze_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market value and value score columns to DataFrame.
        
        Args:
            df: Player DataFrame with stats and percentiles
            
        Returns:
            DataFrame with added columns:
                - Estimated_Value_£M: Estimated market value
                - Value_Score: Performance per £ metric
                - Value_Tier: Classification (Bargain/Good Value/Fair/Expensive)
                - Value_Rank: Ranking by value score (1 = best value)
        """
        df = df.copy()
        
        # Calculate estimated values
        values = []
        for idx, row in df.iterrows():
            est = self.estimator.estimate_value(row)
            values.append(est['estimated_value'])
        
        df['Estimated_Value_£M'] = values
        
        # Calculate value score
        df['Value_Score'] = self._calculate_value_score(df)
        
        # Assign value tiers
        df['Value_Tier'] = df['Value_Score'].apply(self._assign_value_tier)
        
        # Rank by value
        df['Value_Rank'] = df['Value_Score'].rank(ascending=False, method='min').astype(int)
        
        return df
    
    def _calculate_value_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate value score: Performance / sqrt(Price).
        
        Uses average percentile as performance proxy.
        Applies sqrt to price to dampen the effect of very cheap players.
        """
        # Calculate average percentile for each player
        percentile_cols = [col for col in df.columns if col.endswith('_pct')]
        avg_percentiles = df[percentile_cols].mean(axis=1)
        
        # Calculate value score
        # Add small constant to avoid division by zero
        value_scores = avg_percentiles / (np.sqrt(df['Estimated_Value_£M']) + 0.01)
        
        return value_scores
    
    def _assign_value_tier(self, value_score: float) -> str:
        """Assign value tier based on value score."""
        if pd.isna(value_score):
            return 'Unknown'
        elif value_score >= 15:
            return '💎 Bargain'
        elif value_score >= 10:
            return '✅ Good Value'
        elif value_score >= 7:
            return '➖ Fair'
        elif value_score >= 5:
            return '⚠️ Expensive'
        else:
            return '❌ Overpriced'
    
    def find_best_value_players(
        self,
        df: pd.DataFrame,
        league: Optional[str] = None,
        position: Optional[str] = None,
        max_value: Optional[float] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Find the best value players based on criteria.
        
        Args:
            df: DataFrame with value analysis completed
            league: Filter by league (optional)
            position: Filter by position (optional)
            max_value: Maximum estimated value in £M (optional)
            top_n: Number of results to return
            
        Returns:
            DataFrame of top value players sorted by Value_Score
        """
        result = df.copy()
        
        # Apply filters
        if league:
            result = result[result['League'] == league]
        
        if position:
            result = result[result['Primary_Pos'] == position]
        
        if max_value:
            result = result[result['Estimated_Value_£M'] <= max_value]
        
        # Sort by value score and return top N
        result = result.sort_values('Value_Score', ascending=False).head(top_n)
        
        return result
    
    def find_overpriced_players(
        self,
        df: pd.DataFrame,
        league: Optional[str] = None,
        min_value: float = 1.0,
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Find potentially overpriced players.
        
        Args:
            df: DataFrame with value analysis completed
            league: Filter by league (optional)
            min_value: Minimum estimated value in £M (filter noise)
            top_n: Number of results to return
            
        Returns:
            DataFrame of overpriced players sorted by Value_Score ascending
        """
        result = df.copy()
        
        # Filter by minimum value (avoid noise from very cheap players)
        result = result[result['Estimated_Value_£M'] >= min_value]
        
        if league:
            result = result[result['League'] == league]
        
        # Sort by value score ascending (lowest = worst value)
        result = result.sort_values('Value_Score', ascending=True).head(top_n)
        
        return result


def add_market_value_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add market value analysis to a DataFrame.
    
    Args:
        df: Player DataFrame with stats and percentiles
        
    Returns:
        DataFrame with market value columns added
    """
    analyzer = PricePerformanceAnalyzer()
    return analyzer.analyze_value(df)
