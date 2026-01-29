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
    Estimates player market values using a multi-factor model with league normalization.
    
    Key improvements:
    - League-adjusted percentile interpretation (League Two 80th ≠ PL 80th)
    - Position-specific performance weighting
    - Age curves calibrated by league
    - Playing time adjusted by league standards
    """
    
    # League tier multipliers (relative to Premier League = 1.0)
    LEAGUE_MULTIPLIERS = {
        'Premier League': 1.0,
        'Championship': 0.25,
        'League One': 0.08,
        'League Two': 0.025,
        'National League': 0.01,
        'La Liga': 0.9,
        'Serie A': 0.85,
        'Ligue 1': 0.8,
        'Bundesliga': 0.88,
    }
    
    # League-specific base values (in millions £)
    LEAGUE_BASE_VALUES = {
        'Premier League': {'min': 0.5, 'median': 12.0, 'max': 120.0},
        'Championship': {'min': 0.15, 'median': 2.0, 'max': 20.0},
        'League One': {'min': 0.05, 'median': 0.4, 'max': 4.0},
        'League Two': {'min': 0.02, 'median': 0.12, 'max': 1.2},
        'National League': {'min': 0.01, 'median': 0.06, 'max': 0.5},
        'La Liga': {'min': 0.6, 'median': 10.0, 'max': 100.0},
        'Serie A': {'min': 0.5, 'median': 9.0, 'max': 90.0},
        'Ligue 1': {'min': 0.4, 'median': 8.0, 'max': 80.0},
        'Bundesliga': {'min': 0.5, 'median': 10.0, 'max': 95.0},
    }
    
    # League-specific age multipliers
    LEAGUE_AGE_MULTIPLIERS = {
        'Premier League': {
            (16, 19): 1.6,   # Young prospects premium higher in PL
            (20, 22): 1.8,
            (23, 26): 2.0,   # Peak
            (27, 29): 1.5,
            (30, 32): 0.9,
            (33, 40): 0.4,
        },
        'Championship': {
            (16, 19): 1.3,
            (20, 22): 1.5,
            (23, 26): 1.7,
            (27, 29): 1.2,
            (30, 32): 0.8,
            (33, 40): 0.35,
        },
        'League One': {
            (16, 19): 1.2,
            (20, 22): 1.3,
            (23, 26): 1.4,
            (27, 29): 1.0,
            (30, 32): 0.7,
            (33, 40): 0.3,
        },
        'League Two': {
            (16, 19): 1.1,
            (20, 22): 1.2,
            (23, 26): 1.2,
            (27, 29): 0.9,
            (30, 32): 0.6,
            (33, 40): 0.25,
        },
        'National League': {
            (16, 19): 1.0,
            (20, 22): 1.1,
            (23, 26): 1.1,
            (27, 29): 0.8,
            (30, 32): 0.5,
            (33, 40): 0.2,
        },
    }
    
    # Position multipliers (relative to midfielder = 1.0)
    POSITION_MULTIPLIERS = {
        'FW': 1.3,  # Reduced from 1.4
        'MF': 1.0,
        'DF': 0.85,  # Increased from 0.8
        'GK': 0.75,  # Increased from 0.7
    }
    
    # League-specific playing time thresholds
    LEAGUE_PLAYING_TIME_THRESHOLDS = {
        'Premier League': {'high': 30, 'regular': 25, 'rotation': 18, 'squad': 12},
        'Championship': {'high': 35, 'regular': 28, 'rotation': 20, 'squad': 14},
        'League One': {'high': 35, 'regular': 28, 'rotation': 20, 'squad': 14},
        'League Two': {'high': 40, 'regular': 30, 'rotation': 22, 'squad': 15},
        'National League': {'high': 40, 'regular': 30, 'rotation': 22, 'squad': 15},
    }
    
    def __init__(self):
        """Initialize the market value estimator."""
        pass
    
    def estimate_value(self, player_data: pd.Series) -> Dict[str, float]:
        """
        Estimate market value for a player with league normalization.
        
        Args:
            player_data: Player row from DataFrame with all stats
            
        Returns:
            Dict with value estimates
        """
        try:
            age = int(player_data['Age']) if not pd.isna(player_data['Age']) else 25
        except (ValueError, KeyError):
            age = 25
        
        league = player_data.get('League', 'National League')
        position = player_data.get('Primary_Pos', 'MF')
        minutes = player_data.get('90s', 0)
        
        # Get base value from league
        base_ranges = self.LEAGUE_BASE_VALUES.get(league, self.LEAGUE_BASE_VALUES['National League'])
        base_value = base_ranges['median']
        
        # Apply league-specific age multiplier
        age_mult = self._get_age_multiplier(age, league)
        
        # Apply position multiplier
        pos_mult = self.POSITION_MULTIPLIERS.get(position, 1.0)
        
        # Calculate league-adjusted performance multiplier
        perf_mult = self._calculate_performance_multiplier(player_data, position, league)
        
        # Calculate league-adjusted playing time factor
        playing_time_mult = self._calculate_playing_time_multiplier(minutes, league)
        
        # Calculate output bonus with league context
        output_bonus = self._calculate_output_bonus(player_data, league)
        
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
        min_value = estimated_value * 0.7
        max_value = estimated_value * 1.5
        
        confidence = self._calculate_confidence(player_data, minutes)
        
        return {
            'estimated_value': round(estimated_value, 2),
            'min_value': round(min_value, 2),
            'max_value': round(max_value, 2),
            'confidence': round(confidence, 1),
        }
    
    def _get_age_multiplier(self, age: int, league: str) -> float:
        """Get league-specific age multiplier."""
        multipliers = self.LEAGUE_AGE_MULTIPLIERS.get(league, self.LEAGUE_AGE_MULTIPLIERS['National League'])
        for (min_age, max_age), mult in multipliers.items():
            if min_age <= age <= max_age:
                return mult
        return 0.2
    
    def _calculate_performance_multiplier(self, player_data: pd.Series, position: str, league: str) -> float:
        """
        Calculate league-adjusted performance multiplier.
        
        A 75th percentile in League Two = lower multiplier than PL.
        Uses league tier to scale percentile impact.
        """
        percentiles = []
        key_metrics = self._get_key_metrics_for_position(position)
        
        for metric in key_metrics:
            pct_col = f'{metric}_pct'
            if pct_col in player_data.index and not pd.isna(player_data[pct_col]):
                percentiles.append(player_data[pct_col])
        
        if not percentiles:
            return 1.0
        
        avg_pct = np.mean(percentiles)
        league_mult = self.LEAGUE_MULTIPLIERS.get(league, 0.01)
        
        # League-adjusted percentile impact
        # PL multiplier scaling is more aggressive than lower leagues
        if league == 'Premier League':
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
                return 0.5 + (avg_pct / 100)
        
        # Championship
        elif league == 'Championship':
            if avg_pct >= 90:
                return 2.0
            elif avg_pct >= 80:
                return 1.6
            elif avg_pct >= 70:
                return 1.3
            elif avg_pct >= 60:
                return 1.1
            elif avg_pct >= 50:
                return 0.95
            else:
                return 0.5 + (avg_pct / 150)
        
        # Lower leagues
        else:
            if avg_pct >= 90:
                return 1.5
            elif avg_pct >= 80:
                return 1.25
            elif avg_pct >= 70:
                return 1.1
            elif avg_pct >= 60:
                return 1.0
            elif avg_pct >= 50:
                return 0.9
            else:
                return 0.5 + (avg_pct / 200)
    
    def _get_key_metrics_for_position(self, position: str) -> List[str]:
        """Get position-specific key metrics."""
        key_metrics = {
            'FW': ['Gls/90', 'Ast/90', 'Sh/90', 'SoT/90'],
            'MF': ['Ast/90', 'Gls/90', 'Crs/90', 'Int/90', 'TklW/90'],
            'DF': ['Int/90', 'TklW/90', 'Crs/90'],
            'GK': [],
        }
        return key_metrics.get(position, FEATURE_COLUMNS)
    
    def _calculate_playing_time_multiplier(self, minutes: float, league: str) -> float:
        """
        League-adjusted playing time multiplier.
        Higher thresholds for lower leagues (shorter seasons).
        """
        thresholds = self.LEAGUE_PLAYING_TIME_THRESHOLDS.get(league, self.LEAGUE_PLAYING_TIME_THRESHOLDS['National League'])
        
        if minutes >= thresholds['high']:
            return 1.2
        elif minutes >= thresholds['regular']:
            return 1.1
        elif minutes >= thresholds['rotation']:
            return 1.0
        elif minutes >= thresholds['squad']:
            return 0.9
        else:
            return 0.7
    
    def _calculate_output_bonus(self, player_data: pd.Series, league: str) -> float:
        """
        Calculate output bonus with league context.
        Output is worth more in top leagues.
        """
        goals = player_data.get('Gls/90', 0)
        assists = player_data.get('Ast/90', 0)
        total_output = goals + (assists * 0.7)
        
        if league == 'Premier League':
            if total_output >= 0.8:
                return 0.5
            elif total_output >= 0.5:
                return 0.3
            elif total_output >= 0.3:
                return 0.15
        elif league == 'Championship':
            if total_output >= 0.9:
                return 0.4
            elif total_output >= 0.6:
                return 0.25
            elif total_output >= 0.4:
                return 0.12
        else:
            if total_output >= 1.0:
                return 0.3
            elif total_output >= 0.7:
                return 0.18
            elif total_output >= 0.5:
                return 0.08
        
        return 0.0
    
    def _calculate_confidence(self, player_data: pd.Series, minutes: float) -> float:
        """Calculate confidence in the estimate."""
        if minutes >= 25:
            time_conf = 90
        elif minutes >= 18:
            time_conf = 80
        elif minutes >= 12:
            time_conf = 70
        else:
            time_conf = 50
        
        completeness = player_data.get('Completeness_Score', 50)
        data_conf = completeness * 0.3
        
        return min(100, time_conf + data_conf)


class PricePerformanceAnalyzer:
    """
    Analyzes price vs performance with league-tier awareness.
    
    Improved value score prevents cross-league noise.
    """
    
    def __init__(self, estimator: Optional[MarketValueEstimator] = None):
        """Initialize analyzer."""
        self.estimator = estimator or MarketValueEstimator()
    
    def analyze_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market value and value score columns.
        """
        df = df.copy()
        
        # Calculate estimated values
        values = []
        for idx, row in df.iterrows():
            est = self.estimator.estimate_value(row)
            values.append(est['estimated_value'])
        
        df['Estimated_Value_£M'] = values
        
        # Calculate league-adjusted value score
        df['Value_Score'] = self._calculate_value_score(df)
        
        # Assign value tiers
        df['Value_Tier'] = df['Value_Score'].apply(self._assign_value_tier)
        
        # Rank by value within league
        df['Value_Rank'] = df.groupby('League')['Value_Score'].rank(ascending=False, method='min').astype(int)
        
        return df
    
    def _calculate_value_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate league-tier-aware value score.
        
        Prevents cheap League Two players from appearing "better value" than expensive PL players.
        Uses percentile ranking within league as numerator.
        """
        percentile_cols = [col for col in df.columns if col.endswith('_pct')]
        
        if not percentile_cols:
            # Fallback if no percentiles
            return pd.Series([10] * len(df), index=df.index)
        
        # Calculate average percentile within each league
        value_scores = []
        
        for league in df['League'].unique():
            league_df = df[df['League'] == league]
            
            # Get average percentiles for players in this league
            avg_percentiles = league_df[percentile_cols].mean(axis=1)
            
            # Get percentile rank within league (1-100)
            percentile_rank = avg_percentiles.rank(pct=True) * 100
            
            # Value score = percentile rank / sqrt(value)
            # This creates meaningful comparison within league
            league_value_scores = percentile_rank / (np.sqrt(league_df['Estimated_Value_£M']) + 0.01)
            
            value_scores.extend(league_value_scores.values)
        
        result = pd.Series(value_scores, index=df.index)
        return result
    
    def _assign_value_tier(self, value_score: float) -> str:
        """Assign value tier based on value score."""
        if pd.isna(value_score):
            return 'Unknown'
        elif value_score >= 20:
            return '💎 Bargain'
        elif value_score >= 15:
            return '✅ Good Value'
        elif value_score >= 10:
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
        """Find the best value players."""
        result = df.copy()
        
        if league:
            result = result[result['League'] == league]
        
        if position:
            result = result[result['Primary_Pos'] == position]
        
        if max_value:
            result = result[result['Estimated_Value_£M'] <= max_value]
        
        result = result.sort_values('Value_Score', ascending=False).head(top_n)
        
        return result
    
    def find_overpriced_players(
        self,
        df: pd.DataFrame,
        league: Optional[str] = None,
        min_value: float = 1.0,
        top_n: int = 20
    ) -> pd.DataFrame:
        """Find potentially overpriced players."""
        result = df.copy()
        result = result[result['Estimated_Value_£M'] >= min_value]
        
        if league:
            result = result[result['League'] == league]
        
        result = result.sort_values('Value_Score', ascending=True).head(top_n)
        
        return result


def add_market_value_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function to add market value analysis."""
    analyzer = PricePerformanceAnalyzer()
    return analyzer.analyze_value(df)
