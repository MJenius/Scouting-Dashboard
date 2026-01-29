"""
similarity.py - Advanced player similarity engine with position-weighted matching.

This module provides:
- Cosine similarity calculation with StandardScaler normalization
- Position-specific weighting for Attacker/Midfielder/Defender profiles
- Fuzzy player search with rapidfuzz (handles typos, abbreviations, accents)
- Autocomplete suggestions for player selection
- Multiple similarity modes (raw stats, weighted, percentile-normalized)
- Radar chart visualization for player comparisons

Key components:
- SimilarityEngine: Main class for similarity calculations
- RadarChartGenerator: Plotly and Matplotlib radar chart creation
- Fuzzy search utilities for player name matching
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz

from .constants import FEATURE_COLUMNS, GK_FEATURE_COLUMNS, PROFILE_WEIGHTS, LEAGUES



# ============================================================================
# POSITION & PROFILE ENUMERATIONS
# ============================================================================

class Position(Enum):
    """Player position codes."""
    FORWARD = "FW"
    MIDFIELDER = "MF"
    DEFENDER = "DF"
    GOALKEEPER = "GK"
    UNKNOWN = "UKN"


class ProfileType(Enum):
    """Profile types for weighted similarity."""
    ATTACKER = "Attacker"
    MIDFIELDER = "Midfielder"
    DEFENDER = "Defender"
    GOALKEEPER = "Goalkeeper"


# ============================================================================
# RADAR CHART GENERATOR
# ============================================================================

class RadarChartGenerator:
    """
    Creates Plotly and Matplotlib radar charts for player comparisons.
    
    Supports:
    - Single player profile visualization
    - Target vs best match comparison
    - Dual-player deep-dive comparison
    - Toggle between raw stats and percentiles
    """
    
    def __init__(self):
        self.feature_labels = {
            'Gls/90': 'Goals',
            'Ast/90': 'Assists',
            'Sh/90': 'Shots',
            'SoT/90': 'Shots on Target',
            'Crs/90': 'Crosses',
            'Int/90': 'Interceptions',
            'TklW/90': 'Tackles Won',
            'Fls/90': 'Fouls',
            'Fld/90': 'Fouls Drawn',
        }
        
        self.gk_feature_labels = {
            'GA90': 'Goals Against/90',
            'Save%': 'Save %',
            'CS%': 'Clean Sheets %',
            'W': 'Wins',
            'D': 'Draws',
            'L': 'Losses',
            'PKsv': 'Penalties Saved',
            'PKm': 'Penalties Made',
            'Saves': 'Total Saves',
        }

    
    def generate_plotly_radar(
        self,
        target_stats: Dict[str, float],
        comparison_stats: Optional[Dict[str, float]] = None,
        target_name: str = "Target Player",
        comparison_name: str = "Comparison Player",
        use_percentiles: bool = False,
        save_path: Optional[str] = None,
        is_goalkeeper: bool = False,
    ) -> 'go.Figure':
        """
        Generate interactive Plotly radar chart.
        
        Args:
            target_stats: Dict of feature → value for target player
            comparison_stats: Dict of feature → value for comparison (optional)
            target_name: Name of target player
            comparison_name: Name of comparison player
            use_percentiles: Use percentile values (0-100) vs raw stats
            save_path: Optional path to save HTML version
            is_goalkeeper: Use goalkeeper-specific metrics instead of outfield metrics
            
        Returns:
            Plotly Figure object (can be rendered with st.plotly_chart)
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError("Plotly required for radar charts: pip install plotly")
        
        # Select features based on position
        if is_goalkeeper:
            feature_list = GK_FEATURE_COLUMNS
            labels = self.gk_feature_labels
        else:
            feature_list = FEATURE_COLUMNS
            labels = self.feature_labels
        
        # Prepare data
        categories = [labels.get(feat, feat) for feat in feature_list]
        target_values = [target_stats.get(feat, 0) for feat in feature_list]
        
        # Create figure
        fig = go.Figure()
        
        # Add target trace
        fig.add_trace(go.Scatterpolar(
            r=target_values,
            theta=categories,
            fill='toself',
            name=target_name,
            line=dict(color='#003399'),  # Dark Blue
            fillcolor='rgba(0, 51, 153, 0.3)',
        ))
        
        # Add comparison trace if provided
        if comparison_stats is not None:
            comparison_values = [comparison_stats.get(feat, 0) for feat in feature_list]
            fig.add_trace(go.Scatterpolar(
                r=comparison_values,
                theta=categories,
                fill='toself',
                name=comparison_name,
                line=dict(color='#EE1939'),  # Red
                fillcolor='rgba(238, 25, 57, 0.3)',
            ))
        
        # Update layout
        stat_type = "Percentile Ranks (%)" if use_percentiles else "Per-90 Stats"
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100] if use_percentiles else None,
                ),
            ),
            title=dict(
                text=f"Player Comparison: {target_name}",
                font=dict(size=16, color='#2C3E50'),
            ),
            hovermode='closest',
            font=dict(size=11),
            showlegend=True,
            legend=dict(x=1.1, y=1),
        )
        
        return fig
    
    def generate_matplotlib_radar(
        self,
        target_stats: Dict[str, float],
        comparison_stats: Optional[Dict[str, float]] = None,
        target_name: str = "Target",
        comparison_name: str = "Comparison",
        save_path: str = 'radar_comparison.png',
        dpi: int = 300,
    ) -> str:
        """
        Generate high-resolution Matplotlib radar chart and save as PNG.
        
        Args:
            target_stats: Dict of feature → value for target player
            comparison_stats: Dict of feature → value for comparison (optional)
            target_name: Name of target player
            comparison_name: Name of comparison player
            save_path: Path to save PNG
            dpi: Resolution (default: 300)
            
        Returns:
            str: Path to saved PNG file
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from math import pi
        except ImportError:
            raise ImportError("Matplotlib required: pip install matplotlib")
        
        # Setup
        categories = list(FEATURE_COLUMNS)
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot target
        target_values = [target_stats.get(feat, 0) for feat in categories]
        target_values += target_values[:1]
        ax.plot(angles, target_values, 'o-', linewidth=2, label=target_name, color='#003399')
        ax.fill(angles, target_values, alpha=0.25, color='#003399')
        
        # Plot comparison if provided
        if comparison_stats is not None:
            comparison_values = [comparison_stats.get(feat, 0) for feat in categories]
            comparison_values += comparison_values[:1]
            ax.plot(angles, comparison_values, 'o-', linewidth=2, label=comparison_name, color='#EE1939')
            ax.fill(angles, comparison_values, alpha=0.25, color='#EE1939')
        
        # Formatting
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([self.feature_labels.get(cat, cat) for cat in categories])
        ax.set_ylim(0, 100)
        ax.set_title(f"Player Profile Comparison", size=16, weight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        # Save
        plt.tight_layout()
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return save_path


# ============================================================================
# SIMILARITY ENGINE
# ============================================================================

class SimilarityEngine:
    """
    Advanced player similarity engine with position-weighted matching.
    
    Features:
    - Cosine similarity on normalized features
    - Position-specific weighting (Attacker/Midfielder/Defender)
    - Fuzzy player search with typo tolerance
    - Multiple normalization modes (raw, percentile)
    - Top-N results with match scores
    
    Usage:
        engine = SimilarityEngine(df, scaled_features=scaled_features)
        similar = engine.find_similar_players(
            'Erling Haaland',
            league='Premier League',
            top_n=5,
            use_position_weights=True
        )
    """
    
    FUZZY_MATCH_THRESHOLD = 85  # Minimum match score for fuzzy fallback (increased for better accuracy)
    
    def __init__(
        self,
        df: pd.DataFrame,
        scaled_features: np.ndarray,
        scaler: StandardScaler,
        min_90s: int = 10,
    ):
        """
        Initialize similarity engine.
        
        Args:
            df: Processed player dataset
            scaled_features: Pre-scaled feature array (from data_engine)
            scaler: Fitted StandardScaler object (from data_engine)
            min_90s: Minimum 90s for reliability (already filtered in df)
        """
        self.df = df.copy()
        self.scaled_features = scaled_features
        self.scaler = scaler
        self.min_90s = min_90s
        
        # Validate input
        if len(self.df) != len(self.scaled_features):
            raise ValueError(
                f"DataFrame ({len(self.df)}) and scaled_features "
                f"({len(self.scaled_features)}) length mismatch"
            )
    
    def _find_player_index(self, player_name: str) -> Optional[int]:
        """
        Find player by exact match or fuzzy matching.
        
        Strategy:
        1. Try exact substring match (case-insensitive)
        2. Fall back to fuzzy match (rapidfuzz) with threshold
        3. Return None if not found
        
        Args:
            player_name: Player name to search for
            
        Returns:
            Player dataframe index, or None if not found
        """
        # Exact substring match first (faster)
        exact = self.df[
            self.df['Player'].str.contains(player_name, case=False, na=False)
        ]
        if len(exact) > 0:
            return exact.index[0]
        
        # Fuzzy matching fallback
        best_match = None
        best_score = 0
        
        for idx, player in self.df['Player'].items():
            if pd.isna(player):
                continue
            score = fuzz.ratio(player_name.lower(), player.lower())
            if score > best_score and score >= self.FUZZY_MATCH_THRESHOLD:
                best_score = score
                best_match = idx
        
        return best_match
    
    def get_player_suggestions(
        self,
        input_text: str,
        league: str = 'all',
        limit: int = 10,
    ) -> List[Tuple[str, int]]:
        """
        Generate autocomplete suggestions for player names.
        
        Uses fuzzy matching to rank suggestions by match quality.
        
        Args:
            input_text: Partial player name (e.g., "Haal")
            league: Filter by league ('all' for all leagues)
            limit: Maximum suggestions to return
            
        Returns:
            List of (player_name, match_score) tuples sorted by score (descending)
        """
        # Get candidates
        if league.lower() != 'all':
            if league not in LEAGUES:
                return []
            candidates = self.df[self.df['League'] == league]['Player'].unique()
        else:
            candidates = self.df['Player'].unique()
        
        # Score and sort
        suggestions = []
        for player in candidates:
            if pd.isna(player):
                continue
            score = fuzz.ratio(input_text.lower(), player.lower())
            if score > 0:
                suggestions.append((player, score))
        
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:limit]
    
    def apply_position_weighting(
        self,
        features: np.ndarray,
        profile: ProfileType,
    ) -> np.ndarray:
        """
        Apply position-specific weights to feature vector.
        
        Weights boost certain metrics for each profile:
        - Attacker: Gls/90 (3.0), Sh/90 (2.5), SoT/90 (2.0)
        - Midfielder: Ast/90 (2.5), Crs/90 (2.0), TklW/90 (1.8)
        - Defender: Int/90 (2.5), TklW/90 (2.5), Fld/90 (1.5)
        - Goalkeeper: No position weighting (uses unweighted features)
        
        Args:
            features: Normalized feature array (shape: [n, 9] or [9])
            profile: ProfileType enum (ATTACKER, MIDFIELDER, DEFENDER, GOALKEEPER)
            
        Returns:
            Weighted features (same shape as input)
        """
        # Goalkeepers: skip position weighting since scaled features are based on FEATURE_COLUMNS
        # Goalkeeper comparisons work best with raw similarity across standard metrics
        if profile == ProfileType.GOALKEEPER:
            return features
        
        weights_dict = PROFILE_WEIGHTS[profile.value]
        weights = np.array([weights_dict[col] for col in FEATURE_COLUMNS])
        
        # Normalize weights to maintain magnitude
        weights = weights / weights.mean()
        
        return features * weights
    
    def calculate_feature_attribution(
        self,
        target_player: str,
        comparison_player: str,
        use_position_weights: bool = False,
    ) -> Optional[Dict[str, float]]:
        """
        Calculate which features drive the similarity between two players.
        
        Returns feature-level distances after weighting, showing which stats
        are most similar vs different. Lower distance = more similar.
        
        Args:
            target_player: First player name
            comparison_player: Second player name
            use_position_weights: Apply position-specific weighting
            
        Returns:
            Dict of feature_name -> distance (0-1 scale), ordered by importance,
            or None if players not found
        """
        idx1 = self._find_player_index(target_player)
        idx2 = self._find_player_index(comparison_player)
        
        if idx1 is None or idx2 is None:
            return None
        
        row1 = self.df.loc[idx1]
        row2 = self.df.loc[idx2]
        
        # Get vectors
        row1_idx = self.df.index.get_loc(idx1)
        row2_idx = self.df.index.get_loc(idx2)
        
        vec1 = self.scaled_features[row1_idx].copy()
        vec2 = self.scaled_features[row2_idx].copy()
        
        # Apply weighting if needed
        weights = np.ones(len(FEATURE_COLUMNS))
        if use_position_weights:
            pos1 = row1['Primary_Pos']
            profile_map = {
                'FW': ProfileType.ATTACKER,
                'MF': ProfileType.MIDFIELDER,
                'DF': ProfileType.DEFENDER,
                'GK': ProfileType.GOALKEEPER,
            }
            profile1 = profile_map.get(pos1, ProfileType.MIDFIELDER)
            
            if profile1 != ProfileType.GOALKEEPER:
                weights_dict = PROFILE_WEIGHTS[profile1.value]
                weights = np.array([weights_dict[col] for col in FEATURE_COLUMNS])
                weights = weights / weights.mean()
        
        # Calculate feature-level distances (absolute differences)
        feature_distances = np.abs(vec1 - vec2)
        
        # Weight the distances
        weighted_distances = feature_distances * weights
        
        # Normalize to 0-1 scale
        max_distance = weighted_distances.max()
        if max_distance > 0:
            normalized_distances = weighted_distances / max_distance
        else:
            normalized_distances = weighted_distances
        
        # Build result dict, sorted by distance (most similar first)
        attribution = {}
        for i, feat in enumerate(FEATURE_COLUMNS):
            attribution[feat] = float(normalized_distances[i])
        
        # Sort by distance (ascending = more similar features first)
        attribution = dict(sorted(attribution.items(), key=lambda x: x[1]))
        
        return attribution
    
    def find_similar_players(
        self,
        target_player: str,
        league: str = 'all',
        top_n: int = 5,
        use_position_weights: bool = False,
        use_percentiles: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Find similar players to target.
        
        Args:
            target_player: Player name (supports fuzzy matching)
            league: Filter by league ('all' for all)
            top_n: Number of results (default: 5, max: 20)
            use_position_weights: Apply position-specific weighting
            use_percentiles: Use percentile-normalized stats instead of raw
            
        Returns:
            DataFrame of similar players with Match_Score, or None if target not found
        """
        # Find target
        target_idx = self._find_player_index(target_player)
        if target_idx is None:
            return None
        
        target_row = self.df.loc[target_idx]
        target_position = target_row['Primary_Pos']
        
        # Map position to profile
        profile_map = {
            'FW': ProfileType.ATTACKER,
            'MF': ProfileType.MIDFIELDER,
            'DF': ProfileType.DEFENDER,
            'GK': ProfileType.GOALKEEPER,
        }
        profile = profile_map.get(target_position, ProfileType.MIDFIELDER)

        
        # Build search space
        if league.lower() != 'all':
            if league not in LEAGUES:
                return None
            search_mask = self.df['League'] == league
        else:
            search_mask = pd.Series(True, index=self.df.index)
        
        search_indices = self.df[search_mask].index
        search_features = self.scaled_features[self.df.index.isin(search_indices)]
        
        # Get target features
        target_row_idx = self.df.index.get_loc(target_idx)
        target_vector = self.scaled_features[target_row_idx].reshape(1, -1)
        
        # Apply weighting if requested
        if use_position_weights:
            search_features = self.apply_position_weighting(search_features, profile)
            target_vector = self.apply_position_weighting(target_vector, profile)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(target_vector, search_features)[0]
        
        # Build results
        results = self.df[search_mask].copy()
        results['Match_Score'] = similarities * 100
        
        # Exclude target player from results
        results = results[results.index != target_idx].sort_values(
            'Match_Score', ascending=False
        )
        
        # Limit results
        top_n = min(top_n, 20)  # Cap at 20
        return results.head(top_n)
    
    def compare_players(
        self,
        player1_name: str,
        player2_name: str,
        use_position_weights: bool = False,
    ) -> Dict:
        """
        Deep comparison between two players.
        
        Returns detailed comparison including:
        - Match score (how similar they are)
        - Feature-by-feature comparison
        - Position and league info
        - Percentile ranks (if available)
        
        Args:
            player1_name: First player name
            player2_name: Second player name
            use_position_weights: Apply position-specific weighting
            
        Returns:
            Dict with comparison data or None if players not found
        """
        idx1 = self._find_player_index(player1_name)
        idx2 = self._find_player_index(player2_name)
        
        if idx1 is None or idx2 is None:
            return None
        
        row1 = self.df.loc[idx1]
        row2 = self.df.loc[idx2]
        
        # Get vectors
        row1_idx = self.df.index.get_loc(idx1)
        row2_idx = self.df.index.get_loc(idx2)
        
        vec1 = self.scaled_features[row1_idx].reshape(1, -1)
        vec2 = self.scaled_features[row2_idx].reshape(1, -1)
        
        # Apply weighting if needed
        if use_position_weights:
            pos1 = row1['Primary_Pos']
            profile_map = {
                'FW': ProfileType.ATTACKER,
                'MF': ProfileType.MIDFIELDER,
                'DF': ProfileType.DEFENDER,
                'GK': ProfileType.GOALKEEPER,
            }
            profile1 = profile_map.get(pos1, ProfileType.MIDFIELDER)
            vec1 = self.apply_position_weighting(vec1, profile1)
            vec2 = self.apply_position_weighting(vec2, profile1)
        
        # Calculate similarity
        match_score = cosine_similarity(vec1, vec2)[0][0] * 100
        
        # Build feature comparison
        feature_comparison = {}
        for feat in FEATURE_COLUMNS:
            val1 = row1[feat]
            val2 = row2[feat]
            feature_comparison[feat] = {
                'player1': val1,
                'player2': val2,
                'difference': val2 - val1,
                'player1_better': val1 > val2,
            }
            
            # Add percentiles if available
            pct1_col = f'{feat}_pct'
            pct2_col = f'{feat}_pct'
            if pct1_col in row1.index and pct2_col in row2.index:
                feature_comparison[feat]['player1_pct'] = row1[pct1_col]
                feature_comparison[feat]['player2_pct'] = row2[pct2_col]
        
        return {
            'player1': {
                'name': row1['Player'],
                'league': row1['League'],
                'squad': row1['Squad'],
                'position': row1['Primary_Pos'],
                'age': row1['Age'],
            },
            'player2': {
                'name': row2['Player'],
                'league': row2['League'],
                'squad': row2['Squad'],
                'position': row2['Primary_Pos'],
                'age': row2['Age'],
            },
            'match_score': match_score,
            'feature_comparison': feature_comparison,
        }
    
    def get_player_profile(
        self,
        player_name: str,
        use_percentiles: bool = True,
        is_goalkeeper: bool = False,
    ) -> Optional[Dict[str, float]]:
        """
        Get all metric values for a player (raw or percentile).
        
        Args:
            player_name: Player name
            use_percentiles: Return percentile ranks instead of raw stats
            is_goalkeeper: Return goalkeeper-specific metrics instead of outfield metrics
            
        Returns:
            Dict mapping feature → value, or None if not found
        """
        idx = self._find_player_index(player_name)
        if idx is None:
            return None
        
        row = self.df.loc[idx]
        profile = {}
        
        # Determine which features to use based on position
        if is_goalkeeper or (not is_goalkeeper and row['Primary_Pos'] == 'GK'):
            features_to_use = GK_FEATURE_COLUMNS
        else:
            features_to_use = FEATURE_COLUMNS
        
        for feat in features_to_use:
            if feat not in row.index:
                # Feature doesn't exist in data, use 0 or skip
                profile[feat] = 0
            elif use_percentiles:
                pct_col = f'{feat}_pct'
                profile[feat] = row[pct_col] if pct_col in row.index else 0
            else:
                profile[feat] = row[feat]
        
        return profile
