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

from .constants import (
    FEATURE_COLUMNS, 
    GK_FEATURE_COLUMNS, 
    PROFILE_WEIGHTS, 
    LEAGUES, 
    RADAR_LABELS,
    LEAGUE_METRIC_MAP
)



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
        self.labels = RADAR_LABELS

    
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
        else:
            feature_list = FEATURE_COLUMNS
        
        labels = self.labels
        
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
        is_gk = any(feat in target_stats for feat in GK_FEATURE_COLUMNS)
        categories = list(GK_FEATURE_COLUMNS if is_gk else FEATURE_COLUMNS)
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
        ax.set_xticklabels([self.labels.get(cat, cat) for cat in categories])
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
    
    FUZZY_MATCH_THRESHOLD = 75  # Lower threshold to catch more variations (accents, typos)
    
    def __init__(
        self,
        df: pd.DataFrame,
        scaled_features: np.ndarray,
        scalers: Dict[str, StandardScaler],
        min_90s: int = 10,
    ):
        """
        Initialize similarity engine.
        
        Args:
            df: Processed player dataset
            scaled_features: Pre-scaled feature array (from data_engine)
            scalers: Dict of Fitted StandardScaler objects (from data_engine)
            min_90s: Minimum 90s for reliability (already filtered in df)
        """
        self.df = df.copy()
        self.scaled_features = scaled_features
        self.scalers = scalers
        self.min_90s = min_90s
        
        # Validate input
        if len(self.df) != len(self.scaled_features):
            raise ValueError(
                f"DataFrame ({len(self.df)}) and scaled_features "
                f"({len(self.scaled_features)}) length mismatch"
            )
    
    def _find_player_index(self, player_name: str) -> Optional[int]:
        """
        Find player by exact match, descriptive match (Name (Squad)), or fuzzy matching.
        """
        if not player_name:
            return None

        # 1. Descriptive match "Name (Squad)" - handles ambiguity perfectly
        if " (" in player_name and player_name.endswith(")"):
            try:
                name_part = player_name.split(" (")[0].strip()
                squad_part = player_name.split(" (")[1].replace(")", "").strip()
                match = self.df[
                    (self.df['Player'].str.casefold() == name_part.casefold()) &
                    (self.df['Squad'].str.casefold() == squad_part.casefold())
                ]
                if not match.empty:
                    return match.index[0]
            except:
                pass

        # 2. Exact match (case-insensitive)
        exact_match = self.df[self.df['Player'].str.casefold() == player_name.casefold()]
        if len(exact_match) > 0:
            return exact_match.index[0]

        # 3. Exact substring match (case-insensitive)
        substring = self.df[
            self.df['Player'].str.contains(player_name, case=False, na=False)
        ]
        if len(substring) > 0:
            # Prioritize matches that START with the string or are shorter/more exact
            substring = substring.assign(
                len=substring['Player'].str.len(),
                starts=substring['Player'].str.lower().str.startswith(player_name.lower())
            ).sort_values(['starts', 'len'], ascending=[False, True])
            return substring.index[0]
        
        # 4. Fuzzy matching fallback
        best_match = None
        best_score = 0
        
        # Only fuzzy match against a subset for performance if large
        search_space = self.df.head(1000) if len(self.df) > 2000 else self.df
        for idx, player in search_space['Player'].items():
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
        Generate autocomplete suggestions using "Player (Squad)" format for clarity.
        """
        if not input_text:
            return []

        # Filter by league
        if league.lower() != 'all':
            mask = self.df['League'] == league
            candidates = self.df[mask]
        else:
            candidates = self.df
        
        # We want to search in both Name and Squad
        results = []
        for idx, row in candidates.iterrows():
            name = row['Player']
            squad = row['Squad']
            if pd.isna(name): continue
            
            # Use token_set_ratio for better substring matching in names
            score = fuzz.token_set_ratio(input_text.lower(), name.lower())
            
            # Bonus for starting with the input
            if name.lower().startswith(input_text.lower()):
                score += 20
                
            if score >= 60: # Threshold for suggestions
                results.append((f"{name} ({squad})", score))
        
        # Sort and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def apply_position_weighting(
        self,
        features: np.ndarray,
        profile: ProfileType,
        target_indices: Optional[List[int]] = None,
        custom_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Applies importance weighting to feature vectors based on position.
        """
        # Determine vector sections
        outfield_len = len(FEATURE_COLUMNS)
        gk_len = len(GK_FEATURE_COLUMNS)
        total_len = outfield_len + gk_len
        
        # Handle full 22-length vector from the combined matrix
        is_full_vector = features.shape[1] == total_len
        
        if custom_weights is not None:
            # Use provided custom weights (normalized)
            weights = custom_weights / custom_weights.mean()
            return features * weights
            
        # Select correct feature set for weighting defaults
        is_gk = profile == ProfileType.GOALKEEPER
        feature_set = GK_FEATURE_COLUMNS if is_gk else FEATURE_COLUMNS
        
        # Default position weights
        weights_dict = PROFILE_WEIGHTS[profile.value]
        base_weights = np.array([weights_dict.get(col, 1.0) for col in feature_set])
        
        # Apply strength-based bias if target_indices provided
        if target_indices:
            for idx in target_indices:
                if idx < len(base_weights):
                    base_weights[idx] *= 3.0 # Triple weight for strengths
        
        # Create final weight vector matching input shape
        if is_full_vector:
            final_weights = np.ones(total_len)
            if is_gk:
                final_weights[outfield_len:] = base_weights
            else:
                final_weights[:outfield_len] = base_weights
        else:
            final_weights = base_weights
            
        # Normalize weights (only the active part to keep scale consistent)
        active_part = final_weights[outfield_len:] if (is_full_vector and is_gk) else \
                      final_weights[:outfield_len] if (is_full_vector and not is_gk) else \
                      final_weights
        
        # Avoid division by zero
        active_mean = active_part.mean() if active_part.mean() != 0 else 1.0
        final_weights = final_weights / active_mean
        
        return features * final_weights
    
    def calculate_feature_attribution(
        self,
        target_player: str,
        comparison_player: str,
        use_position_weights: bool = True,
    ) -> Optional[Dict[str, float]]:
        """
        Calculates feature-wise distances, sorted by the target player's strengths.
        This ensures the 'Drivers' shown are the player's most important/best stats.
        """
        idx1 = self._find_player_index(target_player)
        idx2 = self._find_player_index(comparison_player)
        
        if idx1 is None or idx2 is None:
            return None
        
        row1 = self.df.loc[idx1]
        row2 = self.df.loc[idx2]
        is_gk = row1['Primary_Pos'] == 'GK'
        col_slice = slice(len(FEATURE_COLUMNS), None) if is_gk else slice(0, len(FEATURE_COLUMNS))
        feature_set = GK_FEATURE_COLUMNS if is_gk else FEATURE_COLUMNS
        
        if is_gk:
            # GKs use their own full metric set, assuming availability
            tracked1 = set(GK_FEATURE_COLUMNS)
            tracked2 = set(GK_FEATURE_COLUMNS)
        else:
            CORE_9 = ['Gls/90', 'Ast/90', 'Sh/90', 'SoT/90', 'Crs/90', 'Int/90', 'TklW/90', 'Fls/90', 'Fld/90']
            tracked1 = set(LEAGUE_METRIC_MAP.get(row1['League'], CORE_9))
            tracked2 = set(LEAGUE_METRIC_MAP.get(row2['League'], CORE_9))
        
        shared = tracked1.intersection(tracked2)
        
        vec1 = self.scaled_features[self.df.index.get_loc(idx1), col_slice]
        vec2 = self.scaled_features[self.df.index.get_loc(idx2), col_slice]
        
        # We calculate attribution based on "Priority Distance"
        # Drivers are shared metrics where the target player is strong
        drivers_priority = {}
        distances = {}
        
        for i, feat in enumerate(feature_set):
            if feat in shared:
                # Raw distance in Z-scores
                distances[feat] = abs(vec1[i] - vec2[i])
                
                # Priority for sorting: (Target Percentile * Position Weight)
                # This ensures the list starts with the target's best/most important stats
                pct = row1.get(f"{feat}_pct", 50)
                pos_bonus = 1.0
                if not is_gk and use_position_weights:
                    profile_map = {'FW': 'Attacker', 'MF': 'Midfielder', 'DF': 'Defender'}
                    profile_key = profile_map.get(row1['Primary_Pos'], 'Midfielder')
                    pos_bonus = PROFILE_WEIGHTS[profile_key].get(feat, 1.0)
                
                drivers_priority[feat] = pct * pos_bonus
        
                
                drivers_priority[feat] = pct * pos_bonus
        
        # Sort by "Shared Excellence" Score
        # Previously just sorted by distance. Now we want: High Target Pct + High Match Pct + Low Distance
        # We construct a synthetic score for sorting purposes
        match_scores = {}
        for feat in drivers_priority.keys():
            p1 = row1.get(f"{feat}_pct", 50)
            p2 = row2.get(f"{feat}_pct", 50)
            dist = distances.get(feat, 100)
            
            # Score = Combined Percentile (max 200) - Penalty for Distance
            # If distance is small, score is high. If percentiles are high, score is bigger.
            # Example: 90+90, dist 0.1 => 180 - 2.5 = 177.5
            # Example: 50+50, dist 0.1 => 100 - 2.5 = 97.5 (Elite stats win)
            # Example: 90+20, dist 2.0 => 110 - 50 = 60 (Mismatch loses)
            score = (p1 + p2) - (dist * 25)
            match_scores[feat] = score

        # Sort features by this excellence score descending
        sorted_feats = sorted(match_scores.keys(), key=lambda x: match_scores[x], reverse=True)
        
        # Return distances in the sorted order (Top items are the Drivers)
        return {feat: distances[feat] for feat in sorted_feats}
    
    def find_similar_players(
        self,
        target_player: str,
        league: str = 'all',
        top_n: int = 5,
        use_position_weights: bool = True,
        use_percentiles: bool = False,
    ) -> Optional[pd.DataFrame]:
        """
        Find similar players with Profile-Centric Weighting & Shared-Metric Masking.
        """
        target_idx = self._find_player_index(target_player)
        if target_idx is None:
            return None
        
        target_row = self.df.loc[target_idx]
        target_pos = target_row['Primary_Pos']
        target_league = target_row['League']
        is_gk = target_pos == 'GK'
        
        col_slice = slice(len(FEATURE_COLUMNS), None) if is_gk else slice(0, len(FEATURE_COLUMNS))
        features_subset = GK_FEATURE_COLUMNS if is_gk else FEATURE_COLUMNS
            
        # Build search space
        pos_mask = (self.df['Primary_Pos'] == 'GK') if is_gk else (self.df['Primary_Pos'] != 'GK')
        if league.lower() != 'all':
            search_mask = (self.df['League'] == league) & pos_mask
        else:
            search_mask = pos_mask
            
        search_df = self.df[search_mask].copy()
        search_indices = search_df.index
        search_feat_matrix = self.scaled_features[self.df.index.isin(search_indices)][:, col_slice].copy()
        
        target_row_idx = self.df.index.get_loc(target_idx)
        target_vector = self.scaled_features[target_row_idx, col_slice].copy()
        
        # --- HYPER-STRENGTH EXPONENTIAL WEIGHTING ---
        weights = np.ones(len(features_subset))
        if not is_gk:
            # 1. Start with position-relevant baseline
            profile_map = {'FW': 'Attacker', 'MF': 'Midfielder', 'DF': 'Defender'}
            profile_key = profile_map.get(target_pos, 'Midfielder')
            pw_dict = PROFILE_WEIGHTS[profile_key]
            weights = np.array([pw_dict.get(col, 1.0) for col in FEATURE_COLUMNS])
            
            # 2. Apply Triple Weighting for Elite Stats (>75th Percentile)
            # This forces the engine to match on STRENGTHS, not generic averages
            for i, feat in enumerate(FEATURE_COLUMNS):
                pct = target_row.get(f"{feat}_pct", 50)
                
                # Triple weighting for elite metrics
                if pct > 75:
                    weights[i] *= 3.0
                
                # Weakness correlation fix:
                # If target is weak in this area (< 25th percentile), reduce its weight.
                # This prevents "Anti-Style" penalties where a better player is rejected 
                # because they don't share the target's flaws.
                elif pct < 25:
                    weights[i] *= 0.5 

        # Normalize weights to preserve cosine scale
        norm_weights = weights / weights.mean()
        target_vector *= norm_weights
        search_feat_matrix *= norm_weights

        # --- SHARED-METRIC MASKING ---
        if is_gk:
            target_tracked = set(GK_FEATURE_COLUMNS)
        else:
            CORE_9 = ['Gls/90', 'Ast/90', 'Sh/90', 'SoT/90', 'Crs/90', 'Int/90', 'TklW/90', 'Fls/90', 'Fld/90']
            target_tracked = set(LEAGUE_METRIC_MAP.get(target_league, CORE_9))
        
        final_similarities = []
        primary_drivers = []
        
        for i, (idx, row) in enumerate(search_df.iterrows()):
            if is_gk:
                comp_tracked = set(GK_FEATURE_COLUMNS)
            else:
                comp_league = row['League']
                comp_tracked = set(LEAGUE_METRIC_MAP.get(comp_league, CORE_9))
            
            shared = target_tracked.intersection(comp_tracked)
            shared_indices = [j for j, f in enumerate(features_subset) if f in shared]
            
            if not shared_indices:
                final_similarities.append(0)
                primary_drivers.append("N/A")
                continue
                
            v1 = target_vector[shared_indices].reshape(1, -1)
            v2 = search_feat_matrix[i, shared_indices].reshape(1, -1)
            
            sim = cosine_similarity(v1, v2)[0][0]
            
            # Find Primary Match Drivers (Shared Excellence)
            # We want to identify top 3 features with High Percentiles in BOTH players + Low Distance
            
            # 1. Get stats
            distances_local = np.abs(v1 - v2).flatten()
            shared_feats_list = [features_subset[x] for x in shared_indices]
            
            driver_scores = []
            for k, feat in enumerate(shared_feats_list):
                 # Get percentiles (slow access but needed for logic)
                 p1 = target_row.get(f"{feat}_pct", 50)
                 p2 = row.get(f"{feat}_pct", 50)
                 dist = distances_local[k]
                 
                 # Same formula as calculate_feature_attribution
                 score = (p1 + p2) - (dist * 25)
                 driver_scores.append((feat, score))
            
            # Sort by score desc
            driver_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Pick top 3
            top_3_feats = [x[0] for x in driver_scores[:3]]
            top_3_labels = [RADAR_LABELS.get(f, f) for f in top_3_feats]
            
            driver_str = "Driven by " + ", ".join(top_3_labels)
            
            # Stylistic Twin Check (>95%)
            # We add this decoration here, but the similarity score penalty below might happen
            # So we defer the decoration until after penalty
            
            # Aggressive Coverage Penalty
            coverage = len(shared) / len(target_tracked)
            if coverage < 1.0:
                sim *= (coverage ** 2.0) # Quadratic penalty for missing data
            
            final_sim_score = sim * 100
            if final_sim_score < 0: final_sim_score = 0
            
            if final_sim_score > 95:
                driver_str += " (Stylistic Twin)"
                
            final_similarities.append(final_sim_score)
            primary_drivers.append(driver_str)
        
        search_df['Match_Score'] = np.array(final_similarities)
        search_df['Primary_Driver'] = primary_drivers
        results = search_df[search_df.index != target_idx].sort_values('Match_Score', ascending=False)
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
        
        # Determine which features to compare
        is_both_gk = row1['Primary_Pos'] == 'GK' and row2['Primary_Pos'] == 'GK'
        feats_to_compare = GK_FEATURE_COLUMNS if is_both_gk else FEATURE_COLUMNS
        
        # Build feature comparison
        feature_comparison = {}
        for feat in feats_to_compare:
            if feat in row1.index and feat in row2.index:
                val1 = row1[feat]
                val2 = row2[feat]
                feature_comparison[feat] = {
                    'player1': val1,
                    'player2': val2,
                    'difference': val2 - val1,
                    'player1_better': val1 > val2 if feat not in ['GA90', 'L'] else val1 < val2,
                }
                
                # Add percentiles if available
                pct_col = f'{feat}_pct'
                if pct_col in row1.index and pct_col in row2.index:
                    feature_comparison[feat]['player1_pct'] = row1[pct_col]
                    feature_comparison[feat]['player2_pct'] = row2[pct_col]
        
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
