"""
clustering.py - K-Means clustering for player archetype assignment.

This module implements:
- K-Means clustering with k=8 to assign players to archetypes
- Archetype-to-cluster mapping and validation
- Cluster stability metrics and quality assurance
- Archetype label generation based on cluster characteristics
- PCA-based 2D projection for visualization
- Caching strategy for Streamlit session state

Main entry point: cluster_players() - assigns all players to archetypes
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings

from .constants import FEATURE_COLUMNS, ARCHETYPES, ARCHETYPE_NAMES


# ============================================================================
# CLUSTERING & ARCHETYPE ASSIGNMENT
# ============================================================================

class PlayerArchetypeClusterer:
    """
    K-Means clustering engine for assigning players to archetypes.
    
    Strategy:
    - Fit K-Means with k=8 clusters on scaled features
    - Analyze cluster centroids to determine archetype labels
    - Assign semantic labels (Target Man, Creative Playmaker, etc.)
    - Generate archetype profiles with key metrics
    - Compute PCA projection for 2D visualization
    
    Attributes:
        n_clusters: Number of clusters (fixed at 8)
        kmeans: Fitted KMeans model
        pca: Fitted PCA model (2D projection)
        archetype_profiles: Dict of cluster_id -> archetype info
        player_archetypes: DataFrame with archetype assignments
    """
    
    N_CLUSTERS = 8  # Number of player archetypes
    RANDOM_STATE = 42  # For reproducibility
    
    def __init__(self, scaled_features: np.ndarray, n_clusters: int = 8):
        """
        Initialize clusterer.
        
        Args:
            scaled_features: Normalized feature array from data_engine
            n_clusters: Number of clusters (default: 8)
        """
        self.scaled_features = scaled_features
        self.n_clusters = n_clusters
        self.kmeans = None
        self.pca = None
        self.archetype_profiles = {}
        self.player_archetypes = None
        self.silhouette_score_: Optional[float] = None  # Cached Silhouette Score

    
    def optimize_k(self, k_range: range = range(6, 14)) -> int:
        """
        Find optimal k using Silhouette Score with logic to handle advanced metric noise.
        
        Strategy:
        - Check Baseline K=8.
        - If Silhouette Score < 0.25 (reduced cohesion), test K=10 and K=12.
        - Return the K that maximizes the score among candidates.
        """
        if len(self.scaled_features) < 20:
            return 3 # Fallback for tiny datasets
            
        print(f"[Clustering] Checking cluster cohesion for K=8...")
        
        # Use a sample for speed if dataset is large
        sample_size = 3000 if len(self.scaled_features) > 5000 else None
        
        # 1. Check Baseline K=8
        baseline_k = 8
        best_k = baseline_k
        best_score = -1.0
        
        try:
            km = KMeans(n_clusters=baseline_k, random_state=self.RANDOM_STATE, n_init=5)
            labels = km.fit_predict(self.scaled_features)
            best_score = silhouette_score(self.scaled_features, labels, sample_size=sample_size)
            print(f"  K=8 Score: {best_score:.3f}")
        except:
             return 8
             
        # 2. Conditional Expansion
        # If cohesion is low (likely due to noise from xG/xA features), try higher K
        if best_score < 0.25:
            print(f"Cluster cohesion low (<0.25). Testing K=10, 12...")
            
            comparison_ks = [10, 12]
            for k in comparison_ks:
                try:
                    km = KMeans(n_clusters=k, random_state=self.RANDOM_STATE, n_init=3)
                    labels = km.fit_predict(self.scaled_features)
                    score = silhouette_score(self.scaled_features, labels, sample_size=sample_size)
                    print(f"  K={k} Score: {score:.3f}")
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                except:
                    pass
        else:
            print("Cluster cohesion is healthy.")

        print(f"Selected K={best_k} (Silhouette: {best_score:.3f})")
        return best_k
    
    def fit(self, df: pd.DataFrame, optimize: bool = True) -> 'PlayerArchetypeClusterer':
        """
        Fit K-Means model and analyze cluster characteristics.
        
        Args:
            df: Player DataFrame (must have same length as scaled_features)
            optimize: If True, automatically find best k
            
        Returns:
            self for method chaining
        """
        if len(df) != len(self.scaled_features):
            raise ValueError(
                f"DataFrame length {len(df)} doesn't match "
                f"scaled_features {len(self.scaled_features)}"
            )
        
        if optimize:
            self.n_clusters = self.optimize_k()
        
        print(f"\n[Clustering] Fitting K-Means with k={self.n_clusters}...")
        
        # Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.RANDOM_STATE,
            n_init=10,
            verbose=0,
        )
        cluster_labels = self.kmeans.fit_predict(self.scaled_features)
        
        print(f"K-Means converged in {self.kmeans.n_iter_} iterations")
        
        # Analyze cluster characteristics and assign archetypes
        print(f"[Clustering] Analyzing {self.n_clusters} clusters...")
        self._analyze_clusters(df, cluster_labels)
        
        # Create player-archetype mapping
        self.player_archetypes = df.copy()
        self.player_archetypes['Cluster'] = cluster_labels
        
        # Map cluster IDs to archetype labels and confidence
        archetype_map = {cid: arch['label'] for cid, arch in self.archetype_profiles.items()}
        confidence_map = {cid: arch['confidence'] for cid, arch in self.archetype_profiles.items()}
        
        self.player_archetypes['Archetype'] = [archetype_map[c] for c in cluster_labels]
        self.player_archetypes['Archetype_Confidence'] = [confidence_map[c] for c in cluster_labels]
        
        print(f"Assigned {len(self.player_archetypes)} players to archetypes")
        
        # Fit PCA for 2D visualization
        print(f"[Clustering] Computing PCA projection...")
        self.pca = PCA(n_components=2, random_state=self.RANDOM_STATE)
        pca_features = self.pca.fit_transform(self.scaled_features)
        self.player_archetypes['PCA_X'] = pca_features[:, 0]
        self.player_archetypes['PCA_Y'] = pca_features[:, 1]
        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"PCA explains {explained_var*100:.1f}% of variance")
        if explained_var < 0.70:
             print("Warning: PCA explained variance is low (<70%). Consider 3D visualization or t-SNE if style separation is unclear.")
        
        return self
    
    def _analyze_clusters(self, df: pd.DataFrame, cluster_labels: np.ndarray) -> None:
        """
        Analyze each cluster's characteristics and assign archetype labels.
        
        Strategy:
        - Compute cluster centroid (mean per feature)
        - Identify top 3 features (highest values in centroid)
        - Match cluster profile to predefined archetypes
        - Compute confidence based on cluster cohesion
        
        Args:
            df: Player DataFrame
            cluster_labels: Cluster assignment for each player
        """
        # Compute cluster centroids
        centroids = self.kmeans.cluster_centers_
        
        for cluster_id in range(self.n_clusters):
            # Get cluster centroid
            centroid = centroids[cluster_id]
            
            # Get players in cluster
            mask = cluster_labels == cluster_id
            cluster_size = mask.sum()
            cluster_players = df[mask]
            
            # Identify top features for this cluster
            top_feature_indices = np.argsort(centroid)[-3:][::-1]
            top_features = [FEATURE_COLUMNS[i] for i in top_feature_indices]
            
            # Compute within-cluster variance (cohesion)
            cluster_scaled = self.scaled_features[cluster_labels == cluster_id]
            distances = np.linalg.norm(
                cluster_scaled - centroid.reshape(1, -1),
                axis=1
            )
            cohesion = 1.0 / (1.0 + distances.mean())  # Normalize to 0-1
            
            # Get primary position in cluster
            position_dist = cluster_players['Primary_Pos'].value_counts()
            primary_pos = position_dist.idxmax() if len(position_dist) > 0 else 'UKN'
            
            # Assign archetype label
            archetype_label = self._select_archetype(
                top_features=top_features,
                primary_position=primary_pos,
                centroid_values=centroid,
            )
            
            # Store profile
            self.archetype_profiles[cluster_id] = {
                'label': archetype_label,
                'cluster_id': cluster_id,
                'size': cluster_size,
                'primary_position': primary_pos,
                'top_features': top_features,
                'centroid': centroid.tolist(),
                'confidence': float(cohesion),
                'avg_age': float(cluster_players['Age'].mean()),
                'leagues': cluster_players['League'].value_counts().to_dict(),
            }
        
        # Log cluster summary
        print("\n  Cluster Summary:")
        for cid, profile in sorted(self.archetype_profiles.items()):
            print(
                f"    Cluster {cid}: {profile['label']} "
                f"({profile['size']} players, confidence: {profile['confidence']:.2f})"
            )
    
    def _select_archetype(
        self,
        top_features: List[str],
        primary_position: str,
        centroid_values: np.ndarray,
    ) -> str:
        """
        Select best-matching archetype label for a cluster using strict positional filtering.
        
        Args:
            top_features: Top 3 features in cluster centroid
            primary_position: Most common position in cluster ('FW', 'MF', 'DF', 'GK')
            centroid_values: Cluster centroid (normalized)
            
        Returns:
            Archetype label string
        """
        # Normalize position for lookup
        pos_map = {'FW': 'FW', 'AM': 'FW', 'CM': 'MF', 'DM': 'MF', 'FB': 'DF', 'CB': 'DF', 'GK': 'GK'}
        norm_pos = pos_map.get(primary_position, primary_position)
        
        # Get candidates that match this position
        candidates = [
            name for name, arch in ARCHETYPES.items() 
            if arch.get('primary_position') == norm_pos
        ]
        
        if not candidates:
            # Fallback if no matching position found (shouldn't happen with norm_pos)
            candidates = ARCHETYPE_NAMES
            
        # Score each candidate based on key metric overlap with top_features
        best_match = candidates[0]
        best_score = -1.0
        
        for archetype_name in candidates:
            archetype = ARCHETYPES[archetype_name]
            key_metrics = archetype.get('key_metrics', [])
            
            # Compute overlap score
            # Base score: number of top features that are key metrics for this archetype
            overlap = sum(1 for feat in top_features if feat in key_metrics)
            
            # Tie-breaker: magnitude of key metrics in the centroid
            # We use the average value of key metrics in this centroid to refine the match
            feature_indices = [FEATURE_COLUMNS.index(m) for m in key_metrics if m in FEATURE_COLUMNS]
            if feature_indices:
                magnitude = np.mean([centroid_values[i] for i in feature_indices])
            else:
                magnitude = 0.0
                
            score = overlap + (magnitude * 0.1)
            
            if score > best_score:
                best_score = score
                best_match = archetype_name
        
        return best_match
    
    def get_cluster_profile(self, cluster_id: int) -> Dict:
        """
        Get detailed profile for a cluster/archetype.
        
        Args:
            cluster_id: Cluster ID (0-7)
            
        Returns:
            Dict with archetype info, centroid, and statistics
        """
        if cluster_id not in self.archetype_profiles:
            return None
        
        return self.archetype_profiles[cluster_id]
    
    def get_player_archetype(self, player_name: str) -> Optional[str]:
        """
        Get archetype for a specific player.
        
        Args:
            player_name: Player name
            
        Returns:
            Archetype label or None if not found
        """
        if self.player_archetypes is None:
            return None
        
        matches = self.player_archetypes[self.player_archetypes['Player'] == player_name]
        if len(matches) > 0:
            return matches.iloc[0]['Archetype']
        
        return None
    
    def get_archetype_players(self, archetype_label: str) -> pd.DataFrame:
        """
        Get all players in a specific archetype.
        
        Args:
            archetype_label: Archetype name (e.g., 'Target Man')
            
        Returns:
            DataFrame of players in archetype
        """
        if self.player_archetypes is None:
            return pd.DataFrame()
        
        return self.player_archetypes[
            self.player_archetypes['Archetype'] == archetype_label
        ].copy()
    
    def get_archetype_statistics(self, archetype_label: str) -> Dict:
        """
        Get aggregate statistics for an archetype.
        
        Args:
            archetype_label: Archetype name
            
        Returns:
            Dict with mean/median stats for archetype
        """
        players = self.get_archetype_players(archetype_label)
        
        if len(players) == 0:
            return {}
        
        stats = {
            'count': len(players),
            'avg_age': players['Age'].mean(),
            'age_range': (players['Age'].min(), players['Age'].max()),
            'avg_90s': players['90s'].mean(),
            'leagues': players['League'].value_counts().to_dict(),
        }
        
        # Add feature statistics
        for feat in FEATURE_COLUMNS:
            stats[f'avg_{feat}'] = players[feat].mean()
            stats[f'med_{feat}'] = players[feat].median()
        
        return stats

    def calculate_outlier_scores(self) -> None:
        """
        Calculate Euclidean distance of each player from their cluster centroid.
        Adds 'Centroid_Distance' and 'Outlier_Score' to player_archetypes.
        
        High Outlier_Score = 'Unicorn' / Stylistic outlier.
        """
        if self.player_archetypes is None or self.kmeans is None:
            return
            
        centroids = self.kmeans.cluster_centers_
        labels = self.player_archetypes['Cluster'].values
        
        # Calculate distance for each player to their assigned centroid
        distances = []
        for i, label in enumerate(labels):
            player_features = self.scaled_features[i]
            centroid = centroids[label]
            dist = np.linalg.norm(player_features - centroid)
            distances.append(dist)
            
        self.player_archetypes['Centroid_Distance'] = distances
        
        # Normalize to 0-100 Score (Outlier Score)
        # We use percentiles: 100 = Furthest away (Unicorn)
        if len(distances) > 0:
            distances = np.array(distances)
            try:
                # Rank distances (Higher distance = Higher rank)
                # Percentile rank * 100
                from scipy.stats import rankdata
                ranks = rankdata(distances)
                scores = (ranks / len(ranks)) * 100
            except:
                scores = np.zeros(len(distances))
                
            self.player_archetypes['Outlier_Score'] = scores

    def get_silhouette_score(self, max_samples: int = 1000) -> float:
        """
        Calculate Silhouette Score for current clustering.
        
        Uses sampling for large datasets to avoid O(N^2) complexity.
        Logs warning if score < 0.35 indicating cluster overlap.
        
        Args:
            max_samples: Maximum samples for calculation (default: 1000)
            
        Returns:
            Silhouette Score between -1 and 1 (higher = better separation)
        """
        import logging
        import time
        logger = logging.getLogger(__name__)
        
        # Return cached score if available
        if self.silhouette_score_ is not None:
            return self.silhouette_score_
        
        if self.kmeans is None or self.scaled_features is None:
            logger.warning("Clustering not fitted. Cannot calculate Silhouette Score.")
            return 0.0
        
        if len(self.scaled_features) < 10:
            logger.warning("Insufficient data for Silhouette Score.")
            return 0.0
        
        start_time = time.time()
        
        try:
            labels = self.kmeans.predict(self.scaled_features)
            n_samples = len(self.scaled_features)
            
            # Use sampling if dataset is large (O(N^2) complexity)
            if n_samples > max_samples:
                sample_size = max_samples
                logger.info(
                    f"Using sampled Silhouette Score ({sample_size}/{n_samples} samples) "
                    "for performance."
                )
            else:
                sample_size = None
            
            score = silhouette_score(
                self.scaled_features,
                labels,
                sample_size=sample_size,
                random_state=self.RANDOM_STATE
            )
            
            elapsed = time.time() - start_time
            
            # Cache the score
            self.silhouette_score_ = float(score)
            
            # Log warning for overlap
            if score < 0.35:
                logger.warning(
                    f"Cluster Overlap Warning: Silhouette Score {score:.3f} < 0.35. "
                    "Consider increasing K or reviewing feature selection."
                )
            else:
                logger.info(f"Cluster Silhouette Score: {score:.3f} (computed in {elapsed:.2f}s)")
            
            return self.silhouette_score_
            
        except Exception as e:
            logger.error(f"Silhouette Score calculation failed: {e}")
            return 0.0


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def cluster_players(
    df: pd.DataFrame,
    scaled_features: np.ndarray,
    n_clusters_default: int = 5,
) -> Tuple[pd.DataFrame, 'PlayerArchetypeClusterer']:
    """
    Assign players to archetypes using Position-Specific K-Means clustering.
    Strictly preserves the input index and length.
    """
    print("\n" + "=" * 80)
    print("STARTING POSITIONAL PLAYER CLUSTERING")
    print("=" * 80)
    
    # 1. Broad Positional Mapping
    POS_TO_GROUP = {
        'FW': 'FW', 'ST': 'FW', 'RW': 'FW', 'LW': 'FW', 'AM': 'FW',
        'CM': 'MF', 'DM': 'MF', 'MF': 'MF', 'RM': 'MF', 'LM': 'MF',
        'FB': 'DF', 'CB': 'DF', 'WB': 'DF', 'DF': 'DF', 'LB': 'DF', 'RB': 'DF',
    }
    
    # Work on a copy to avoid side effects
    df_work = df.copy()
    
    # Separate Goalkeepers
    is_gk = df_work['Primary_Pos'] == 'GK'
    df_outfield = df_work[~is_gk].copy()
    outfield_indices = df_outfield.index
    outfield_features = scaled_features[~is_gk, :len(FEATURE_COLUMNS)]
    
    if len(df_outfield) > 0:
        # Assign broad groups (Fallback to MF for anything else)
        df_outfield['Clustering_Group'] = df_outfield['Primary_Pos'].apply(
            lambda x: POS_TO_GROUP.get(str(x).upper(), 'MF')
        )
        
        # Global PCA for Outfield (Visualization Consistency)
        pca_global = PCA(n_components=2, random_state=42)
        pca_coords = pca_global.fit_transform(outfield_features)
        df_outfield['PCA_X'] = pca_coords[:, 0]
        df_outfield['PCA_Y'] = pca_coords[:, 1]
    
    clustered_parts = []
    main_clusterer = None

    # 2. Independent Clustering for each Outfield Group
    if len(df_outfield) > 0:
        # Ensure we handle every group including potential NaNs (which should be 'MF' now)
        unique_groups = [g for g in df_outfield['Clustering_Group'].unique() if pd.notna(g)]
        
        for group_name in sorted(unique_groups):
            mask = df_outfield['Clustering_Group'] == group_name
            if not mask.any():
                continue
                
            print(f"[Clustering] Group: {group_name} ({mask.sum()} players)")
            
            group_df = df_outfield[mask].copy()
            # Find integer positions of these rows in the outfield matrix
            group_pos_indices = np.where(mask.values)[0]
            group_features = outfield_features[group_pos_indices]
            
            n_c = min(n_clusters_default, len(group_df))
            if n_c < 2 and len(group_df) >= 2: n_c = 2
            elif n_c < 1: n_c = 1
            
            clusterer = PlayerArchetypeClusterer(group_features, n_clusters=n_c)
            clusterer.fit(group_df, optimize=False) 
            clusterer.calculate_outlier_scores()
            
            clustered_parts.append(clusterer.player_archetypes)
            main_clusterer = clusterer

    # 3. Handle Goalkeepers
    df_gk = df_work[is_gk].copy()
    if len(df_gk) > 0:
        gk_features = scaled_features[is_gk, len(FEATURE_COLUMNS):]
        n_gk_clusters = min(3, len(df_gk))
        
        from .constants import GK_FEATURE_COLUMNS
        import utils.clustering as cl
        original_fc = cl.FEATURE_COLUMNS
        cl.FEATURE_COLUMNS = GK_FEATURE_COLUMNS
        try:
            gk_clusterer = PlayerArchetypeClusterer(gk_features, n_clusters=n_gk_clusters)
            gk_clusterer.fit(df_gk, optimize=False)
            df_gk_clustered = gk_clusterer.player_archetypes
            
            # Map labels
            gk_archetype_map = {0: 'Shot-Stopper', 1: 'Sweeper-Keeper', 2: 'Ball-Playing GK'}
            df_gk_clustered['Archetype'] = df_gk_clustered['Cluster'].map(lambda x: gk_archetype_map.get(x, 'Elite Keeper'))
            df_gk_clustered['PCA_X'] = 0 
            df_gk_clustered['PCA_Y'] = 0
            clustered_parts.append(df_gk_clustered)
        finally:
            cl.FEATURE_COLUMNS = original_fc

    # 4. Final Combination and Integrity Check
    if not clustered_parts:
        # Emergency fallback if no players clustered
        df_result = df.copy()
        df_result['Archetype'] = 'Unknown'
        df_result['Cluster'] = -1
    else:
        # Combine all parts
        df_result = pd.concat(clustered_parts, axis=0)
        
        # FINAL INTEGRITY CHECK: Ensure we haven't lost any rows
        if len(df_result) < len(df):
            print(f"WARNING: Row count mismatch after clustering ({len(df_result)} vs {len(df)}). Recovering missing rows...")
            missing_indices = df.index.difference(df_result.index)
            if not missing_indices.empty:
                df_missing = df.loc[missing_indices].copy()
                df_missing['Archetype'] = 'Unknown'
                df_missing['Cluster'] = -1
                df_result = pd.concat([df_result, df_missing], axis=0)
        
        # Restore original order exactly
        df_result = df_result.reindex(df.index)

    print(f"CLUSTERING COMPLETE: {len(df_result)} players processed.")
    print("=" * 80)
    
    return df_result, main_clusterer



# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_archetype_color(archetype_label: str) -> str:
    """
    Get color code for archetype.
    
    Args:
        archetype_label: Archetype name
        
    Returns:
        Hex color code
    """
    if archetype_label in ARCHETYPES:
        return ARCHETYPES[archetype_label]['color']
    return '#7F8C8D'  # Default gray


def get_archetype_description(archetype_label: str) -> str:
    """
    Get description for archetype.
    
    Args:
        archetype_label: Archetype name
        
    Returns:
        Description string
    """
    if archetype_label in ARCHETYPES:
        return ARCHETYPES[archetype_label]['description']
    return 'Unknown archetype'


def get_archetypes_by_position(position: str) -> List[str]:
    """
    Get archetypes typical for a position.
    
    Args:
        position: Position code (FW, MF, DF, GK)
        
    Returns:
        List of archetype names
    """
    return [
        name for name, arch in ARCHETYPES.items()
        if arch.get('primary_position') == position
    ]
