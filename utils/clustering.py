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
        archetype_profiles: Dict of cluster_id → archetype info
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
    
    def optimize_k(self, k_range: range = range(8, 13)) -> int:
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
            print(f"⚠️  Cluster cohesion low (<0.25). Testing K=10, 12...")
            
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
            print("✓ Cluster cohesion is healthy.")

        print(f"✓ Selected K={best_k} (Silhouette: {best_score:.3f})")
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
        
        print(f"✓ K-Means converged in {self.kmeans.n_iter_} iterations")
        
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
        
        print(f"✓ Assigned {len(self.player_archetypes)} players to archetypes")
        
        # Fit PCA for 2D visualization
        print(f"[Clustering] Computing PCA projection...")
        self.pca = PCA(n_components=2, random_state=self.RANDOM_STATE)
        pca_features = self.pca.fit_transform(self.scaled_features)
        self.player_archetypes['PCA_X'] = pca_features[:, 0]
        self.player_archetypes['PCA_Y'] = pca_features[:, 1]
        explained_var = self.pca.explained_variance_ratio_.sum()
        print(f"✓ PCA explains {explained_var*100:.1f}% of variance")
        if explained_var < 0.65:
             print("⚠️  Warning: PCA explained variance is low (<65%). Consider 3D or t-SNE if visualization is cluttered.")
        
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
        Select best-matching archetype label for a cluster.
        
        Strategy:
        - Match cluster's top features against archetype definitions
        - Prioritize position alignment
        - Fall back to generic label if no clear match
        
        Args:
            top_features: Top 3 features in cluster centroid
            primary_position: Most common position in cluster
            centroid_values: Cluster centroid (normalized)
            
        Returns:
            Archetype label string
        """
        # Position-based archetype mapping
        position_archetypes = {
            'FW': ['Target Man', 'Elite Keeper'],  # Fallback to elite if weird
            'MF': ['Creative Playmaker', 'Ball-Winning Midfielder', 'Box-to-Box', 'Buildup Boss'],
            'DF': ['Full-Back Playmaker', 'Aggressive Defender', 'Sweeper', 'Buildup Boss'],
            'GK': ['Elite Keeper'],
        }
        
        # Get candidate archetypes for position
        candidates = position_archetypes.get(primary_position, ARCHETYPE_NAMES)
        
        # Score each candidate based on feature match
        best_match = candidates[0]
        best_score = 0
        
        for archetype_name in candidates:
            if archetype_name not in ARCHETYPES:
                continue
            
            archetype = ARCHETYPES[archetype_name]
            key_metrics = archetype['key_metrics']
            
            # Compute overlap between top features and key metrics
            overlap = sum(1 for feat in top_features if feat in key_metrics)
            score = overlap + 0.5 * (primary_position == archetype.get('primary_position', ''))
            
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


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def cluster_players(
    df: pd.DataFrame,
    scaled_features: np.ndarray,
    n_clusters: int = 8,
) -> Tuple[pd.DataFrame, 'PlayerArchetypeClusterer']:
    """
    Assign players to archetypes using K-Means clustering.
    Handles the split 22-wide feature matrix [outfield(13) | gk(9)].
    """
    print("\n" + "=" * 80)
    print("🎯 STARTING PLAYER ARCHETYPE CLUSTERING")
    print("=" * 80)
    
    # Separate goalkeepers from outfield players
    is_gk = df['Primary_Pos'] == 'GK'
    df_gk = df[is_gk].copy()
    df_outfield = df[~is_gk].copy()
    
    # Cluster outfield players (8 clusters) using columns 0-13
    if len(df_outfield) > 0:
        outfield_features = scaled_features[~is_gk, :len(FEATURE_COLUMNS)]
        clusterer_outfield = PlayerArchetypeClusterer(outfield_features, n_clusters=n_clusters)
        # Enable dynamic optimization for outfield players
        clusterer_outfield.fit(df_outfield, optimize=True)
        df_outfield_clustered = clusterer_outfield.player_archetypes
    else:
        df_outfield_clustered = df_outfield
        clusterer_outfield = None
    
    # Cluster goalkeepers separately using columns 13-22
    if len(df_gk) > 0:
        # GK logic with columns 13 to 22
        gk_features = scaled_features[is_gk, len(FEATURE_COLUMNS):]
        n_gk_clusters = min(3, len(df_gk))
        
        if n_gk_clusters >= 2:
            # Need a modified clusterer or just use the class
            # Note: FEATURE_COLUMNS internally in PlayerArchetypeClusterer 
            # might cause issues if it's hardcoded to outfield stats.
            # We'll temporarily mock it or handle it.
            
            # Since PlayerArchetypeClusterer uses FEATURE_COLUMNS, we need to pass GK equivalents
            from .constants import GK_FEATURE_COLUMNS
            
            # Temporary monkeypatch of FEATURE_COLUMNS for GK clustering
            import utils.clustering as cl
            original_fc = cl.FEATURE_COLUMNS
            cl.FEATURE_COLUMNS = GK_FEATURE_COLUMNS
            
            try:
                clusterer_gk = PlayerArchetypeClusterer(gk_features, n_clusters=n_gk_clusters)
                clusterer_gk.fit(df_gk)
                df_gk_clustered = clusterer_gk.player_archetypes
            finally:
                cl.FEATURE_COLUMNS = original_fc
            
            gk_archetype_map = {0: 'Shot-Stopper', 1: 'Sweeper-Keeper', 2: 'Ball-Playing GK'}
            df_gk_clustered['Archetype'] = df_gk_clustered['Cluster'].map(lambda x: gk_archetype_map.get(x, 'Elite Keeper'))
        else:
            df_gk_clustered = df_gk.copy()
            df_gk_clustered['Archetype'] = 'Elite Keeper'
            df_gk_clustered['Cluster'] = 0
            df_gk_clustered['Archetype_Confidence'] = 1.0
            df_gk_clustered['PCA_X'] = 0
            df_gk_clustered['PCA_Y'] = 0
    else:
        df_gk_clustered = df_gk
    
    # Combine results
    df_combined = pd.concat([df_outfield_clustered, df_gk_clustered], axis=0).loc[df.index]
    
    print("\n" + "=" * 80)
    print("✓ CLUSTERING COMPLETE")
    print("=" * 80)
    
    return df_combined, clusterer_outfield



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
