"""
evaluator.py - ML Model Evaluation and SHAP Explainability Service.

This module provides:
- Random Forest market value model evaluation (MAE, R²)
- SHAP-based feature contribution explanations using TreeExplainer
- Human-readable metric name mappings for frontend display

Performance optimizations:
- Uses TreeExplainer (exact for tree models, O(TLD) complexity)
- Caches trained model and SHAP explainer
- Background sampling not needed with TreeExplainer
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import os
import sys
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Define model path
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'market_value_model_bundle.joblib')

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# HUMAN-READABLE METRIC MAPPINGS
# =============================================================================

METRIC_DISPLAY_NAMES = {
    # Core performance metrics
    'Gls/90': 'Goals per 90',
    'Ast/90': 'Assists per 90',
    'Sh/90': 'Shots per 90',
    'SoT/90': 'Shots on Target per 90',
    'Crs/90': 'Crosses per 90',
    'Int/90': 'Interceptions per 90',
    'TklW/90': 'Tackles Won per 90',
    'Fls/90': 'Fouls per 90',
    'Fld/90': 'Fouls Drawn per 90',
    
    # Expected metrics
    'xG90': 'Expected Goals',
    'xAG90': 'Expected Assists',
    'xGChain90': 'xG Chain per 90',
    'xGBuildup90': 'xG Buildup per 90',
    
    # Goalkeeper metrics
    'GA90': 'Goals Against per 90',
    'Save%': 'Save Percentage',
    'CS%': 'Clean Sheet Percentage',
    
    # Engineered features
    'Age': 'Age',
    'Age_Squared': 'Age (Squared)',
    'Age_Decay': 'Age Decay Factor',
    'Is_Prime_Age': 'In Prime Age (24-28)',
    'Is_Young_Prospect': 'Young Prospect (<23)',
    'League_Tier': 'League Quality Tier',
    'Position_Multiplier': 'Position Value Multiplier',
    'Avg_Percentile': 'Average Percentile (All Stats)',
    'Top3_Percentile': 'Top 3 Skills Average',
    'Percentile_Consistency': 'Statistical Consistency',
    'Goal_Contribution': 'Goal Contribution (G+A/90)',
    'Games_Played': 'Games Played (90s)',
    'Is_Regular_Starter': 'Regular Starter (20+ 90s)',
    '90s': 'Minutes Played (90s)',
    
    # Percentile columns
    'Gls/90_pct': 'Goals Percentile',
    'Ast/90_pct': 'Assists Percentile',
    'xG90_pct': 'xG Percentile',
    'xAG90_pct': 'xAG Percentile',
}


def get_display_name(column_name: str) -> str:
    """Convert technical column name to human-readable display name."""
    # Check direct mapping
    if column_name in METRIC_DISPLAY_NAMES:
        return METRIC_DISPLAY_NAMES[column_name]
    
    # Handle archetype dummies
    if column_name.startswith('Archetype_'):
        archetype = column_name.replace('Archetype_', '')
        return f'Archetype: {archetype}'
    
    # Handle percentile columns
    if column_name.endswith('_pct'):
        base = column_name.replace('_pct', '')
        base_name = METRIC_DISPLAY_NAMES.get(base, base)
        return f'{base_name} Percentile'
    
    # Fallback: Clean up the name
    return column_name.replace('_', ' ').replace('/', ' per ')


# =============================================================================
# MODEL EVALUATOR CLASS
# =============================================================================

class ModelEvaluator:
    """
    ML Model Evaluator for Market Value Random Forest.
    
    Features:
    - Train/test split evaluation (80/20)
    - MAE and R² score calculation
    - SHAP TreeExplainer for feature contributions
    - Human-readable metric names for UI
    
    Performance:
    - TreeExplainer is O(TLD) where T=trees, L=leaves, D=depth
    - Much faster than KernelExplainer for tree-based models
    - Exact SHAP values (not approximations)
    """
    
    def __init__(self):
        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.explainer = None  # SHAP TreeExplainer
        self.feature_columns: List[str] = []
        self.metrics: Dict[str, float] = {}
        self.is_trained: bool = False
        self.df_with_features: Optional[pd.DataFrame] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[pd.Series] = None
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for ML model.
        Mirrors the logic from utils/ml_value_model.py.
        """
        df_features = df.copy()
        
        # Age features
        df_features['Age_Squared'] = df_features['Age'] ** 2
        df_features['Age_Decay'] = np.where(
            df_features['Age'] <= 25,
            1.0,
            np.maximum(0.5, 1.0 - (df_features['Age'] - 25) * 0.05)
        )
        df_features['Is_Prime_Age'] = (
            (df_features['Age'] >= 24) & (df_features['Age'] <= 28)
        ).astype(int)
        df_features['Is_Young_Prospect'] = (df_features['Age'] <= 23).astype(int)
        
        # League tier encoding
        league_tiers = {
            'Premier League': 5,
            'Championship': 4,
            'League One': 3,
            'League Two': 2,
            'National League': 1,
        }
        df_features['League_Tier'] = df_features['League'].map(league_tiers).fillna(1)
        
        # Position encoding
        position_values = {
            'FW': 1.2,
            'MF': 1.0,
            'DF': 0.9,
            'GK': 0.8,
        }
        df_features['Position_Multiplier'] = df_features['Primary_Pos'].map(
            position_values
        ).fillna(1.0)
        
        # Archetype encoding (one-hot)
        if 'Archetype' in df_features.columns:
            archetype_dummies = pd.get_dummies(df_features['Archetype'], prefix='Archetype')
            df_features = pd.concat([df_features, archetype_dummies], axis=1)
        
        # Percentile-based features
        percentile_cols = [col for col in df_features.columns if col.endswith('_pct')]
        
        if len(percentile_cols) > 0:
            df_features['Avg_Percentile'] = df_features[percentile_cols].mean(axis=1)
            df_features['Top3_Percentile'] = df_features[percentile_cols].apply(
                lambda row: np.mean(sorted(row, reverse=True)[:3]),
                axis=1
            )
            df_features['Percentile_Consistency'] = df_features[percentile_cols].std(axis=1)
        
        # Performance features
        if 'Gls/90' in df_features.columns:
            df_features['Goal_Contribution'] = (
                df_features['Gls/90'] + df_features.get('Ast/90', 0)
            )
        
        if '90s' in df_features.columns:
            df_features['Games_Played'] = df_features['90s']
            df_features['Is_Regular_Starter'] = (df_features['90s'] >= 20).astype(int)
        
        return df_features
    
    def train_and_evaluate(
        self,
        df: pd.DataFrame,
        target_column: str = 'Transfermarkt_Value_£M',
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """
        Train model with 80/20 split and evaluate.
        
        Args:
            df: DataFrame with player data and target values
            target_column: Column containing market values
            test_size: Test set proportion (default: 0.2 = 20%)
            random_state: Random seed for reproducibility
            
        Returns:
            Dict with MAE, R², feature importance, and metadata
        """
        # Check if target column exists
        if target_column not in df.columns:
            # Try alternative column names
            alt_columns = [
                'Transfermarkt_Value_£M', 
                'Market_Value', 
                'Value', 
                'TM_Value',
                'Estimated_Value_£M'
            ]
            found_col = None
            for alt in alt_columns:
                if alt in df.columns:
                    found_col = alt
                    break
            
            if found_col:
                logger.info(f"Using '{found_col}' as target column instead of '{target_column}'")
                target_column = found_col
            else:
                # Use synthetic value based on performance metrics
                logger.warning(
                    f"Target column '{target_column}' not found. "
                    "Creating synthetic value score for demonstration."
                )
                
                # Create synthetic value score from available stats
                df = df.copy()
                synthetic_cols = []
                
                # Score based on available percentile columns
                pct_cols = [c for c in df.columns if c.endswith('_pct')]
                if pct_cols:
                    df['_synthetic_value'] = df[pct_cols].mean(axis=1)
                    target_column = '_synthetic_value'
                    logger.info(f"Created synthetic target from {len(pct_cols)} percentile columns")
                else:
                    # Last fallback: use 90s (games played) as a proxy
                    if '90s' in df.columns:
                        df['_synthetic_value'] = df['90s'] * 0.1  # Scale down
                        target_column = '_synthetic_value'
                        logger.info("Created synthetic target from 90s column")
                    else:
                        raise ValueError(
                            f"No suitable target column found. "
                            f"Available columns: {list(df.columns)[:20]}..."
                        )
        
        # Filter to players with known values
        df_train = df[df[target_column].notna()].copy()
        
        if len(df_train) < 50:
            raise ValueError(
                f"Insufficient training data: only {len(df_train)} players with values "
                f"in column '{target_column}'. Need at least 50."
            )


        
        logger.info(f"Training evaluator on {len(df_train)} players...")
        
        # Engineer features
        df_train = self._engineer_features(df_train)
        self.df_with_features = df_train.copy()
        
        # Select feature columns
        exclude_cols = [
            'Player', 'Squad', 'League', 'Primary_Pos', 'Archetype',
            target_column, 'Estimated_Value_£M', 'Value_Tier', 'Value_Score',
            'Transfermarkt_Value_£M', 'TM_Name', 'Cluster', 'PCA_X', 'PCA_Y',
            'Completeness_Score', 'Archetype_Confidence', 'Centroid_Distance',
            'Outlier_Score', 'Nation', 'Proxy_Warnings',
        ]
        
        self.feature_columns = [
            col for col in df_train.columns
            if col not in exclude_cols 
            and df_train[col].dtype in ['int64', 'float64', 'int32', 'float32']
            and not col.startswith('Unnamed')
        ]
        
        # Prepare X and y
        X = df_train[self.feature_columns].fillna(0)
        y = df_train[target_column]
        
        # Train-test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Store test set for SHAP explanations
        self.X_test = X_test
        self.y_test = y_test
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=40,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        self.metrics = {
            'mae': float(mean_absolute_error(y_test, y_pred_test)),
            'r2_score': float(r2_score(y_test, y_pred_test)),
            'train_mae': float(mean_absolute_error(y_train, y_pred_train)),
            'train_r2': float(r2_score(y_train, y_pred_train)),
            'sample_count': len(df_train),
            'train_count': len(X_train),
            'test_count': len(X_test),
            'feature_count': len(self.feature_columns),
            'model_type': 'RandomForest',
        }
        
        # Feature importance (sorted)
        feature_importance = dict(zip(
            self.feature_columns,
            self.model.feature_importances_
        ))
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        self.metrics['feature_importance'] = feature_importance
        
        # Initialize SHAP TreeExplainer
        try:
            import shap
            # TreeExplainer is exact and fast for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("✅ SHAP TreeExplainer initialized")
        except ImportError:
            logger.warning("⚠️ SHAP not installed. Explainability features disabled.")
            self.explainer = None
        except Exception as e:
            logger.warning(f"⚠️ SHAP initialization failed: {e}")
            self.explainer = None
        
        self.is_trained = True
        
        logger.info(
            f"✅ Model trained: MAE=£{self.metrics['mae']:.2f}M, "
            f"R²={self.metrics['r2_score']:.3f}"
        )

        # Save model bundle
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            model_bundle = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'metrics': self.metrics
            }
            joblib.dump(model_bundle, MODEL_FILE)
            logger.info(f"✅ Model bundle saved to {MODEL_FILE}")
        except Exception as e:
            logger.error(f"❌ Failed to save model bundle: {e}")
        
        return self.metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current model evaluation metrics."""
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        # Return metrics with human-readable feature importance
        result = self.metrics.copy()
        
        if 'feature_importance' in result:
            # Convert to human-readable names (top 10)
            readable_importance = {}
            for feat, imp in list(result['feature_importance'].items())[:10]:
                display_name = get_display_name(feat)
                readable_importance[display_name] = round(imp, 4)
            result['feature_importance'] = readable_importance
        
        return result
    
    def get_shap_explanation(
        self,
        player_name: str,
        top_n: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Generate SHAP explanation for a specific player.
        
        Uses TreeExplainer for exact SHAP values (fast for tree models).
        
        Args:
            player_name: Player name to explain
            top_n: Number of top features to return
            
        Returns:
            Dict with base_value, prediction, and sorted contributions
        """
        if not self.is_trained or self.explainer is None:
            logger.warning("Model or SHAP explainer not initialized")
            return None
        
        if self.df_with_features is None:
            logger.warning("Feature data not available")
            return None
        
        # Find player in dataset
        player_mask = self.df_with_features['Player'].str.contains(
            player_name.split(' (')[0],  # Handle "Player (Squad)" format
            case=False,
            na=False
        )
        
        if not player_mask.any():
            logger.warning(f"Player '{player_name}' not found in dataset")
            return None
        
        player_row = self.df_with_features[player_mask].iloc[0]
        
        # Get features for this player
        X_player = player_row[self.feature_columns].fillna(0).values.reshape(1, -1)
        X_player_scaled = self.scaler.transform(X_player)
        
        # Get prediction
        prediction = float(self.model.predict(X_player_scaled)[0])
        
        # Calculate SHAP values
        try:
            shap_values = self.explainer.shap_values(X_player_scaled)
            
            # TreeExplainer returns array directly
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For multi-output
            
            shap_values = shap_values.flatten()
            
            # Get base value (expected value)
            base_value = float(self.explainer.expected_value)
            if isinstance(base_value, np.ndarray):
                base_value = float(base_value[0])
            
            # Create contributions list with human-readable names
            contributions = []
            for i, (feat, shap_val) in enumerate(zip(self.feature_columns, shap_values)):
                contributions.append({
                    'feature': get_display_name(feat),
                    'feature_raw': feat,
                    'value': float(X_player.flatten()[i]),
                    'contribution': float(shap_val),
                })
            
            # Sort by absolute contribution (most impactful first)
            contributions = sorted(
                contributions,
                key=lambda x: abs(x['contribution']),
                reverse=True
            )[:top_n]
            
            return {
                'player_name': player_name,
                'base_value': round(base_value, 2),
                'prediction': round(prediction, 2),
                'contributions': contributions,
            }
            
        except Exception as e:
            logger.error(f"SHAP calculation failed: {e}")
            return None


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

# Global evaluator instance (loaded once)
_evaluator_instance: Optional[ModelEvaluator] = None


def get_evaluator() -> ModelEvaluator:
    """Get or create the global ModelEvaluator instance."""
    global _evaluator_instance
    if _evaluator_instance is None:
        _evaluator_instance = ModelEvaluator()
    return _evaluator_instance


def load_model_if_exists(evaluator: ModelEvaluator) -> bool:
    """Attempt to load trained model from disk."""
    if os.path.exists(MODEL_FILE):
        try:
            logger.info(f"Loading model bundle from {MODEL_FILE}...")
            bundle = joblib.load(MODEL_FILE)
            
            evaluator.model = bundle['model']
            evaluator.scaler = bundle['scaler']
            evaluator.feature_columns = bundle['feature_columns']
            evaluator.metrics = bundle.get('metrics', {})
            
            # Re-initialize SHAP explainer
            try:
                import shap
                evaluator.explainer = shap.TreeExplainer(evaluator.model)
            except Exception as e:
                logger.warning(f"Failed to re-initialize SHAP: {e}")
            
            evaluator.is_trained = True
            logger.info("✅ Model loaded successfully from disk")
            return True
        except Exception as e:
            logger.error(f"Failed to load model from disk: {e}")
            return False
    return False


def initialize_evaluator(df: pd.DataFrame) -> ModelEvaluator:
    """Initialize the evaluator with training data."""
    evaluator = get_evaluator()
    if not evaluator.is_trained:
        # Try loading from disk first
        if load_model_if_exists(evaluator):
            return evaluator
            
        try:
            evaluator.train_and_evaluate(df)
        except Exception as e:
            logger.error(f"Failed to train evaluator: {e}")
    return evaluator
