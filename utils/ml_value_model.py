"""
ml_value_model.py - Machine Learning-based transfer value prediction.

This module provides:
- Random Forest regression model for transfer value prediction
- Training on Transfermarkt data
- Feature engineering (age decay, percentile boosts, league multipliers)
- "Fair Value" vs "Market Premium" analysis
- Undervalued bargain identification
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple, List
import pickle
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


class MLValuePredictor:
    """
    Machine Learning-based transfer value predictor.
    
    Features:
    - Random Forest / Gradient Boosting regression
    - Feature engineering (age, percentiles, league, archetype)
    - Cross-validation for model selection
    - Fair value vs market premium analysis
    - Undervalued bargain detection
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize ML value predictor.
        
        Args:
            model_type: 'random_forest' or 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_trained = False
        self.feature_importance = {}
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for ML model.
        
        Args:
            df: DataFrame with player data
            
        Returns:
            DataFrame with engineered features
        """
        df_features = df.copy()
        
        # Age features
        df_features['Age_Squared'] = df_features['Age'] ** 2
        df_features['Age_Decay'] = np.where(
            df_features['Age'] <= 25,
            1.0,
            np.maximum(0.5, 1.0 - (df_features['Age'] - 25) * 0.05)
        )
        df_features['Is_Prime_Age'] = ((df_features['Age'] >= 24) & (df_features['Age'] <= 28)).astype(int)
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
            'FW': 1.2,  # Attackers typically more expensive
            'MF': 1.0,
            'DF': 0.9,
            'GK': 0.8,
        }
        df_features['Position_Multiplier'] = df_features['Primary_Pos'].map(position_values).fillna(1.0)
        
        # Archetype encoding (one-hot)
        if 'Archetype' in df_features.columns:
            archetype_dummies = pd.get_dummies(df_features['Archetype'], prefix='Archetype')
            df_features = pd.concat([df_features, archetype_dummies], axis=1)
        
        # Percentile-based features
        percentile_cols = [col for col in df_features.columns if col.endswith('_pct')]
        
        if len(percentile_cols) > 0:
            # Average percentile across all metrics
            df_features['Avg_Percentile'] = df_features[percentile_cols].mean(axis=1)
            
            # Top 3 percentiles average (elite skills)
            df_features['Top3_Percentile'] = df_features[percentile_cols].apply(
                lambda row: sorted(row, reverse=True)[:3],
                axis=1
            ).apply(lambda x: np.mean(x))
            
            # Consistency (std dev of percentiles)
            df_features['Percentile_Consistency'] = df_features[percentile_cols].std(axis=1)
        
        # Performance features
        if 'Gls/90' in df_features.columns:
            df_features['Goal_Contribution'] = df_features['Gls/90'] + df_features.get('Ast/90', 0)
        
        # Minutes played (reliability indicator)
        if '90s' in df_features.columns:
            df_features['Games_Played'] = df_features['90s']
            df_features['Is_Regular_Starter'] = (df_features['90s'] >= 20).astype(int)
        
        return df_features
    
    def train(
        self,
        df: pd.DataFrame,
        target_column: str = 'Transfermarkt_Value_£M',
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Dict:
        """
        Train ML model on Transfermarkt data.
        
        Args:
            df: DataFrame with features and target values
            target_column: Column containing actual market values
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dict with training metrics
        """
        # Filter to players with known values
        df_train = df[df[target_column].notna()].copy()
        
        if len(df_train) < 50:
            raise ValueError(f"Insufficient training data: only {len(df_train)} players with values")
        
        print(f"Training on {len(df_train)} players with Transfermarkt values...")
        
        # Engineer features
        df_train = self._engineer_features(df_train)
        
        # Select feature columns
        exclude_cols = [
            'Player', 'Squad', 'League', 'Primary_Pos', 'Archetype',
            target_column, 'Estimated_Value_£M', 'Value_Tier', 'Value_Score',
            'Transfermarkt_Value_£M', 'TM_Name', 'Cluster', 'PCA_X', 'PCA_Y',
            'Completeness_Score', 'Archetype_Confidence',
        ]
        
        self.feature_columns = [
            col for col in df_train.columns
            if col not in exclude_cols and df_train[col].dtype in ['int64', 'float64']
        ]
        
        # Prepare X and y
        X = df_train[self.feature_columns].fillna(0)
        y = df_train[target_column]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
            )
        else:  # gradient_boosting
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=random_state,
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': len(self.feature_columns),
        }
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(
                self.feature_columns,
                self.model.feature_importances_
            ))
            # Sort by importance
            self.feature_importance = dict(
                sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
        
        self.is_trained = True
        
        # Print results
        print(f"\n✓ Model trained successfully!")
        print(f"  Test MAE: £{metrics['test_mae']:.2f}M")
        print(f"  Test R²: {metrics['test_r2']:.3f}")
        print(f"  Test RMSE: £{metrics['test_rmse']:.2f}M")
        print(f"\nTop 5 Important Features:")
        for i, (feat, imp) in enumerate(list(self.feature_importance.items())[:5], 1):
            print(f"  {i}. {feat}: {imp:.3f}")
        
        return metrics
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict market values for players.
        
        Args:
            df: DataFrame with player features
            
        Returns:
            Array of predicted values (£M)
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Engineer features
        df_pred = self._engineer_features(df.copy())
        
        # Prepare features
        X = df_pred[self.feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        
        # Ensure non-negative
        predictions = np.maximum(predictions, 0.1)
        
        return predictions
    
    def analyze_value_premium(
        self,
        df: pd.DataFrame,
        actual_column: str = 'Transfermarkt_Value_£M',
    ) -> pd.DataFrame:
        """
        Analyze fair value vs market premium.
        
        Args:
            df: DataFrame with player data
            actual_column: Column with actual market values
            
        Returns:
            DataFrame with value analysis columns
        """
        df_analysis = df.copy()
        
        # Predict fair values
        df_analysis['Predicted_Value_£M'] = self.predict(df)
        
        # Calculate premium/discount
        if actual_column in df_analysis.columns:
            df_analysis['Value_Premium_%'] = (
                (df_analysis[actual_column] - df_analysis['Predicted_Value_£M']) /
                df_analysis['Predicted_Value_£M'] * 100
            )
            
            # Classify
            df_analysis['Value_Category'] = pd.cut(
                df_analysis['Value_Premium_%'],
                bins=[-np.inf, -20, -10, 10, 20, np.inf],
                labels=['Bargain', 'Undervalued', 'Fair', 'Overvalued', 'Premium']
            )
        
        return df_analysis
    
    def find_undervalued_bargains(
        self,
        df: pd.DataFrame,
        min_discount: float = 20.0,
        max_age: int = 26,
        top_n: int = 20,
    ) -> pd.DataFrame:
        """
        Find undervalued bargain players.
        
        Args:
            df: DataFrame with player data
            min_discount: Minimum discount percentage
            max_age: Maximum age for prospects
            top_n: Number of bargains to return
            
        Returns:
            DataFrame of undervalued players
        """
        df_analysis = self.analyze_value_premium(df)
        
        # Filter
        bargains = df_analysis[
            (df_analysis['Value_Premium_%'] <= -min_discount) &
            (df_analysis['Age'] <= max_age) &
            (df_analysis['Predicted_Value_£M'] >= 1.0)  # Minimum predicted value
        ].copy()
        
        # Sort by discount
        bargains = bargains.sort_values('Value_Premium_%', ascending=True)
        
        return bargains.head(top_n)
    
    def save_model(self, filepath: str = 'ml_value_model.pkl'):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'ml_value_model.pkl'):
        """Load trained model from file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.feature_importance = model_data.get('feature_importance', {})
        self.model_type = model_data.get('model_type', 'random_forest')
        self.is_trained = True
        
        print(f"✓ Model loaded from {filepath}")


def train_and_save_model(
    df: pd.DataFrame,
    output_path: str = 'ml_value_model.pkl',
) -> MLValuePredictor:
    """
    Convenience function to train and save model.
    
    Args:
        df: DataFrame with Transfermarkt values
        output_path: Path to save model
        
    Returns:
        Trained MLValuePredictor
    """
    predictor = MLValuePredictor(model_type='random_forest')
    predictor.train(df)
    predictor.save_model(output_path)
    return predictor
