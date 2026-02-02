"""
config_loader.py - Load and manage configuration from YAML file.

This module allows non-coders to adjust model parameters by editing config.yaml
without modifying Python code.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigLoader:
    """Load and cache configuration from config.yaml."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Singleton pattern - return same instance."""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize config loader."""
        if self._config is None:
            self.reload()
    
    def reload(self):
        """Load config from YAML file."""
        config_path = Path(__file__).parent.parent / 'config.yaml'
        
        if not config_path.exists():
            print(f"Config file not found at {config_path}")
            self._config = self._get_default_config()
            return
        
        try:
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f)
            print(f"Configuration loaded from {config_path}")
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            self._config = self._get_default_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get config value by dot-notation path.
        
        Args:
            key: Path to config value (e.g., 'profile_weights.attacker.Gls/90')
            default: Default value if key not found
            
        Returns:
            Config value or default
        """
        parts = key.split('.')
        value = self._config
        
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
                if value is None:
                    return default
            else:
                return default
        
        return value if value is not None else default
    
    def get_profile_weights(self) -> Dict[str, Dict[str, float]]:
        """Get position-specific weight matrix from config."""
        weights = self.get('profile_weights', {})
        
        # Convert snake_case keys to original format for compatibility
        return {
            'Attacker': weights.get('attacker', {}),
            'Midfielder': weights.get('midfielder', {}),
            'Defender': weights.get('defender', {}),
            'Goalkeeper': weights.get('goalkeeper', {}),
        }
    
    def get_league_base_values(self) -> Dict[str, float]:
        """Get league base values for market value calculation."""
        return self.get('market_value.league_base_values', {})
    
    def get_scout_bias(self) -> str:
        """Get current scout bias setting (Conservative/Neutral/Aggressive)."""
        return self.get('market_value.scout_bias', 'Neutral')
    
    def get_age_multipliers(self) -> Dict[str, float]:
        """Get age-based multipliers for market value."""
        return self.get('market_value.age_multipliers', {})
    
    def get_scouting_confidence_labels(self) -> Dict[str, Any]:
        """Get confidence label configuration."""
        return self.get('scouting_confidence', {})
    
    def get_hidden_gems_config(self) -> Dict[str, Any]:
        """Get hidden gems discovery settings."""
        return self.get('hidden_gems', {})
    
    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """
        Return default configuration if YAML file not found.
        This should match config.yaml defaults.
        """
        return {
            'profile_weights': {
                'attacker': {
                    'Gls/90': 3.0,
                    'Ast/90': 1.5,
                    'Sh/90': 2.5,
                    'SoT/90': 2.0,
                    'Crs/90': 1.0,
                    'Int/90': 0.3,
                    'TklW/90': 0.5,
                    'Fls/90': 1.0,
                    'Fld/90': 1.0,
                },
                'midfielder': {
                    'Gls/90': 0.5,
                    'Ast/90': 2.5,
                    'Sh/90': 0.8,
                    'SoT/90': 0.6,
                    'Crs/90': 2.0,
                    'Int/90': 1.5,
                    'TklW/90': 1.8,
                    'Fls/90': 1.2,
                    'Fld/90': 1.2,
                },
                'defender': {
                    'Gls/90': 0.2,
                    'Ast/90': 0.3,
                    'Sh/90': 0.1,
                    'SoT/90': 0.1,
                    'Crs/90': 0.5,
                    'Int/90': 2.5,
                    'TklW/90': 2.5,
                    'Fls/90': 1.5,
                    'Fld/90': 1.5,
                },
                'goalkeeper': {
                    'GA90': 2.5,
                    'Save%': 3.0,
                    'CS%': 2.0,
                    'W': 1.5,
                    'D': 1.0,
                    'L': 0.5,
                    'PKsv': 1.5,
                    'PKm': 1.0,
                    'Saves': 1.0,
                },
            },
            'market_value': {
                'scout_bias': 'Neutral',
                'age_multipliers': {
                    '18-21': 0.8,
                    '22-24': 1.0,
                    '25-28': 1.2,
                    '29-32': 0.95,
                    '33+': 0.6,
                },
                'league_base_values': {
                    'Premier League': 5.0,
                    'Championship': 2.5,
                    'League One': 1.2,
                    'League Two': 0.8,
                    'National League': 0.4,
                    'Bundesliga': 4.5,
                    'La Liga': 5.0,
                    'Serie A': 4.5,
                    'Ligue 1': 4.0,
                },
                'percentile_multipliers': {
                    'elite': 2.5,
                    'excellent': 2.0,
                    'very_good': 1.5,
                    'good': 1.0,
                    'below_average': 0.7,
                },
            },
            'scouting_confidence': {
                'elite_threshold': 90,
                'elite_label': 'Verified Elite Data',
                'good_threshold': 70,
                'good_label': 'Good Scouting Data',
                'directional_threshold': 40,
                'directional_label': 'Directional Data (Further Vetting Required)',
                'poor_threshold': 0,
                'poor_label': 'Incomplete Data - Caution Advised',
            },
            'hidden_gems': {
                'age_threshold': 23,
                'percentile_threshold': 80,
                'exclude_league': 'Premier League',
                'min_games': 10,
            },
            'clustering': {
                'n_clusters': 8,
                'random_state': 42,
                'pca_components': 2,
            },
            'ui': {
                'theme': 'dark',
                'fuzzy_search_limit': 10,
                'top_similar_players': 5,
                'leaderboard_size': 25,
            },
            'data_validation': {
                'min_minutes_played': 10,
                'min_players_per_group': 5,
                'data_completeness_warning': 30,
            },
        }


def get_config() -> ConfigLoader:
    """Get global config instance (singleton)."""
    return ConfigLoader()
