"""
utils/__init__.py - Package initialization for utility modules

Exports commonly used classes and functions for easy importing:
    from utils import SimilarityEngine, process_all_data, cluster_players, FEATURE_COLUMNS
"""

from .constants import (
    FEATURE_COLUMNS,
    GK_FEATURE_COLUMNS,
    LEAGUE_METRIC_MAP,
    LEAGUE_COLORS,
    LEAGUES,
    PRIMARY_POSITIONS,
    ARCHETYPE_NAMES,
    PROFILE_WEIGHTS,
    METRIC_TOOLTIPS,
)

from .data_engine import (
    load_data,
    process_all_data,
    get_player_by_name,
    get_players_in_league,
    get_players_in_position,
    get_league_position_stats,
    get_percentile_for_player,
)

from .similarity import (
    SimilarityEngine,
    RadarChartGenerator,
    Position,
    ProfileType,
)

from .clustering import (
    cluster_players,
    PlayerArchetypeClusterer,
    get_archetype_color,
    get_archetype_description,
    get_archetypes_by_position,
)

from .narrative_generator import (
    ScoutNarrativeGenerator,
    generate_narrative_for_player,
)

from .market_value import (
    MarketValueEstimator,
    PricePerformanceAnalyzer,
    add_market_value_to_dataframe,
)

from .config_loader import (
    ConfigLoader,
    get_config,
)

from .age_curve_analysis import (
    AgeCurveAnalyzer,
    AgeCurveAnomaly,
    format_age_curve_badge,
)


__all__ = [
    # Constants
    'FEATURE_COLUMNS',
    'GK_FEATURE_COLUMNS',
    'LEAGUE_METRIC_MAP',
    'LEAGUE_COLORS',
    'LEAGUES',
    'PRIMARY_POSITIONS',
    'ARCHETYPE_NAMES',
    'PROFILE_WEIGHTS',
    'METRIC_TOOLTIPS',
    # Data Engine
    'load_data',
    'process_all_data',
    'get_player_by_name',
    'get_players_in_league',
    'get_players_in_position',
    'get_league_position_stats',
    'get_percentile_for_player',
    # Similarity
    'SimilarityEngine',
    'RadarChartGenerator',
    'Position',
    'ProfileType',
    # Clustering
    'cluster_players',
    'PlayerArchetypeClusterer',
    'get_archetype_color',
    'get_archetype_description',
    'get_archetypes_by_position',
    # Narrative Generator
    'ScoutNarrativeGenerator',
    'generate_narrative_for_player',
    # Market Value
    'MarketValueEstimator',
    'PricePerformanceAnalyzer',
    'add_market_value_to_dataframe',
    # Config
    'ConfigLoader',
    'get_config',
    # Age Curve Analysis
    'AgeCurveAnalyzer',
    'AgeCurveAnomaly',
    'format_age_curve_badge',
]

