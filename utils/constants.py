"""
constants.py - Configuration constants for the Scouting Dashboard.

Includes:
- Feature columns (Outfield & GK)
- League definitions and colors
- Archetype definitions
- Scoring weights
- League Tiers and Metric Availability
"""

# ============================================================================
# 1. FEATURES & COLUMNS
# ============================================================================

# The core 13 metrics used for outfield analysis (Tier 1 standard)
FEATURE_COLUMNS = [
    'Gls/90', 'Ast/90', 'xG90', 'xA90', 
    'Sh/90', 'SoT/90', 'Crs/90', 'Drib/90', 
    'TklW/90', 'Int/90', 'Fls/90', 'Fld/90', 
    'AerWon/90'
]

# Supplementary columns that strictly relate to possession/progression
POSSESSION_FEATURES = ['PrgC', 'PrgP', 'Drib/90']
OFFENSIVE_FEATURES = ['Gls/90', 'Ast/90', 'xG90', 'xA90', 'Sh/90', 'SoT/90']

# Goalkeeper-specific metrics
GK_FEATURE_COLUMNS = [
    'PSxG+/-', 'Save%', 'CS%', 'GA90', 'L', 
    'CrsStp%', 'Swp/90', 'Launch%', 'PassLen'
]

# ============================================================================
# 2. LEAGUES & POSITIONS
# ============================================================================

LEAGUES = [
    'Premier League',
    'La Liga',
    'Bundesliga', 
    'Serie A',
    'Ligue 1',
    'Championship',
    'League One',
    'League Two',
    'National League'
]

LOW_DATA_LEAGUES = ['National League']
HIDDEN_GEMS_EXCLUDE_LEAGUE = ['Premier League']

PRIMARY_POSITIONS = ['FW', 'AM', 'CM', 'DM', 'FB', 'CB', 'GK']

LEAGUE_COLORS = {
    'Premier League': '#38003c', # Purple
    'La Liga': '#ee8707', # Orange
    'Bundesliga': '#d3010c', # Red
    'Serie A': '#008fd7', # Blue
    'Ligue 1': '#dae505', # Lime
    'Championship': '#EF3340', # Red
    'League One': '#B2B2B2', # Silver
    'League Two': '#7F7F7F', # Dark Gray
    'National League': '#333333', # Black
}

# ============================================================================
# 3. MAPPINGS & QUALITY
# ============================================================================

LEAGUE_METRIC_MAP = {
    # Standard map for renaming if needed, currently unused but good for Ref
}

LEAGUE_DATA_QUALITY = {
    'Premier League': 'High',
    'La Liga': 'High',
    'Bundesliga': 'High',
    'Serie A': 'High',
    'Ligue 1': 'High',
    'Championship': 'Medium',
    'League One': 'Medium',
    'League Two': 'Medium',
    'National League': 'Low'
}

# League Tiers for Data Availability (NEW)
LEAGUE_TO_TIER = {
    'Premier League': 'Tier 1',
    'La Liga': 'Tier 1',
    'Bundesliga': 'Tier 1',
    'Serie A': 'Tier 1',
    'Ligue 1': 'Tier 1',
    'Championship': 'Tier 2',
    'League One': 'Tier 3',
    'League Two': 'Tier 3',
    'National League': 'Tier 4'
}

# Integer Tier Mapping (for Narrative Generation)
LEAGUE_TIERS = {
    'Premier League': 1,
    'La Liga': 1,
    'Bundesliga': 1,
    'Serie A': 1,
    'Ligue 1': 1,
    'Championship': 2,
    'League One': 3,
    'League Two': 4,
    'National League': 5
}

# Define what metrics exist in which tier
LEAGUE_METRIC_AVAILABILITY = {
    'Tier 1': ['Gls/90', 'Ast/90', 'xG90', 'xA90', 'Sh/90', 'SoT/90', 'Crs/90', 'Drib/90', 'TklW/90'],  # Top 5 Leagues
    'Tier 2': ['Gls/90', 'Ast/90', 'Sh/90', 'SoT/90', 'CrdY', 'CrdR'],  # Championship
    'Tier 3': ['Gls/90', 'Ast/90', 'Sh/90', 'SoT/90', 'CrdY', 'CrdR'],  # League One/Two
    'Tier 4': ['Gls/90', 'Ast/90', 'CrdY', 'CrdR']  # National League
}

# Metric Proxies
METRIC_PROXIES = {
    'xG90': ('Gls/90', 'Using raw Goals instead of xG'),
    'xA90': ('Ast/90', 'Using raw Assists instead of xA'),
    'Crs/90': ('Ast/90', 'Using Assists as proxy for Crossing volume'),
    'Drib/90': ('Gls/90', 'Using Goal Threat as proxy for Dribbling ability (Low Confidence)'),
    'TklW/90': ('CrdY', 'Using Card history as proxy for Aggression (High Risk Proxy)'),
}

SCOUTING_PRIORITIES = {
    'Standard': {}, # Uses default weights
    'Clinical Finisher': {'Gls/90': 3.0, 'xG90': 2.0, 'Sh/90': 1.5},
    'Creative Hub': {'Ast/90': 3.0, 'xA90': 2.0},
    'Ball Progressor': {'Drib/90': 3.0, 'PrgC': 2.0},
    'Aerial Threat': {'AerWon/90': 3.0}
}


# ============================================================================
# 4. ARCHETYPES
# ============================================================================

ARCHETYPE_NAMES = [
    'Target Man', 'Creative Playmaker', 'Box-to-Box', 
    'Ball-Winning Midfielder', 'Aggressive Defender', 'Sweeper', 
    'Full-Back Playmaker', 'Buildup Boss', 
    'Elite Keeper', 'Shot-Stopper', 'Ball-Playing GK'
]

# Definitions for clustering logic and visualization
ARCHETYPES = {
    'Target Man': {
        'description': 'Aerial dominant forward who holds up play',
        'color': '#ff0000',
        'primary_position': 'FW',
        'key_metrics': ['AerWon/90', 'Gls/90']
    },
    'Creative Playmaker': {
        'description': 'Creator who operates in the half-spaces',
        'color': '#00ff00',
        'primary_position': 'MF',
        'key_metrics': ['Ast/90', 'xA90', 'KeyP/90']
    },
    'Box-to-Box': {
        'description': 'All-around midfielder contributing to both phases',
        'color': '#0000ff',
        'primary_position': 'MF',
        'key_metrics': ['TklW/90', 'Int/90', 'Gls/90']
    },
    'Ball-Winning Midfielder': {
        'description': 'Defensive specialist who breaks up play',
        'color': '#ffff00',
        'primary_position': 'MF',
        'key_metrics': ['TklW/90', 'Int/90']
    },
    'Aggressive Defender': {
        'description': 'Proactive defender who steps out to engage',
        'color': '#ff00ff',
        'primary_position': 'DF',
        'key_metrics': ['TklW/90', 'Int/90', 'Fls/90']
    },
    'Sweeper': {
        'description': 'Covering defender who cleans up behind',
        'color': '#00ffff',
        'primary_position': 'DF',
        'key_metrics': ['Int/90', 'Clr/90']
    },
    'Full-Back Playmaker': {
        'description': 'Wide defender who contributes to attack',
        'color': '#ff8800',
        'primary_position': 'DF',
        'key_metrics': ['Crs/90', 'xA90']
    },
    'Buildup Boss': {
        'description': 'Deep lying playmaker or ball-playing CB',
        'color': '#8800ff',
        'primary_position': 'MF', # or DF
        'key_metrics': ['PrgP', 'PassComp%']
    },
    'Elite Keeper': {
        'description': 'Top tier shot-stopper',
        'color': '#ffffff',
        'primary_position': 'GK',
        'key_metrics': ['PSxG+/-']
    },
    'Shot-Stopper': {
        'description': 'Keeper focused on saving',
        'color': '#aaaaaa',
        'primary_position': 'GK',
        'key_metrics': ['Save%']
    },
    'Ball-Playing GK': {
        'description': 'Keeper comfortable with feet',
        'color': '#333333',
        'primary_position': 'GK',
        'key_metrics': ['Launch%']
    }
}

# ============================================================================
# 5. UI CONFIG
# ============================================================================

METRIC_TOOLTIPS = {
    'Gls/90': 'Goals per 90 minutes',
    'Ast/90': 'Assists per 90 minutes',
    'xG90': 'Expected Goals per 90',
    'xA90': 'Expected Assists per 90'
}

PROFILE_WEIGHTS = {
    # Generic Profile Weights (used by similarity engine)
    'Attacker': {'Gls/90': 2.0, 'xG90': 2.0, 'Sh/90': 1.5},
    'Midfielder': {'Ast/90': 1.5, 'xA90': 1.5, 'TklW/90': 1.2},
    'Defender': {'TklW/90': 2.0, 'Int/90': 2.0, 'AerWon/90': 1.5},
    
    # Specific Overrides (if needed)
    'FW': {'Gls/90': 2.0, 'xG90': 2.0},
    'AM': {'Ast/90': 2.0, 'xA90': 2.0},
    'CB': {'Int/90': 2.0, 'AerWon/90': 1.5},
    'GK': {'PSxG+/-': 3.0, 'Save%': 2.0},
    'Goalkeeper': {'PSxG+/-': 3.0, 'Save%': 2.0}
}

# Threshold constants from data engine
MIN_MINUTES_PLAYED = 10.0
MIN_PLAYERS_PER_GROUP = 5
PERCENTILE_QUALITY_THRESHOLDS = {
    'High': 50,
    'Medium': 20,
    'Low': 0
}

# Radar chart labels
RADAR_LABELS = {
    'Gls/90': 'Goals',
    'Ast/90': 'Assists',
    'xG90': 'xG',
    'xA90': 'xA',
    'TklW/90': 'Tackles',
    'Int/90': 'Intercepts'
}
