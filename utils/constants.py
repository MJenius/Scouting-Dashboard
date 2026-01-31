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
    # Defenders
    'No-Nonsense Stopper', 'Ball-Playing Defender', 'Modern Full-Back', 'Wing-Back Creator',
    # Midfielders
    'Deep-Lying Playmaker', 'Ball-Winning Midfielder', 'Box-to-Box Engine', 'Advanced Playmaker',
    # Forwards/Attackers
    'Clinical Finisher', 'Target Man', 'Creative Winger', 'Inside Forward', 'Shadow Striker',
    # Goalkeepers
    'Elite Keeper', 'Shot-Stopper', 'Ball-Playing GK', 'Sweeper-Keeper'
]

# Definitions for clustering logic and visualization
ARCHETYPES = {
    # DEFENDERS
    'No-Nonsense Stopper': {
        'description': 'Traditional defender prioritized on clearances and physical dominance.',
        'color': '#ff0000',
        'primary_position': 'DF',
        'key_metrics': ['Clr/90', 'AerWon/90', 'TklW/90']
    },
    'Ball-Playing Defender': {
        'description': 'Modern center-back comfortable with possession and building from the back.',
        'color': '#ff4444',
        'primary_position': 'DF',
        'key_metrics': ['PrgP', 'PassComp%', 'Int/90']
    },
    'Modern Full-Back': {
        'description': 'Balanced wide defender contributing to both defense and ball progression.',
        'color': '#ff8888',
        'primary_position': 'DF',
        'key_metrics': ['Drib/90', 'PrgC', 'TklW/90']
    },
    'Wing-Back Creator': {
        'description': 'Highly offensive wide player who acts as a primary source of crosses and chances.',
        'color': '#ffaa00',
        'primary_position': 'DF',
        'key_metrics': ['Crs/90', 'xA90', 'Ast/90']
    },

    # MIDFIELDERS
    'Deep-Lying Playmaker': {
        'description': 'Midfielder who operates from deep to dictate the tempo with passing range.',
        'color': '#00ff00',
        'primary_position': 'MF',
        'key_metrics': ['PrgP', 'PassComp%', 'xA90']
    },
    'Ball-Winning Midfielder': {
        'description': 'Protective specialist focused on interceptions and breaking up opposition play.',
        'color': '#00cc00',
        'primary_position': 'MF',
        'key_metrics': ['TklW/90', 'Int/90', 'Fls/90']
    },
    'Box-to-Box Engine': {
        'description': 'High-workrate midfielder who covers the entire pitch in both phases.',
        'color': '#008800',
        'primary_position': 'MF',
        'key_metrics': ['PrgC', 'TklW/90', 'Gls/90']
    },
    'Advanced Playmaker': {
        'description': 'Creative hub operating between the lines to create high-value chances.',
        'color': '#00ffff',
        'primary_position': 'MF',
        'key_metrics': ['xA90', 'Ast/90', 'KeyP/90']
    },

    # FORWARDS / ATTACKERS
    'Clinical Finisher': {
        'description': 'Goal-focused forward who excels at converting shots and finding space in the box.',
        'color': '#0000ff',
        'primary_position': 'FW',
        'key_metrics': ['Gls/90', 'SoT/90', 'xG90']
    },
    'Target Man': {
        'description': 'Physical presence who excels in aerial duels and holding up the ball.',
        'color': '#000088',
        'primary_position': 'FW',
        'key_metrics': ['AerWon/90', 'Gls/90', 'Fld/90']
    },
    'Creative Winger': {
        'description': 'Wide attacker focused on beating markers and delivering high-quality crosses.',
        'color': '#4444ff',
        'primary_position': 'FW',
        'key_metrics': ['Drib/90', 'Crs/90', 'xA90']
    },
    'Inside Forward': {
        'description': 'Wide player who cuts inside onto their stronger foot to shoot or create.',
        'color': '#8888ff',
        'primary_position': 'FW',
        'key_metrics': ['Sh/90', 'PrgC', 'Gls/90']
    },
    'Shadow Striker': {
        'description': 'Advanced attacker who finds space behind the main forward to score.',
        'color': '#aa00ff',
        'primary_position': 'FW',
        'key_metrics': ['xG90', 'Gls/90', 'SoT/90']
    },

    # GOALKEEPERS
    'Elite Keeper': {
        'description': 'All-around top tier shot-stopper with high command of the area.',
        'color': '#ffffff',
        'primary_position': 'GK',
        'key_metrics': ['PSxG+/-', 'Save%', 'CS%']
    },
    'Shot-Stopper': {
        'description': 'Keeper who primarily excels at reactionary saves and goal prevention.',
        'color': '#aaaaaa',
        'primary_position': 'GK',
        'key_metrics': ['Save%', 'PSxG+/-']
    },
    'Ball-Playing GK': {
        'description': 'Modern keeper involved in buildup with high distribution accuracy.',
        'color': '#333333',
        'primary_position': 'GK',
        'key_metrics': ['PassLen', 'Launch%']
    },
    'Sweeper-Keeper': {
        'description': 'Proactive keeper who manages the space behind the defensive line.',
        'color': '#666666',
        'primary_position': 'GK',
        'key_metrics': ['Swp/90', 'CrsStp%']
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
