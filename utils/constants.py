"""
constants.py - Global configuration and feature definitions for the Scouting Dashboard.

This module defines:
- Feature columns for per-90 statistics
- League metadata and color schemes
- Position types and archetype definitions
- League-specific metric availability maps
"""

# ============================================================================
# CORE FEATURES: Per-90 Metrics used for player analysis
# ============================================================================
FEATURE_COLUMNS = [
    'Gls/90',    # Goals per 90 minutes
    'Ast/90',    # Assists per 90 minutes
    'Sh/90',     # Shots per 90 minutes
    'SoT/90',    # Shots on Target per 90 minutes
    'Crs/90',    # Crosses per 90 minutes
    'Int/90',    # Interceptions per 90 minutes
    'TklW/90',   # Tackles Won per 90 minutes
    'Fls/90',    # Fouls Committed per 90 minutes
    'Fld/90',    # Fouls Drawn per 90 minutes
]

# Goalkeeper-specific features (for comparison when both players are GK)
GK_FEATURE_COLUMNS = [
    'GA90',      # Goals Against per 90 minutes
    'Save%',     # Save percentage
    'CS%',       # Clean Sheet percentage
    'W',         # Wins
    'D',         # Draws
    'L',         # Losses
    'PKsv',      # Penalty kicks saved
    'PKm',       # Penalty kicks missed
    'Saves',     # Total saves
]

# Grouping for readability in UI
OFFENSIVE_FEATURES = ['Gls/90', 'Ast/90', 'Sh/90', 'SoT/90']
POSSESSION_FEATURES = ['Crs/90', 'Int/90', 'TklW/90', 'Fls/90', 'Fld/90']

# ============================================================================
# LEAGUE METADATA
# ============================================================================
LEAGUES = [
    'Premier League',
    'Championship',
    'League One',
    'League Two',
    'National League',
    'Bundesliga',
    'La Liga',
    'Serie A',
    'Ligue 1'
]

# League hierarchy and tier assignment
LEAGUE_TIERS = {
    'Premier League': 1,
    'Championship': 2,
    'League One': 3,
    'League Two': 4,
    'National League': 5,
    'Bundesliga': 1,
    'La Liga': 1,
    'Serie A': 1,
    'Ligue 1': 1,
}

# Color scheme for consistent visualization across dashboard
LEAGUE_COLORS = {
    'Premier League': '#003399',      # Dark Blue
    'Championship': '#EE1939',        # Red
    'League One': '#118C3B',          # Green
    'League Two': '#002868',          # Navy
    'National League': '#9B59B6',     # Purple
    'Bundesliga': '#D3010C',          # Red/Black
    'La Liga': '#FF6900',             # Orange
    'Serie A': '#024494',             # Blue
    'Ligue 1': '#003D7C',             # Dark Blue
}

# ============================================================================
# POSITION DEFINITIONS
# ============================================================================
POSITION_TYPES = {
    'FW': 'Forward',
    'MF': 'Midfielder',
    'DF': 'Defender',
    'GK': 'Goalkeeper',
}

# Primary positions (used for percentile grouping)
PRIMARY_POSITIONS = ['FW', 'MF', 'DF', 'GK']

# ============================================================================
# LEAGUE-METRIC AVAILABILITY MAP
# Decision (B): League-aware metric availability for completeness scoring
# ============================================================================
LEAGUE_METRIC_MAP = {
    'Premier League': FEATURE_COLUMNS,           # All 9 metrics
    'Championship': FEATURE_COLUMNS,             # All 9 metrics
    'League One': FEATURE_COLUMNS,               # All 9 metrics
    'League Two': FEATURE_COLUMNS,               # All 9 metrics
    'National League': ['Gls/90', 'Ast/90'],    # Only attacking metrics tracked
    'Bundesliga': FEATURE_COLUMNS,               # All 9 metrics
    'La Liga': FEATURE_COLUMNS,                  # All 9 metrics
    'Serie A': FEATURE_COLUMNS,                  # All 9 metrics
    'Ligue 1': FEATURE_COLUMNS,                  # All 9 metrics
}

# Detailed explanation of data availability per league
LEAGUE_DATA_QUALITY = {
    'Premier League': {
        'tier': 1,
        'data_completeness': 'Full',
        'defensive_stats': True,
        'shooting_stats': True,
        'sample_size': 'Large',
    },
    'Championship': {
        'tier': 2,
        'data_completeness': 'Full',
        'defensive_stats': True,
        'shooting_stats': True,
        'sample_size': 'Large',
    },
    'League One': {
        'tier': 3,
        'data_completeness': 'Full',
        'defensive_stats': True,
        'shooting_stats': True,
        'sample_size': 'Medium',
    },
    'League Two': {
        'tier': 4,
        'data_completeness': 'Full',
        'defensive_stats': True,
        'shooting_stats': True,
        'sample_size': 'Medium',
    },
    'National League': {
        'tier': 5,
        'data_completeness': 'Partial',
        'defensive_stats': False,
        'shooting_stats': False,
        'sample_size': 'Small',
        'note': 'Defensive metrics not tracked at this tier',
    },
    'Bundesliga': {
        'tier': 1,
        'data_completeness': 'Full',
        'defensive_stats': True,
        'shooting_stats': True,
        'sample_size': 'Large',
    },
    'La Liga': {
        'tier': 1,
        'data_completeness': 'Full',
        'defensive_stats': True,
        'shooting_stats': True,
        'sample_size': 'Large',
    },
    'Serie A': {
        'tier': 1,
        'data_completeness': 'Full',
        'defensive_stats': True,
        'shooting_stats': True,
        'sample_size': 'Large',
    },
    'Ligue 1': {
        'tier': 1,
        'data_completeness': 'Full',
        'defensive_stats': True,
        'shooting_stats': True,
        'sample_size': 'Large',
    },
}

# ============================================================================
# PLAYER ARCHETYPE DEFINITIONS
# K-Means clustering will assign players to one of 8 archetypes
# ============================================================================
ARCHETYPES = {
    'Target Man': {
        'description': 'Tall, physical forward. High aerial dominance, strong in hold-up play.',
        'primary_position': 'FW',
        'key_metrics': ['Gls/90', 'Fld/90'],
        'color': '#E74C3C',  # Red
    },
    'Creative Playmaker': {
        'description': 'Dynamic midfielder. High pass completion, creative passing range.',
        'primary_position': 'MF',
        'key_metrics': ['Ast/90', 'Crs/90'],
        'color': '#3498DB',  # Blue
    },
    'Ball-Winning Midfielder': {
        'description': 'Defensive midfielder. High tackle wins, interception rate.',
        'primary_position': 'MF',
        'key_metrics': ['TklW/90', 'Int/90'],
        'color': '#F39C12',  # Orange
    },
    'Box-to-Box': {
        'description': 'Versatile midfielder. Balanced attacking and defensive contributions.',
        'primary_position': 'MF',
        'key_metrics': ['Gls/90', 'Ast/90', 'TklW/90'],
        'color': '#9B59B6',  # Purple
    },
    'Full-Back Playmaker': {
        'description': 'Attacking full-back. High crossing and build-up play contribution.',
        'primary_position': 'DF',
        'key_metrics': ['Crs/90', 'Ast/90'],
        'color': '#1ABC9C',  # Teal
    },
    'Aggressive Defender': {
        'description': 'Aggressive defender. High tackle/interception rate, physical presence.',
        'primary_position': 'DF',
        'key_metrics': ['TklW/90', 'Int/90'],
        'color': '#E67E22',  # Dark Orange
    },
    'Sweeper': {
        'description': 'Covering defender. High interception rate, positioning-based defense.',
        'primary_position': 'DF',
        'key_metrics': ['Int/90'],
        'color': '#2C3E50',  # Dark Blue-Gray
    },
    # Goalkeeper Archetypes (separate clustering)
    'Shot-Stopper': {
        'description': 'Traditional goalkeeper. Elite shot-stopping, high save percentage.',
        'primary_position': 'GK',
        'key_metrics': ['Save%', 'Saves'],
        'color': '#34495E',  # Gray
    },
    'Sweeper-Keeper': {
        'description': 'Modern sweeper-keeper. Active outside box, good distribution.',
        'primary_position': 'GK',
        'key_metrics': ['Save%', 'CS%'],
        'color': '#16A085',  # Teal
    },
    'Ball-Playing GK': {
        'description': 'Ball-playing goalkeeper. Excellent distribution, comfortable with feet.',
        'primary_position': 'GK',
        'key_metrics': ['Save%', 'W'],
        'color': '#27AE60',  # Green
    },
    'Elite Keeper': {
        'description': 'All-round elite goalkeeper. Balanced across all GK metrics.',
        'primary_position': 'GK',
        'key_metrics': ['Save%', 'GA90', 'CS%'],
        'color': '#8E44AD',  # Purple
    },
}


ARCHETYPE_NAMES = list(ARCHETYPES.keys())

# ============================================================================
# POSITION-SPECIFIC WEIGHT MATRIX
# Used for weighted cosine similarity in player comparisons
# Higher weights = more important for similarity calculation
# ============================================================================
PROFILE_WEIGHTS = {
    'Attacker': {
        'Gls/90': 5.0,    # Goals are critical for attackers
        'Ast/90': 2.0,    # Assists matter but less than goals
        'Sh/90': 3.5,     # Shot volume is very important
        'SoT/90': 3.0,    # Shot accuracy is very important
        'Crs/90': 1.2,    # Crosses matter for some attackers
        'Int/90': 0.2,    # Defensive stats matter very little
        'TklW/90': 0.3,   # Defensive stats matter very little
        'Fls/90': 0.8,    # Discipline has some relevance
        'Fld/90': 1.5,    # Drawing fouls is relevant (dribbling)
    },
    'Midfielder': {
        'Gls/90': 1.0,    # Goals matter but not primary
        'Ast/90': 3.5,    # Assists are critical for midfielders
        'Sh/90': 1.2,     # Shot volume matters moderately
        'SoT/90': 1.0,    # Shot accuracy matters moderately
        'Crs/90': 2.5,    # Crossing is very important
        'Int/90': 2.0,    # Defensive contribution important
        'TklW/90': 2.2,   # Tackling is very important
        'Fls/90': 1.5,    # Discipline matters
        'Fld/90': 1.5,    # Drawing fouls matters
    },
    'Defender': {
        'Gls/90': 0.1,    # Goals matter very little for defenders
        'Ast/90': 0.2,    # Assists matter very little
        'Sh/90': 0.1,     # Shots matter very little
        'SoT/90': 0.1,    # Shot accuracy irrelevant
        'Crs/90': 0.8,    # Crossing matters for attacking full-backs
        'Int/90': 4.0,    # Interceptions are critical
        'TklW/90': 4.0,   # Tackles are critical
        'Fls/90': 2.0,    # Discipline is very important
        'Fld/90': 1.8,    # Drawing fouls matters (positioning)
    },
    'Goalkeeper': {
        'GA90': 2.5,      # Goals Against per 90 (lower is better)
        'Save%': 3.0,     # Save percentage (higher is better)
        'CS%': 2.0,       # Clean sheet percentage
        'W': 1.5,         # Wins
        'D': 1.0,         # Draws
        'L': 0.5,         # Losses (lower is better)
        'PKsv': 1.5,      # Penalty saves
        'PKm': 1.0,       # Penalty saves made
        'Saves': 1.0,     # Total saves
    },
}


# ============================================================================
# PERCENTILE QUALITY FLAGS
# Decision (B): Low sample size handling for percentile reliability
# ============================================================================
PERCENTILE_QUALITY_THRESHOLDS = {
    'High': 30,      # >= 30 players in position-league group
    'Medium': 10,    # 10-29 players
    'Low': 0,        # < 10 players (flagged for caution)
}

# ============================================================================
# DATA VALIDATION THRESHOLDS
# ============================================================================
MIN_MINUTES_PLAYED = 10  # Minimum 90s for statistical reliability
MIN_PLAYERS_PER_GROUP = 5  # Minimum players in position-league group for percentiles
DATA_COMPLETENESS_WARNING = 30  # Warn if player has < 30% completeness score

# ============================================================================
# UI/UX CONFIGURATION
# ============================================================================
PAGE_TITLES = {
    '1_🔍_Player_Search': 'Player Search',
    '2_⚔️_Head_to_Head': 'Head-to-Head Comparison',
    '3_💎_Hidden_Gems': 'Hidden Gems Discovery',
    '4_🏆_Leaderboards': 'League Leaderboards',
}

# Color scheme for completeness and percentile visualization
COMPLETENESS_COLORS = {
    'high': '#27AE60',    # Green (>70%)
    'medium': '#F39C12',  # Orange (30-70%)
    'low': '#E74C3C',     # Red (<30%)
}

PERCENTILE_COLORS = {
    'elite': '#C0392B',       # Dark Red (>90th percentile)
    'excellent': '#E74C3C',   # Red (80-90th)
    'very_good': '#F39C12',   # Orange (60-80th)
    'good': '#3498DB',        # Blue (40-60th)
    'below_average': '#95A5A6', # Gray (20-40th)
    'poor': '#7F8C8D',        # Dark Gray (<20th)
}

# ============================================================================
# HIDDEN GEMS CONFIGURATION
# ============================================================================
HIDDEN_GEMS_AGE_THRESHOLD = 23  # Max age for "young prospect"
HIDDEN_GEMS_PERCENTILE_THRESHOLD = 80  # Min percentile for "high priority"
HIDDEN_GEMS_EXCLUDE_LEAGUE = 'Premier League'  # Don't flag PL players as hidden gems

# ============================================================================
# TRANSFER RECOMMENDATION TIERS
# ============================================================================
TRANSFER_PRIORITY_TIERS = {
    'Elite': {
        'percentile_min': 90,
        'age_max': 25,
        'icon': '🌟',
        'color': '#E74C3C',
    },
    'High': {
        'percentile_min': 80,
        'age_max': 23,
        'icon': '⭐',
        'color': '#F39C12',
    },
    'Medium': {
        'percentile_min': 70,
        'age_max': 21,
        'icon': '✓',
        'color': '#3498DB',
    },
}

# ============================================================================
# METRIC TOOLTIPS & HELP TEXT
# Displayed in UI to help scouts understand each statistic
# ============================================================================
METRIC_TOOLTIPS = {
    'Gls/90': 'Goals scored per 90 minutes played. Key metric for forwards and attacking midfielders.',
    'Ast/90': 'Assists per 90 minutes. Measures creative output and goal-creating ability.',
    'Sh/90': 'Shots taken per 90 minutes. Indicates offensive involvement and risk-taking.',
    'SoT/90': 'Shots on target per 90 minutes. Higher accuracy/quality shot selection.',
    'Crs/90': 'Crosses delivered per 90 minutes. Critical for attacking players and set-piece specialists.',
    'Int/90': 'Interceptions per 90 minutes. Shows defensive positioning and reading of the game.',
    'TklW/90': 'Tackles won per 90 minutes. Direct tackle success. Lower than total tackles.',
    'Fls/90': 'Fouls committed per 90 minutes. Discipline indicator—higher fouls suggests physical play.',
    'Fld/90': 'Fouls drawn per 90 minutes. Shows dribbling ability and how opponents react.',
    'GA90': 'Goals against per 90 minutes (Goalkeepers). Lower is better. Affected by defense quality.',
    'Save%': 'Save percentage (Goalkeepers). Ratio of shots stopped to shots faced. Elite GKs: 70%+.',
    'CS%': 'Clean sheet percentage (Goalkeepers). Percentage of matches with zero goals conceded.',
    'Age': 'Player age. Peak performance typically 25-28 years old.',
    '90s': 'Total 90-minute equivalents played in season. Higher = more data reliability.',
    'Squad': 'Current club/team.',
    'League': 'Top-level league (PL, Championship, Bundesliga, etc.).',
}
