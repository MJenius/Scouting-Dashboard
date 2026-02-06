"""
app.py - Main Streamlit application entry point for Football Scouting Dashboard.

Multi-page app with global filters (Age, League, Position, Minutes) persisted via st.session_state.
Pages:
  - Player Search: Fuzzy search + similar players
  - Head-to-Head: Player comparison + radar charts
  - Hidden Gems: Metric filters + transfer recommendations
  - Leaderboards: Rankings by metric + archetype filters
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import unicodedata
from streamlit_option_menu import option_menu

# Load environment variables from .env file
load_dotenv()

# Import utilities
from utils import (
    process_all_data,
    SimilarityEngine,
    cluster_players,
    LEAGUES,
    PRIMARY_POSITIONS,
    ARCHETYPE_NAMES,
    LEAGUE_COLORS,
    METRIC_TOOLTIPS,
    LOW_DATA_LEAGUES,
    HIDDEN_GEMS_EXCLUDE_LEAGUE,
    SCOUTING_PRIORITIES,
    LEAGUE_TO_TIER,

    generate_narrative_for_player,
    generate_comparison_narrative,
)
from utils.visualizations import PlotlyVisualizations
from utils.recruitment_logic import project_to_tier
from utils.llm_integration import AgenticScoutChat, is_ollama_available, generate_llm_narrative
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, List, Any

# Import API client for FastAPI backend integration
try:
    from frontend.api_client import (
        get_players,
        get_player_by_id,
        search_players as api_search_players,
        get_similar_players as api_get_similar,
        check_backend_health,
        is_backend_available,
        APIResponse,
    )
    API_CLIENT_AVAILABLE = True
except ImportError:
    API_CLIENT_AVAILABLE = False

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Football Scouting Dashboard",
    page_icon="Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }
    .stMetric {
        background-color: #2a2a2a;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.data_result = None
    st.session_state.engine = None
    st.session_state.clusterer = None
    st.session_state.df_clustered = None

if 'filters' not in st.session_state:
    st.session_state.filters = {
        'age_min': 18,
        'age_max': 35,
        'leagues': LEAGUES,
        'positions': PRIMARY_POSITIONS,
        'min_90s': 5,
        'metrics': {}  # Dynamic metric filters (e.g., {'Gls/90': 0.5})
    }

if 'page' not in st.session_state:
    st.session_state.page = 'Player Search'

# Track backend availability
if 'backend_available' not in st.session_state:
    st.session_state.backend_available = False
    if API_CLIENT_AVAILABLE:
        try:
            st.session_state.backend_available = is_backend_available()
        except:
            st.session_state.backend_available = False

# ============================================================================
# DATA LOADING (CACHED)
# ============================================================================


@st.cache_data
def load_all_data():
    """Load and process all data once per session."""
    result = process_all_data('english_football_pyramid_master.csv', min_90s=0)
    df = result['dataframe']
    scaled = result['scaled_features']
    scalers = result['scalers']
    engine = SimilarityEngine(df, scaled, scalers)
    return result, engine, df, scaled

# Separate clustering resource
@st.cache_data
def get_clustered_players(df, scaled):
    df_clustered, clusterer = cluster_players(df, scaled)

    return df_clustered, clusterer

def ensure_data_loaded():
    """Ensure data is loaded into session state."""
    if not st.session_state.data_loaded:
        with st.spinner("Loading and processing data..."):
            result, engine, df, scaled = load_all_data()
            df_clustered, clusterer = get_clustered_players(df, scaled)
            st.session_state.data_result = result
            st.session_state.engine = engine
            st.session_state.clusterer = clusterer
            st.session_state.df_clustered = df_clustered
            st.session_state.data_loaded = True

# ============================================================================
# SIDEBAR - GLOBAL FILTERS
# ============================================================================

with st.sidebar:
    st.title("Scouting Dashboard")
    
    # Navigation Menu
    menu_options = ['Player Search', 'Head-to-Head', 'Hidden Gems', 'Leaderboards', 'Squad Analysis', 'Squad Planner']
    
    # Get index of current page for manual_select (must be integer)
    try:
        current_page_index = menu_options.index(st.session_state.page)
    except (ValueError, AttributeError):
        current_page_index = 0

    selected_page = option_menu(
        menu_title=None,
        options=menu_options,
        icons=['search', 'intersect', 'gem', 'trophy', 'bar-chart-line', 'clipboard-check'],
        menu_icon="cast",
        default_index=current_page_index,
        manual_select=current_page_index,
        key='sidebar_nav'
    )
    
    # Update session state page immediately
    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()

def normalize_name(name: str) -> str:
    """
    Standardize player/squad names by removing accents and special characters (like Ø, Æ).
    """
    if not isinstance(name, str):
        return ""
    
    # Manual mapping for non-decomposable characters common in football
    special_map = {
        'Ø': 'O', 'ø': 'o',
        'Æ': 'AE', 'æ': 'ae',
        'Å': 'A', 'å': 'a',
        'ẞ': 'SS', 'ß': 'ss',
        'Ð': 'D', 'ð': 'd',
        'Þ': 'TH', 'þ': 'th',
        'Ĳ': 'IJ', 'ĳ': 'ij',
        'Ł': 'L', 'ł': 'l',
        'Ń': 'N', 'ń': 'n',
        'Œ': 'OE', 'œ': 'oe',
        'Š': 'S', 'š': 's',
        'Ÿ': 'Y', 'ÿ': 'y',
        'Ž': 'Z', 'ž': 'z',
    }
    for char, replacement in special_map.items():
        name = name.replace(char, replacement)
        
    # Normalize to decomposed form
    nfkd_form = unicodedata.normalize('NFKD', name)
    # Filter out non-spacing mark characters (accents)
    plain_text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return plain_text.casefold().replace('-', ' ').replace("'", "").strip()

    st.divider()
    st.subheader("Filters & Settings")
    
    # Age filter
    st.subheader("Age Range")
    age_min, age_max = st.slider(
        "Select age range:",
        min_value=16,
        max_value=40,
        value=(st.session_state.filters['age_min'], st.session_state.filters['age_max']),
        key='age_slider'
    )
    st.session_state.filters['age_min'] = age_min
    st.session_state.filters['age_max'] = age_max
    
    # League filter
    st.subheader("Leagues")
    selected_leagues = st.multiselect(
        "Select leagues:",
        options=LEAGUES,
        default=st.session_state.filters['leagues'],
        key='league_select'
    )
    st.session_state.filters['leagues'] = selected_leagues
    
    # Position filter
    st.subheader("Positions")
    selected_positions = st.multiselect(
        "Select positions:",
        options=PRIMARY_POSITIONS,
        default=st.session_state.filters['positions'],
        key='position_select'
    )
    st.session_state.filters['positions'] = selected_positions
    
    st.subheader("Minimum Minutes Played")
    min_90s = st.slider(
        "Min 90s:",
        min_value=0,
        max_value=30,
        value=int(st.session_state.filters['min_90s']),
        step=1,
        key='min_90s_slider'
    )
    st.session_state.filters['min_90s'] = min_90s
    
    st.divider()
    
    # Info panel
    st.subheader("Dataset Info")
    if st.session_state.data_loaded:
        df = st.session_state.df_clustered
        # Apply filters
        filtered = df[
            (df['Age'] >= age_min) &
            (df['Age'] <= age_max) &
            (df['League'].isin(selected_leagues)) &
            (df['Primary_Pos'].isin(selected_positions)) &
            (df['90s'] >= min_90s)
        ]
        
        # Apply metric filters if they exist
        if 'metrics' in st.session_state.filters:
            for metric, min_val in st.session_state.filters['metrics'].items():
                if metric in filtered.columns:
                    filtered = filtered[filtered[metric] >= min_val]
        
        st.metric("Players (Filtered)", len(filtered))
        st.metric("Leagues", len(selected_leagues))
        st.metric("Positions", len(selected_positions))
        
        if len(filtered) > 0:
            st.metric("Avg Age", f"{filtered['Age'].mean():.1f}")
            st.metric("Avg Goals/90", f"{filtered['Gls/90'].mean():.2f}")

    st.divider()
    
    # Backend connection status indicator
    st.subheader("API Status")
    if API_CLIENT_AVAILABLE and st.session_state.backend_available:
        st.success("FastAPI Backend Connected")
        try:
            health_ok, health_data = check_backend_health()
            if health_ok:
                st.caption(f"Database contains {health_data.get('player_count', '?')} players")
                if health_data.get('engine_loaded'):
                    st.caption("Similarity Engine: Active")
        except:
            pass
    elif API_CLIENT_AVAILABLE:
        st.warning("Backend Unavailable")
        st.caption("Using local fallback mode")
        if st.button("Retry Connection", key='retry_backend'):
            try:
                st.session_state.backend_available = is_backend_available()
                st.rerun()
            except:
                pass
    else:
        st.info("Running in Local Mode")
        st.caption("API client not installed")
    
    # Local AI Status
    st.divider()
    st.subheader("Local AI (Ollama)")
    
    if is_ollama_available():
        st.success("Local AI: Online")
        st.caption("Ready for Agentic Chat & Narratives")
    else:
        st.error("Local AI: Offline")
        st.caption("Start Ollama to enable AI features")
    
    st.divider()
    if st.button("Reset Cache & Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.data_loaded = False
        st.rerun()

# ============================================================================
# SQUAD ANALYSIS & PLANNER HELPERS
# ============================================================================

# 4-3-3 Coordinates for Plotly Pitch (0-100 scale)
PITCH_COORDS = {
    'GK':  (50, 5),
    'RB':  (85, 25), 'RCB': (60, 25), 'LCB': (40, 25), 'LB': (15, 25),
    'CDM': (50, 45), 'RCM': (70, 60), 'LCM': (30, 60),
    'RW':  (85, 85), 'ST':  (50, 90), 'LW':  (15, 85)
}

def get_squad_roster(df: pd.DataFrame, squad_name: str) -> pd.DataFrame:
    """Get all players in a specific squad."""
    return df[df['Squad'] == squad_name].copy()

def get_archetype_distribution(squad_df: pd.DataFrame) -> Dict[str, int]:
    """Calculate archetype distribution for a squad."""
    if 'Archetype' not in squad_df.columns:
        return {}
    return squad_df['Archetype'].value_counts().to_dict()

def get_squad_top_11(squad_df: pd.DataFrame) -> pd.DataFrame:
    """Get top 11 players by minutes played (90s)."""
    return squad_df.nlargest(11, '90s')

def calculate_squad_mean_percentiles(squad_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate mean percentile scores for the squad's top 11."""
    top_11 = get_squad_top_11(squad_df)
    
    # Only calculate for outfield players
    outfield = top_11[top_11['Primary_Pos'] != 'GK']
    
    percentiles = {}
    from utils.constants import FEATURE_COLUMNS # Ensure this is available
    for feat in FEATURE_COLUMNS[:9]:  # Core 9 metrics for comparison
        pct_col = f'{feat}_pct'
        if pct_col in outfield.columns:
            percentiles[feat] = outfield[pct_col].mean()
    
    return percentiles

def calculate_league_mean_percentiles(df: pd.DataFrame, league: str) -> Dict[str, float]:
    """Calculate mean percentile scores for the entire league."""
    league_df = df[df['League'] == league]
    outfield = league_df[league_df['Primary_Pos'] != 'GK']
    
    percentiles = {}
    from utils.constants import FEATURE_COLUMNS
    for feat in FEATURE_COLUMNS[:9]:
        pct_col = f'{feat}_pct'
        if pct_col in outfield.columns:
            percentiles[feat] = outfield[pct_col].mean()
    
    return percentiles

def calculate_squad_dominance_summary(squad_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate average dominance scores for the squad."""
    dominance_cols = [col for col in squad_df.columns if '_Dominance' in col]
    
    summary = {}
    for col in dominance_cols:
        metric_name = col.replace('_Dominance', '')
        summary[metric_name] = squad_df[col].mean()
    
    return summary

def create_archetype_pie_chart(archetype_dist: Dict[str, int], squad_name: str) -> go.Figure:
    """Create a pie chart of archetype distribution."""
    if not archetype_dist:
        return None
    
    labels = list(archetype_dist.keys())
    values = list(archetype_dist.values())
    
    # Color palette
    colors = px.colors.qualitative.Set3[:len(labels)]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='outside',
        pull=[0.05 if v == max(values) else 0 for v in values]
    )])
    
    fig.update_layout(
        title=dict(
            text=f"Tactical DNA: {squad_name}",
            x=0.5,
            font=dict(size=18)
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        height=450,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_age_histogram(squad_df: pd.DataFrame, league_df: pd.DataFrame, squad_name: str) -> go.Figure:
    """Create age histogram comparing squad to league average."""
    fig = go.Figure()
    
    # Squad ages
    fig.add_trace(go.Histogram(
        x=squad_df['Age'],
        name=squad_name,
        opacity=0.7,
        marker_color='#4CAF50',
        nbinsx=15
    ))
    
    # League ages
    fig.add_trace(go.Histogram(
        x=league_df['Age'],
        name='League Average',
        opacity=0.4,
        marker_color='#2196F3',
        nbinsx=15
    ))
    
    # Add vertical lines for means
    squad_mean = squad_df['Age'].mean()
    league_mean = league_df['Age'].mean()
    
    fig.add_vline(x=squad_mean, line_dash="dash", line_color="#4CAF50",
                  annotation_text=f"Squad: {squad_mean:.1f}")
    fig.add_vline(x=league_mean, line_dash="dash", line_color="#2196F3",
                  annotation_text=f"League: {league_mean:.1f}")
    
    fig.update_layout(
        title=dict(
            text=f"Age Distribution: {squad_name} vs League",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title="Age",
        yaxis_title="Number of Players",
        barmode='overlay',
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_squad_radar(squad_pcts: Dict[str, float], league_pcts: Dict[str, float], squad_name: str) -> go.Figure:
    """Create radar chart comparing squad to league averages."""
    categories = list(squad_pcts.keys())
    
    fig = go.Figure()
    
    # Squad values
    squad_values = [squad_pcts.get(cat, 50) for cat in categories]
    squad_values.append(squad_values[0])  # Close the polygon
    
    # League values
    league_values = [league_pcts.get(cat, 50) for cat in categories]
    league_values.append(league_values[0])
    
    categories_closed = categories + [categories[0]]
    
    fig.add_trace(go.Scatterpolar(
        r=squad_values,
        theta=categories_closed,
        fill='toself',
        name=f'{squad_name} (Top 11)',
        line_color='#4CAF50',
        fillcolor='rgba(76, 175, 80, 0.3)'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=league_values,
        theta=categories_closed,
        fill='toself',
        name='League Average',
        line_color='#2196F3',
        fillcolor='rgba(33, 150, 243, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        title=dict(
            text=f"Squad Profile: {squad_name}",
            x=0.5,
            font=dict(size=16)
        ),
        showlegend=True,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_dominance_bar_chart(dominance_summary: Dict[str, float], squad_name: str) -> go.Figure:
    """Create bar chart of squad's average dominance scores."""
    if not dominance_summary:
        return None
    
    metrics = list(dominance_summary.keys())
    values = list(dominance_summary.values())
    
    # Color based on positive/negative
    colors = ['#4CAF50' if v > 0 else '#f44336' for v in values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            text=[f'{v:.2f}' for v in values],
            textposition='outside'
        )
    ])
    
    fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=1)
    
    fig.update_layout(
        title=dict(
            text=f"League Dominance: {squad_name}",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis_title="Metric",
        yaxis_title="Dominance Z-Score",
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

def create_pitch_visualization(shadow_squad):
    """Create a football pitch with player markers."""
    fig = go.Figure()
    
    # Draw Pitch Outline
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100,
                  line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)")
    
    # Half-way line
    fig.add_shape(type="line", x0=0, y0=50, x1=100, y1=50,
                  line=dict(color="white", width=1, dash="dot"))
    
    # Penalty Areas (approx)
    fig.add_shape(type="rect", x0=20, y0=0, x1=80, y1=16,
                  line=dict(color="white", width=1))
    fig.add_shape(type="rect", x0=20, y0=84, x1=80, y1=100,
                  line=dict(color="white", width=1))
    
    # Player Markers
    x_vals = []
    y_vals = []
    text_vals = []
    colors = []
    hover_texts = []
    
    for pos, player_data in shadow_squad.items():
        x, y = PITCH_COORDS.get(pos, (50, 50))
        x_vals.append(x)
        y_vals.append(y)
        
        if player_data:
            name = player_data['Player']
            squad = player_data.get('Squad', 'Unknown')
            age = player_data.get('Age', '')
            # Dom score if available
            dom = player_data.get('Gls/90_Dominance', 0) if pos in ['ST', 'RW', 'LW'] else 0
            
            text_vals.append(f"<b>{pos}</b><br>{name}")
            colors.append('#4CAF50')  # Green for filled
            hover_texts.append(f"{name} ({age})<br>{squad}<br>Dominance: {dom:.2f}")
        else:
            text_vals.append(f"<b>{pos}</b><br>Empty")
            colors.append('#333333')  # Grey for empty
            hover_texts.append("Click to add player")
            
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers+text',
        marker=dict(size=40, color=colors, line=dict(color='white', width=1)),
        text=text_vals,
        textposition="bottom center",
        textfont=dict(color='white'),
        hovertext=hover_texts,
        hoverinfo='text'
    ))
    
    fig.update_layout(
        xaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-5, 105]),
        yaxis=dict(showgrid=False, zeroline=False, visible=False, range=[-5, 105]),
        height=600,
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False
    )
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def apply_filters(df):
    """Apply global filters to dataframe."""
    filters = st.session_state.filters
    df_out = df[
        (df['Age'] >= filters['age_min']) &
        (df['Age'] <= filters['age_max']) &
        (df['League'].isin(filters['leagues'])) &
        (df['Primary_Pos'].isin(filters['positions'])) &
        (df['90s'] >= filters['min_90s'])
    ]
    
    # Apply metric filters
    if 'metrics' in filters:
        for metric, min_val in filters['metrics'].items():
            if metric in df_out.columns:
                df_out = df_out[df_out[metric] >= min_val]
                
    return df_out

# Ensure data is loaded
ensure_data_loaded()

# Get data
df = st.session_state.df_clustered
engine = st.session_state.engine

# Apply filters
df_filtered = apply_filters(df)

# Apply filters
df_filtered = apply_filters(df)

# Header
st.title("Football Scouting Dashboard")
st.caption(f"Multi-League Global Scouting Dashboard | {len(df_filtered):,} players after filters")
st.divider()

# ============================================================================
# GLOBAL AGENTIC CHAT INTERFACE (Processes intents BEFORE page rendering)
# ============================================================================

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

def dispatch_agentic_action(api_params: Dict[str, Any]) -> str:
    """
    Route the AI's intent to the appropriate page and apply filters.
    Returns a response message describing what was done.
    """
    action = api_params.get('action')
    target_page = api_params.get('target_page')
    
    response_parts = []
    needs_rerun = False
    
    # Helper for exact choice matching (Leagues, Priorities, Positions)
    def find_exact_choice(query: str, options: List[str]) -> str:
        if not query: return None
        query_norm = normalize_name(query)
        for opt in options:
            opt_norm = normalize_name(opt)
            if query_norm == opt_norm or query_norm in opt_norm:
                return opt
        return None

    def find_best_match(query: str, options: List[str]) -> str:
        """Find the best matching player option using substring matching and priority scoring."""
        if not query: return None
        query_norm = normalize_name(query)
        
        matches = []
        for opt in options:
            opt_name = opt.split(' (')[0]
            opt_norm = normalize_name(opt_name)
            
            if query_norm == opt_norm:
                return opt # Exact match found
            
            if opt_norm.startswith(query_norm):
                # Penalty based on how much longer the name is than the query
                score = 1 + (len(opt_norm) - len(query_norm)) * 0.1
                matches.append((opt, score))
            elif query_norm in opt_norm:
                # Higher score (lower priority) for substring matches
                score = 5 + (len(opt_norm) - len(query_norm)) * 0.1
                matches.append((opt, score))
        
        if matches:
            # Return the match with the lowest score (highest priority)
            matches.sort(key=lambda x: x[1])
            return matches[0][0]
            
        return None

    # Navigate to target page
    valid_pages = ['Leaderboards', 'Head-to-Head', 'Player Search', 'Hidden Gems', 'Squad Analysis', 'Squad Planner']
    if target_page and target_page in valid_pages:
        st.session_state.page = target_page
        response_parts.append(f"Navigating to **{target_page}**")
        needs_rerun = True
    
    # Apply page-specific filters based on action
    if action == 'leaderboard':
        # Set Leaderboards page filters
        if 'league' in api_params:
            matched = find_exact_choice(api_params['league'], LEAGUES)
            if matched:
                st.session_state.leaderboard_league = matched
                response_parts.append(f"League: {matched}")
        if 'position' in api_params:
            matched = find_exact_choice(api_params['position'], PRIMARY_POSITIONS)
            if matched:
                st.session_state.leaderboard_position = matched
                response_parts.append(f"Position: {matched}")
        if 'metric' in api_params:
            st.session_state.leaderboard_metric = api_params['metric']
            response_parts.append(f"Metric: {api_params['metric']}")
        needs_rerun = True
    
    elif action == 'compare':
        # Set Head-to-Head page filters  
        # Player selection uses "Player (Squad)" format - we need to fuzzy match
        df = st.session_state.df_clustered
        player_options = [f"{row['Player']} ({row['Squad']})" for _, row in df.iterrows()]
        player_options = sorted(list(set(player_options)))
        
        if 'player_name' in api_params:
            matched = find_best_match(api_params['player_name'], player_options)
            if matched:
                st.session_state.player1_select = matched
                response_parts.append(f"Player 1: {matched}")
        if 'compare_player' in api_params:
            matched = find_best_match(api_params['compare_player'], player_options)
            if matched:
                st.session_state.player2_select = matched
                response_parts.append(f"Player 2: {matched}")
        needs_rerun = True
    
    elif action == 'hidden_gems':
        # Set Hidden Gems page filters
        if 'max_age' in api_params:
            st.session_state.gems_max_age = api_params['max_age']
            response_parts.append(f"Max Age: {api_params['max_age']}")
        needs_rerun = True
    
    elif action in ['search', 'find_similar']:
        # Set Player Search or Hidden Gems page filters
        if target_page == 'Hidden Gems':
            # Mode Selection
            st.session_state.gems_search_mode = "Benchmark (Player Match)"
            
            df = st.session_state.df_clustered
            player_options = [f"{row['Player']} ({row['Squad']})" for _, row in df.iterrows()]
            player_options = sorted(list(set(player_options)))
            
            if 'player_name' in api_params:
                matched = find_best_match(api_params['player_name'], player_options)
                if matched:
                    st.session_state.benchmark_player = matched
                    response_parts.append(f"Benchmark: {matched}")
            
            # Normalize League
            league_query = api_params.get('league') or api_params.get('target_league')
            if league_query:
                matched_league = find_exact_choice(league_query, LEAGUES)
                if matched_league:
                    st.session_state.benchmark_target_league = matched_league
                    response_parts.append(f"Target League: {matched_league}")
                
            # Normalize Priority
            if 'priority' in api_params:
                matched_priority = find_exact_choice(api_params['priority'], list(SCOUTING_PRIORITIES.keys()))
                if matched_priority:
                    st.session_state.benchmark_priority = matched_priority
                    response_parts.append(f"Priority: {matched_priority}")
        else:
            # Standard Player Search
            if 'player_name' in api_params:
                st.session_state.player_search_query = api_params['player_name']
                response_parts.append(f"Searching: {api_params['player_name']}")
        needs_rerun = True
    
    elif action == 'squad_analysis':
        # Set Squad Analysis page - pre-fill team search
        if 'team_name' in api_params:
            st.session_state.squad_search = api_params['team_name']
            response_parts.append(f"Team: {api_params['team_name']}")
        needs_rerun = True
    
    elif action == 'squad_planner':
        # Set Squad Planner page - add players to squad
        if 'squad_players' in api_params:
            players = api_params['squad_players']
            df = st.session_state.df_clustered
            
            # Clear existing squad and start fresh
            st.session_state.shadow_squad = {
                'GK': None, 'RB': None, 'LB': None, 'RCB': None, 'LCB': None,
                'CDM': None, 'RCM': None, 'LCM': None,
                'RW': None, 'ST': None, 'LW': None
            }
            
            def get_best_slot(pos_string: str) -> str:
                """Determine best slot from raw Pos column like 'DF,FB' or 'FW,RW'"""
                if pd.isna(pos_string):
                    return 'RCM'
                pos_str = str(pos_string).upper()
                
                # Check for specific positions in the raw string
                if 'GK' in pos_str:
                    return 'GK'
                if 'LB' in pos_str:
                    return 'LB'
                if 'RB' in pos_str or 'FB' in pos_str:
                    return 'RB'
                if 'CB' in pos_str:
                    return 'RCB'
                if 'LW' in pos_str:
                    return 'LW'
                if 'RW' in pos_str:
                    return 'RW'
                if 'DM' in pos_str:
                    return 'CDM'
                if 'AM' in pos_str or 'CM' in pos_str:
                    return 'RCM'
                if 'FW' in pos_str or 'ST' in pos_str:
                    return 'ST'
                if 'DF' in pos_str:
                    return 'RCB'
                if 'MF' in pos_str:
                    return 'RCM'
                return 'RCM'  # Default
            
            def normalize_for_match(name: str) -> str:
                """Normalize player name for matching by removing accents and converting to lowercase."""
                return normalize_name(name)
            
            added_players = []
            not_found = []
            
            for player_query in players:
                # Normalize query
                query_norm = normalize_for_match(player_query)
                
                # Try exact substring match first
                matches = df[df['Player'].apply(normalize_for_match).str.contains(query_norm, regex=False)]
                
                if matches.empty:
                    # Try partial word match
                    query_words = query_norm.split()
                    for word in query_words:
                        if len(word) > 3:  # Skip short words
                            matches = df[df['Player'].apply(normalize_for_match).str.contains(word, regex=False)]
                            if not matches.empty:
                                break
                
                if not matches.empty:
                    player_row = matches.iloc[0]
                    player_obj = player_row.to_dict()
                    
                    # Get position from raw Pos column
                    raw_pos = player_row.get('Pos', player_row.get('Primary_Pos', 'MF'))
                    slot = get_best_slot(raw_pos)
                    
                    # Find empty slot if preferred is taken
                    if st.session_state.shadow_squad.get(slot) is not None:
                        # Try to find alternative slot based on position group
                        pos_alternatives = {
                            'GK': ['GK'],
                            'RB': ['RB', 'LB', 'RCB', 'LCB'],
                            'LB': ['LB', 'RB', 'LCB', 'RCB'],
                            'RCB': ['RCB', 'LCB', 'RB', 'LB'],
                            'LCB': ['LCB', 'RCB', 'LB', 'RB'],
                            'CDM': ['CDM', 'RCM', 'LCM'],
                            'RCM': ['RCM', 'LCM', 'CDM'],
                            'LCM': ['LCM', 'RCM', 'CDM'],
                            'RW': ['RW', 'ST', 'LW'],
                            'LW': ['LW', 'ST', 'RW'],
                            'ST': ['ST', 'RW', 'LW']
                        }
                        alternatives = pos_alternatives.get(slot, list(st.session_state.shadow_squad.keys()))
                        for alt_slot in alternatives:
                            if st.session_state.shadow_squad.get(alt_slot) is None:
                                slot = alt_slot
                                break
                    
                    st.session_state.shadow_squad[slot] = player_obj
                    added_players.append(f"{player_obj['Player']} -> {slot}")
                else:
                    not_found.append(player_query)
            
            if added_players:
                response_parts.append(f"Added: {', '.join(added_players)}")
            if not_found:
                response_parts.append(f"Not found: {', '.join(not_found)}")
        needs_rerun = True
    
    # Also apply global filters (age, league, position for sidebar)
    # Also apply global filters (age, league, position for sidebar)
    if 'min_age' in api_params: 
        try: st.session_state.filters['age_min'] = int(api_params['min_age'])
        except: pass
    if 'max_age' in api_params: 
        try: st.session_state.filters['age_max'] = int(api_params['max_age'])
        except: pass
    
    if 'league' in api_params: 
        matched = find_exact_choice(api_params['league'], LEAGUES)
        if matched:
            st.session_state.filters['leagues'] = [matched]
    
    if 'position' in api_params:
        pos = api_params['position']
        # Map generic Chat positions to Dashboard specific positions
        if pos == 'DF':
            st.session_state.filters['positions'] = ['CB', 'FB']
        elif pos == 'MF':
            st.session_state.filters['positions'] = ['DM', 'CM', 'AM']
        else:
            matched = find_exact_choice(pos, PRIMARY_POSITIONS)
            if matched:
                st.session_state.filters['positions'] = [matched]
    
    return " | ".join(response_parts) if response_parts else "Filters updated", needs_rerun

# Global Chat Expander
with st.expander("Agentic Scout Chat", expanded=False):
    st.caption("Ask questions like 'Find me the best striker in Serie A' or 'Compare Haaland with Mbappe'")
    
    # Scrollable chat container with fixed height
    chat_container = st.container(height=300)
    
    # Function to display chat messages
    def display_chat_messages():
        # Only show last 10 messages to keep UI clean
        messages_to_show = st.session_state.chat_messages[-10:] if len(st.session_state.chat_messages) > 10 else st.session_state.chat_messages
        for msg in messages_to_show:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Render current messages
    with chat_container:
        display_chat_messages()
    
    # Chat input (outside scrollable container)
    if prompt := st.chat_input("Ask the AI Scout..."):
        # 1. Add user message and display immediately
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # 2. Show loading indicator while processing
        with chat_container:
            with st.chat_message("assistant"):
                with st.status("Analyzing request...", expanded=True) as status:
                    # Process with Agentic AI
                    chat_agent = AgenticScoutChat()
                    if chat_agent.available:
                        st.write("Parsing intent...")
                        filters = chat_agent.parse_intent(prompt)
                        
                        if "error" in filters:
                            response_text = f"Error: {filters['error']}"
                            needs_rerun = False
                        else:
                            st.write("Applying filters and navigation...")
                            # Get API params with actions
                            api_params = chat_agent.get_api_params(filters)
                            
                            # Dispatch action (navigate + apply filters)
                            action_result, needs_rerun = dispatch_agentic_action(api_params)
                            
                            # Build response
                            response_text = f"**Action:** {action_result}\n\n"
                            response_text += "**Parsed Intent:**\n"
                            for k, v in filters.items():
                                response_text += f"- `{k}`: {v}\n"
                        status.update(label="Analysis complete!", state="complete", expanded=False)
                    else:
                        response_text = "**Local AI is Offline.** Please start Ollama to use this feature."
                        needs_rerun = False
                        status.update(label="Ollama Offline", state="error")
        
        # 3. Save response to session state
        st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
        
        # 4. CRITICAL: Rerun to apply navigation/filter changes immediately
        if needs_rerun:
            st.rerun()
        else:
            st.rerun() # Refresh to clear the status box and show the persistent markdown message instead

st.divider()

# ============================================================================
# PAGE 1: PLAYER SEARCH
# ============================================================================

if st.session_state.page == 'Player Search':
    st.header("Player Search")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_input = st.text_input(
            "Search player name:",
            placeholder="Type to search (fuzzy matching enabled)...",
            key='player_search'
        )
    
    with col2:
        search_league = st.selectbox(
            "Filter by league:",
            options=['all'] + LEAGUES,
            key='search_league'
        )
    
    if search_input:
        # Get suggestions - Use API if available, fallback to local engine
        suggestions = None
        
        if API_CLIENT_AVAILABLE and st.session_state.backend_available:
            # Try API first (cached via @st.cache_data in api_client)
            api_response = api_search_players(
                query=search_input,
                league=search_league if search_league != 'all' else 'all',
                limit=10
            )
            if api_response.success and api_response.data:
                # Convert API response to local format [(name, score), ...]
                suggestions = [(item['name'], item['score']) for item in api_response.data]
        
        # Fallback to local engine if API failed or unavailable
        if suggestions is None:
            suggestions = engine.get_player_suggestions(
                search_input,
                league=search_league if search_league != 'all' else 'all',
                limit=10
            )
        
        if suggestions:
            # Create selectbox from suggestions
            player_options = [s[0] for s in suggestions]
            selected_player = st.selectbox(
                "Select player:",
                options=player_options,
                key='selected_player'
            )
            
            if selected_player:
                # Get player data using the expanded index lookup
                idx = engine._find_player_index(selected_player)
                if idx is not None and idx in df.index:
                    player_data = df.loc[idx]
                else:
                    st.error(f"**Data Alignment Issue**: Could not find data for '{selected_player}' in the current tactical map.")
                    st.info("This can happen if the cache is out of sync. Please click **'Reset Cache & Reload Data'** in the sidebar.")
                    st.stop()
                
                # Professional Player Card (Bio & Physicals)
                with st.container(border=True):
                    st.subheader(f"Player Profile: {player_data['Player']}")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Age", int(player_data['Age']))
                        st.metric("League", player_data['League'])
                    with col2:
                        st.metric("Position", player_data['Primary_Pos'])
                        st.metric("Squad", player_data['Squad'])
                    with col3:
                        st.metric("90s Played", f"{player_data['90s']:.1f}")
                        archetype = player_data.get('Archetype', 'Unknown')
                        st.metric("Archetype", archetype[:15])
                
                # Performance & Data Confidence Card
                with st.container(border=True):
                    st.subheader("Performance Context")
                
                # Completeness score with professional confidence labels
                completeness = player_data['Completeness_Score']
                player_league = player_data['League']
                
                # Special handling for limited data leagues (capped at 33% by design)
                if player_league in LOW_DATA_LEAGUES:
                    st.write(f"**Data Availability**: Limited Data Tier ({player_league})")
                    st.caption(
                        f"_{player_league} players have limited statistical coverage in our dataset. "
                        "This does not reflect player quality--further manual scouting recommended._"
                    )
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.info(
                            f"**Scouting Note**: {player_league} data is capped at 33% completeness by design. "
                            "Use this profile for directional insights only."
                        )
                    with col2:
                        st.metric("Data Coverage", f"{completeness:.0f}%")
                else:
                    # Standard confidence labels for other leagues
                    # Determine confidence label and color
                    if completeness >= 90:
                        confidence_label = "Verified Elite Data"
                        confidence_desc = "Full scouting confidence - all key metrics available"
                    elif completeness >= 70:
                        confidence_label = "Good Scouting Data"
                        confidence_desc = "Sufficient data for reliable assessment"
                    elif completeness >= 40:
                        confidence_label = "Directional Data"
                        confidence_desc = "Further vetting required - use with caution"
                    else:
                        confidence_label = "Low Data"
                        confidence_desc = "Very limited statistical coverage"
                        
                    color = "green" if completeness >= 70 else "orange" if completeness >= 40 else "red"
                    st.caption(f":{color}[{confidence_label}]: {confidence_desc}")



                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Scouting Confidence**: {confidence_label}")
                        st.caption(f"_{confidence_desc}_ ({completeness:.0f}% complete)")
                    with col2:
                        st.metric("Completeness", f"{completeness:.0f}%")

                

                
                st.divider()
                
                # Scout's Take - ENHANCED WITH LLM
                st.subheader("Scout's Take")
                
                # Toggle for LLM vs rule-based
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write("Automated scouting report")
                with col2:
                    use_llm = st.checkbox(
                        "Use AI",
                        value=False,
                        help="Use Google Gemini for context-aware narratives (requires API key)",
                        key='use_llm_narrative'
                    )
                
                with st.expander("View Automated Scouting Report", expanded=True):
                    try:
                        if use_llm:
                            # Try LLM-powered generation
                            from utils.llm_integration import generate_llm_narrative
                            
                            with st.spinner("Generating AI-powered scouting report..."):
                                narrative = generate_llm_narrative(
                                    player_data,
                                    use_llm=True
                                )
                            
                            st.success("AI-Generated Report (Google Gemini)")
                            st.markdown(narrative)
                        else:
                            # Use rule-based generation
                            narrative = generate_narrative_for_player(player_data)
                            st.info("Rule-Based Report")
                            st.markdown(narrative)
                    except RuntimeError as e:
                        # AI not available or failed
                        st.error(f"AI Generation Failed: {e}")
                        st.warning("**AI is not integrated.** Please check your GEMINI_API_KEY in the .env file.")
                        st.info("Uncheck 'Use AI' to see the rule-based report instead.")
                    except Exception as e:
                        st.error(f"Error generating narrative: {e}")

                
                # PDF Export button - ENHANCED
                st.divider()
                st.subheader("Export Scouting Dossier")
                st.write("Generate a professional PDF report for recruitment meetings")
                
                try:
                    from utils.pdf_export import export_scouting_pdf
                    import tempfile
                    import os
                    
                    # Generate PDF in memory
                    # Generate PDF in memory
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpfile:
                        tmp_filename = tmpfile.name
                    
                    # Close the file handle so FPDF can open it
                    try:
                        export_scouting_pdf(player_data, narrative, tmp_filename)
                        
                        with open(tmp_filename, "rb") as f:
                            pdf_data = f.read()
                    finally:
                        # Clean up
                        if os.path.exists(tmp_filename):
                            os.unlink(tmp_filename)
                    
                    # Direct download button
                    st.download_button(
                        label="Download PDF Scouting Dossier",
                        data=pdf_data,
                        file_name=f"{player_data.get('Player','player').replace(' ', '_')}_scouting_report.pdf",
                        mime="application/pdf",
                        help="Download a one-page scouting dossier for recruitment meetings",
                        use_container_width=True
                    )
                except Exception as e:
                    st.warning(f"PDF export unavailable: {e}")

                
                st.divider()
                
                # Percentile bars - show position-specific stats
                st.subheader("Key Statistics")
                
                is_goalkeeper = player_data['Primary_Pos'] == 'GK'
                
                if is_goalkeeper:
                    # Show goalkeeper-specific stats
                    from utils.constants import GK_FEATURE_COLUMNS
                    percentiles = {}
                    for feat in ['GA90', 'Save%', 'CS%']:
                        pct_col = f'{feat}_pct'
                        if pct_col in player_data.index:
                            percentiles[feat] = player_data[pct_col]
                    
                    if percentiles:
                        pct_df = PlotlyVisualizations.percentile_progress_bars(percentiles)
                        st.dataframe(
                            pct_df, 
                            use_container_width=True, 
                            hide_index=True,
                            column_config={
                                "Metric": st.column_config.TextColumn("Key Metric", width="medium"),
                                "Percentile": st.column_config.ProgressColumn(
                                    "Percentile Rank",
                                    help="Rank against same-position players",
                                    format="%d%%",
                                    min_value=0,
                                    max_value=100,
                                )
                            }
                        )
                    else:
                        st.info("Goalkeeper percentile data not available")
                else:
                    # Show standard outfield stats 
                    # Decision (C): Include advanced stats in percentile bars
                    from utils.constants import FEATURE_COLUMNS
                    percentiles = {}
                    for feat in FEATURE_COLUMNS:
                        pct_col = f'{feat}_pct'
                        if pct_col in player_data.index:
                            percentiles[feat] = player_data[pct_col]
                    
                    pct_df = PlotlyVisualizations.percentile_progress_bars(percentiles)
                    st.dataframe(
                        pct_df, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Metric": st.column_config.TextColumn("Key Metric", width="medium"),
                            "Percentile": st.column_config.ProgressColumn(
                                "Percentile Rank",
                                help="Rank against same-position players",
                                format="%d%%",
                                min_value=0,
                                max_value=100,
                            )
                        }
                    )
                
                
                st.divider()
                
                # NEW: Age-Curve Anomaly Detection (High-Ceiling Prospects)
                st.subheader("Age-Curve Analysis")
                
                try:
                    from utils.age_curve_analysis import AgeCurveAnalyzer, format_age_curve_badge
                    
                    analyzer = AgeCurveAnalyzer(df)
                    
                    # Let user select metric to analyze
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**Select metric to analyze:**")
                    
                    with col2:
                        # Position-specific metric options
                        if is_goalkeeper:
                            available_metrics = ['Save%', 'GA90', 'CS%', 'Saves']
                            default_metric = 'Save%'
                        elif player_data['Primary_Pos'] == 'FW':
                            available_metrics = ['Gls/90', 'Ast/90', 'Sh/90', 'SoT/90', 'Fld/90']
                            default_metric = 'Gls/90'
                        elif player_data['Primary_Pos'] == 'MF':
                            available_metrics = ['Ast/90', 'Gls/90', 'Crs/90', 'TklW/90', 'Int/90']
                            default_metric = 'Ast/90'
                        else:  # DF
                            available_metrics = ['Int/90', 'TklW/90', 'Crs/90', 'Ast/90']
                            default_metric = 'Int/90'
                        
                        # Filter to only metrics that exist in the data
                        available_metrics = [m for m in available_metrics if m in df.columns]
                        
                        if len(available_metrics) > 0:
                            key_metric = st.selectbox(
                                "Metric",
                                options=available_metrics,
                                index=0 if default_metric not in available_metrics else available_metrics.index(default_metric),
                                key='age_curve_metric',
                                label_visibility='collapsed'
                            )
                        else:
                            key_metric = default_metric
                    
                    # Get age-curve status for selected metric
                    age_status = analyzer.get_player_age_curve_status(
                        selected_player,
                        key_metric
                    )
                    
                    if age_status:
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            # Show badge if high-ceiling
                            if age_status.is_high_ceiling:
                                badge = format_age_curve_badge(age_status)
                                st.success(f"**{badge}**")
                                st.caption(
                                    f"This player's {key_metric} ({age_status.player_value:.2f}) is "
                                    f"{age_status.z_score:.1f} standard deviations above the average "
                                    f"for {age_status.age}-year-olds in {age_status.league}."
                                )
                            else:
                                st.info(
                                    f"**Age-Appropriate Performance** - "
                                    f"{key_metric}: {age_status.player_value:.2f} "
                                    f"(Age cohort average: {age_status.age_mean:.2f})"
                                )
                        
                        with col2:
                            st.metric(
                                "Z-Score",
                                f"{age_status.z_score:.2f}",
                                help="Standard deviations above age cohort mean"
                            )
                        
                        with col3:
                            st.metric(
                                "Age Percentile",
                                f"{age_status.percentile_rank:.0f}%",
                                help="Rank within same-age players"
                            )
                        
                        # Show age cohort comparison
                        with st.expander(f"View Age Cohort Comparison ({key_metric})"):
                            age_curves = analyzer.calculate_age_curves(
                                key_metric,
                                position=player_data['Primary_Pos'],
                                league=player_data['League']
                            )
                            
                            if len(age_curves) > 0:
                                from utils.visualizations import PlotlyVisualizations
                                age_curve_fig = PlotlyVisualizations.age_curve(
                                    df[df['Primary_Pos'] == player_data['Primary_Pos']],
                                    key_metric,
                                    position=player_data['Primary_Pos'],
                                    target_player=selected_player,
                                    height=400
                                )
                                st.plotly_chart(age_curve_fig, use_container_width=True)
                                
                                st.write(f"**Age {age_status.age} Statistics ({player_data['Primary_Pos']}, {player_data['League']}):**")
                                st.write(f"- Mean: {age_status.age_mean:.2f}")
                                st.write(f"- Std Dev: {age_status.age_std:.2f}")
                                st.write(f"- Player Value: {age_status.player_value:.2f}")
                    else:
                        st.info("Age-curve analysis unavailable (insufficient age cohort data)")
                
                except Exception as e:
                    st.warning(f"Age-curve analysis unavailable: {e}")

                
                st.divider()
                
                # Similar players
                st.subheader("Top 5 Similar Players")

                
                col1, col2 = st.columns([2, 1])
                with col1:
                    similarity_league = st.selectbox(
                        "Compare in league:",
                        options=['all'] + LEAGUES,
                        key='similarity_league'
                    )
                with col2:
                    use_weights = st.checkbox("Use position weights", value=True, key='use_weights')
                
                similar = None
                player_db_id = None
                
                # Try API first if available (server-side similarity calculation)
                if API_CLIENT_AVAILABLE and st.session_state.backend_available:
                    # We need the player's database ID for API call
                    # Search for this player to get their ID
                    search_response = api_search_players(
                        query=selected_player.split(' (')[0],  # Get name part
                        league='all',
                        limit=1
                    )
                    if search_response.success and search_response.data:
                        player_db_id = search_response.data[0].get('id')
                    
                    if player_db_id:
                        api_response = api_get_similar(
                            player_id=player_db_id,
                            league=similarity_league if similarity_league != 'all' else 'all',
                            top_n=5,
                            use_position_weights=use_weights
                        )
                        if api_response.success and api_response.data:
                            # Convert API response to DataFrame format for UI compatibility
                            matches = api_response.data.get('matches', [])
                            if matches:
                                similar = pd.DataFrame([{
                                    'Player': m['name'],
                                    'Squad': m['squad'],
                                    'League': m['league'],
                                    'Primary_Pos': m['position'],
                                    'Match_Score': m['match_score'],
                                    'Primary_Drivers': m.get('primary_drivers', ''),
                                    'Gls/90': m['stats'].get('Gls/90', 0),
                                    'Ast/90': m['stats'].get('Ast/90', 0),
                                    'Age': m['stats'].get('Age', 0),
                                } for m in matches])
                
                # Fallback to local engine if API failed or unavailable
                if similar is None:
                    similar = engine.find_similar_players(
                        selected_player,
                        league=similarity_league if similarity_league != 'all' else 'all',
                        top_n=5,
                        use_position_weights=use_weights
                    )
                
                if similar is not None:
                    # Show position-appropriate columns
                    if is_goalkeeper:
                        # Show goalkeeper stats for comparison
                        display_cols = ['Player', 'Squad', 'League', 'Match_Score', 'GA90', 'Save%', 'Primary_Pos']
                    else:
                        # Show attacking stats for outfield players
                        display_cols = ['Player', 'Squad', 'League', 'Match_Score', 'Gls/90', 'Ast/90', 'Primary_Pos']
                    
                    # Only include columns that exist in the data
                    display_cols = [col for col in display_cols if col in similar.columns]
                    if 'Primary_Driver' in similar.columns:
                        display_cols.insert(4, 'Primary_Driver')
                    
                    # Show as table for quick scanning
                    st.write("**Similarity Results Table**")
                    st.dataframe(
                        similar[display_cols].head(5),
                        use_container_width=True,
                        hide_index=True
                    )

                    # Detailed cards
                    st.write("**Detailed Match Breakdown**")
                    
                    for idx, (_, row) in enumerate(similar.head(5).iterrows(), 1):
                        match_player = row['Player']
                        match_score = row['Match_Score']
                        
                        # Calculate feature attribution for this match
                        try:
                            attribution = engine.calculate_feature_attribution(
                                selected_player,
                                match_player,
                                use_position_weights=use_weights
                            )
                            
                            if attribution:
                                # Get top 2 primary drivers
                                # Drivers = High similarity (low dist) among the player's top-priority stats
                                # Since attribution is already sorted by target priority, we just pick the ones with lowest distance in the top 5
                                top_priority_stats = list(attribution.items())[:5]
                                best_matches = sorted(top_priority_stats, key=lambda x: x[1])[:2]
                                driver_text = ", ".join([f"{feat}" for feat, _ in best_matches])
                                
                                # Create expander for each match
                                with st.expander(
                                    f"#{idx}: {match_player} - {match_score:.1f}% Match "
                                    f"(Driven by: {driver_text})",
                                    expanded=(idx == 1)
                                ):
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Squad", row['Squad'])
                                    with col2:
                                        st.metric("League", row['League'])
                                    with col3:
                                        st.metric("Position", row['Primary_Pos'])
                                    
                                    # Show key stats
                                    st.write("**Key Stats:**")
                                    stat_cols = st.columns(3)
                                    
                                    if row['Primary_Pos'] == 'GK':
                                        if 'Save%' in row.index:
                                            stat_cols[0].metric("Save%", f"{row['Save%']:.1f}%")
                                        if 'CS%' in row.index:
                                            stat_cols[1].metric("Clean Sheet%", f"{row['CS%']:.1f}%")
                                        if 'GA90' in row.index:
                                            stat_cols[2].metric("GA90", f"{row['GA90']:.2f}")
                                    else:
                                        if 'Gls/90' in row.index:
                                            stat_cols[0].metric("Gls/90", f"{row['Gls/90']:.2f}")
                                        if 'Ast/90' in row.index:
                                            stat_cols[1].metric("Ast/90", f"{row['Ast/90']:.2f}")
                                        if 'Age' in row.index:
                                            age_val = row['Age']
                                            age_display = int(age_val) if not pd.isna(age_val) else "??"
                                            stat_cols[2].metric("Age", age_display)
                                    
                                    # Show similarity breakdown
                                    st.write("**Similarity Breakdown (Target's Key Strengths):**")
                                    for feat, dist in list(attribution.items())[:5]:
                                        # Convert Z-score distance to % similarity
                                        # 0 distance = 100%, 2+ distance = 0%
                                        similarity_pct = int(max(0, (1 - (dist / 2.0))) * 100)
                                        bar_length = int(similarity_pct / 5)
                                        bar = "|" * bar_length + "-" * (20 - bar_length)
                                        st.write(f"{feat}: {bar} {similarity_pct}%")
                            else:
                                # Fallback if attribution fails
                                st.write(f"**#{idx}: {match_player}** - {match_score:.1f}% Match")
                                st.write(f"Squad: {row['Squad']} | League: {row['League']}")
                        except Exception as e:
                            # Fallback display
                            st.write(f"**#{idx}: {match_player}** - {match_score:.1f}% Match")
                            st.write(f"Squad: {row['Squad']} | League: {row['League']}")
                    

                    # NEW: Similarity Driver Analysis
                    st.divider()
                    st.subheader("Explainable Similarity - What Makes Them Similar?")
                    
                    # Show top match
                    if len(similar) > 0:
                        top_match_name = similar.iloc[0]['Player']
                        top_match_score = similar.iloc[0]['Match_Score']
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**Analyzing: {selected_player} vs {top_match_name}**")
                        
                        with col2:
                            show_attribution = st.checkbox(
                                "Show similarity breakdown",
                                value=False,
                                key='show_attribution'
                            )
                        
                        if show_attribution:
                            try:
                                # Calculate feature attribution
                                attribution = engine.calculate_feature_attribution(
                                    selected_player,
                                    top_match_name,
                                    use_position_weights=use_weights
                                )
                                
                                if attribution:
                                    # Create visualization
                                    from utils.visualizations import create_similarity_driver_chart
                                    driver_fig = create_similarity_driver_chart(attribution)
                                    st.plotly_chart(driver_fig, use_container_width=True)
                                    
                                    # Narrative explanation
                                    most_similar_features = list(attribution.items())[:3]  # Top 3
                                    most_different_features = list(attribution.items())[-2:]  # Bottom 2
                                    
                                    st.write("**Most Similar Aspects:**")
                                    for feat, dist in most_similar_features:
                                        similarity_pct = int(max(0, (1 - (dist / 2.0))) * 100)
                                        st.write(f"- {feat}: {similarity_pct}% similar")
                                    
                                    st.write("**Key Differences:**")
                                    for feat, dist in most_different_features:
                                        similarity_pct = int(max(0, (1 - (dist / 2.0))) * 100)
                                        st.write(f"- {feat}: {similarity_pct}% similar (Lower match here)")
                                    
                                    # Summary
                                    if most_different_features:
                                        diff_feat = most_different_features[0][0]
                                        st.write(f"**Overall Match**: {top_match_score:.1f}% - " +
                                               "Strong profile alignment with key differences in " +
                                               f"{diff_feat}")
                                    else:
                                         st.write(f"**Overall Match**: {top_match_score:.1f}% - " +
                                               "Strong profile alignment across all tracked metrics.")
                            except Exception as e:
                                st.warning(f"Could not calculate similarity breakdown: {e}")
                    
                    # PDF Download Button
                    st.divider()
                    st.subheader("Export Scouting Dossier")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write("Download a comprehensive one-page scouting report with radar chart and AI analysis")
                    
                    with col2:
                        if st.button("Download PDF Dossier", key='download_pdf_player_search'):
                            try:
                                from utils.pdf_export import generate_dossier
                                from utils.llm_integration import generate_llm_narrative
                                import tempfile
                                import os
                                
                                # Generate narrative using LLM
                                narrative = generate_llm_narrative(player_data)
                                
                                # Create temp PDF file using a more robust method to avoid WinError 32
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                                    pdf_path = tmp.name
                                
                                try:
                                    # Generate PDF
                                    generate_dossier(player_data, narrative, pdf_path)
                                    
                                    # Read into bytes
                                    with open(pdf_path, "rb") as f:
                                        pdf_bytes = f.read()
                                    
                                    # Store in session state for download button
                                    st.session_state['pdf_bytes'] = pdf_bytes
                                    st.session_state['pdf_filename'] = f"{player_data.get('Player','player').replace(' ', '_')}_scouting_report.pdf"
                                    
                                    st.success("Dossier generated successfully! Click below to download.")
                                    
                                except Exception as e:
                                    st.error(f"Error generating PDF: {e}")
                                finally:
                                    # Clean up temp file
                                    if os.path.exists(pdf_path):
                                        os.unlink(pdf_path)
                                        
                            except Exception as e:
                                st.error(f"Error preparing PDF: {e}")
                        
                        # Show download button if ready (Persistent)
                        if 'pdf_bytes' in st.session_state:
                            st.download_button(
                                label="Download PDF Dossier",
                                data=st.session_state['pdf_bytes'],
                                file_name=st.session_state['pdf_filename'],
                                mime="application/pdf",
                                use_container_width=True
                            )
        else:
            st.info("No players found. Try a different search term.")
    else:
        # Check if filters are active (subset of data)
        # Note: We check if len(df_filtered) < len(df) OR if metrics are in filters
        filters_active = len(df_filtered) < len(df) or (
            'metrics' in st.session_state.filters and st.session_state.filters['metrics']
        )

        if filters_active:
             # SHOW FILTERED RESULTS TABLE
            st.subheader(f"Filtered Candidates ({len(df_filtered)})")
            
            # Determine columns to show
            cols_to_show = ['Player', 'Squad', 'Age', 'Primary_Pos', '90s', 'Gls/90', 'Ast/90', 'xG90', 'xA90', 'Archetype']
            
            # Sort by relevant metric if metric filter exists, else by 90s or value
            sort_col = '90s'
            ascending = False
            
            if 'metrics' in st.session_state.filters and st.session_state.filters['metrics']:
                # Sort by the first metric filter
                first_metric = list(st.session_state.filters['metrics'].keys())[0]
                sort_col = first_metric
            
            display_df = df_filtered.sort_values(sort_col, ascending=False).head(20)
            
            st.dataframe(
                display_df[cols_to_show],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Player": st.column_config.TextColumn("Player", width="medium"),
                    "Gls/90": st.column_config.NumberColumn("Goals/90", format="%.2f"),
                    "Ast/90": st.column_config.NumberColumn("Ast/90", format="%.2f"),
                    "xG90": st.column_config.NumberColumn("xG/90", format="%.2f"),
                }
            )
            st.caption("Showing top 20 matches based on active filters.")
            
        else:
            # Show Trending Prospects when search is empty AND no specific filters
            st.subheader("Trending Prospects (U23, Elite Stats)")
            st.write("Young players with exceptional performance metrics (>80th percentile)")
            
            # Filter for trending prospects
            trending = df[
                (df['Age'] <= 23) &
                ((df['Gls/90_pct'] >= 80) | (df['Ast/90_pct'] >= 80))
            ].sort_values('Gls/90_pct', ascending=False)
            
            if len(trending) > 0:
                display_cols = ['Player', 'Squad', 'League', 'Age', 'Primary_Pos', 'Gls/90', 'xG90', 'Ast/90', 'xA90', 'Archetype']
                display_cols = [col for col in display_cols if col in trending.columns]
                
                st.dataframe(
                    trending[display_cols].head(10),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No trending prospects found with current filters.")

# ============================================================================
# PAGE 2: HEAD-TO-HEAD
# ============================================================================

elif st.session_state.page == 'Head-to-Head':
    st.header("Head-to-Head Comparison")
    
    col1, col2 = st.columns(2)
    
    # We use a search + selectbox pattern to allow accent-insensitive and fuzzy matching
    with col1:
        p1_search = st.text_input("Search first player:", placeholder="e.g. 'Odegaard'", key='p1_search_box')
        
        # Get suggestions from engine (handles normalization/accents)
        p1_suggestions = engine.get_player_suggestions(p1_search or "", limit=20)
        p1_options = [s[0] for s in p1_suggestions]
        
        # If no search or no results, show a few top players or current selection
        if not p1_options:
            if 'player1_select' in st.session_state:
                p1_options = [st.session_state.player1_select]
            else:
                p1_options = ["Search to see players..."]

        player1 = st.selectbox(
            "Select first player:",
            options=p1_options,
            key='player1_select'
        )
    
    with col2:
        p2_search = st.text_input("Search second player:", placeholder="e.g. 'Mbappe'", key='p2_search_box')
        
        # Get suggestions from engine
        p2_suggestions = engine.get_player_suggestions(p2_search or "", limit=20)
        p2_options = [s[0] for s in p2_suggestions]
        
        if not p2_options:
            if 'player2_select' in st.session_state:
                p2_options = [st.session_state.player2_select]
            else:
                p2_options = ["Search to see players..."]

        player2 = st.selectbox(
            "Select second player:",
            options=p2_options,
            key='player2_select'
        )
    
    if player1 and player2:
        # Get comparison
        comparison = engine.compare_players(player1, player2, use_position_weights=True)
        
        if comparison:
            # Player info
            col1, col2, col3, col4 = st.columns(4)
            
            p1 = comparison['player1']
            p2 = comparison['player2']
            
            with col1:
                st.metric(f"{p1['name']}", f"{p1['league']} | {p1['position']}")
            with col2:
                p1_age = p1['age']
                p2_age = p2['age']
                p1_age_display = int(p1_age) if not pd.isna(p1_age) else "??"
                p2_age_display = int(p2_age) if not pd.isna(p2_age) else "??"
                st.metric("Age", f"{p1_age_display} vs {p2_age_display}")
            with col3:
                st.metric(f"{p2['name']}", f"{p2['league']} | {p2['position']}")
            with col4:
                st.metric("Match Score", f"{comparison['match_score']:.1f}%")

            st.divider()

            # Smart Summary
            # Using engine to find correct indices (handles "Name (Squad)" format)
            idx1 = engine._find_player_index(player1)
            idx2 = engine._find_player_index(player2)
            
            if idx1 is not None and idx2 is not None:
                p1_row = df.loc[idx1]
                p2_row = df.loc[idx2]
                
                comp_narrative = generate_comparison_narrative(p1_row, p2_row)
                
                st.subheader("Smart Analysis")
                st.info(comp_narrative)
            
            st.divider()
            
            # Check if both players are goalkeepers
            is_both_gk = p1['position'] == 'GK' and p2['position'] == 'GK'
            
            # Radar chart
            st.subheader("Radar Comparison")
            
            use_pct_radar = st.checkbox("Use Position Percentiles (Relative Quality)", value=True, key='h2h_radar_pct')
            
            profile1 = engine.get_player_profile(player1, use_percentiles=use_pct_radar, is_goalkeeper=is_both_gk)
            profile2 = engine.get_player_profile(player2, use_percentiles=use_pct_radar, is_goalkeeper=is_both_gk)
            
            from utils.similarity import RadarChartGenerator
            generator = RadarChartGenerator()
            radar_fig = generator.generate_plotly_radar(
                profile1,
                profile2,
                player1,
                player2,
                use_percentiles=use_pct_radar,
                is_goalkeeper=is_both_gk
            )
            st.plotly_chart(radar_fig, use_container_width=True)
            
            # Feature comparison table
            st.subheader("Feature-by-Feature Comparison")
            
            # Determine which features to show
            if is_both_gk:
                from utils.constants import GK_FEATURE_COLUMNS
                features_to_show = [feat for feat in GK_FEATURE_COLUMNS if feat in comparison['feature_comparison']]
            else:
                from utils.constants import FEATURE_COLUMNS
                features_to_show = FEATURE_COLUMNS
            
            comp_data = []
            for feat in features_to_show:
                if feat in comparison['feature_comparison']:
                    data = comparison['feature_comparison'][feat]
                    winner = "P1" if data['player1_better'] else "P2"
                    comp_data.append({
                        'Feature': feat,
                        p1['name'][:15]: f"{data['player1']:.2f}",
                        p2['name'][:15]: f"{data['player2']:.2f}",
                        'Diff': f"{data['difference']:+.2f}",
                        'Winner': winner,
                    })
            
            comp_df = pd.DataFrame(comp_data)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            # PDF Download Button
            st.divider()
            st.subheader("Export Comparison Dossier")
            
            if st.button("Download Comparison PDF", key='download_pdf_h2h'):
                try:
                    from utils.pdf_export import generate_dossier, save_radar_chart_image
                    from utils.llm_integration import generate_llm_narrative
                    import tempfile
                    import os
                    
                    with st.spinner("Generating professional comparison report..."):
                        # Get data for both players
                        p1_data = df[df['Player'] == player1].iloc[0]
                        p2_data = df[df['Player'] == player2].iloc[0]
                        
                        # Generate narrative for comparison
                        narrative = generate_llm_narrative(p1_data)
                        narrative += f"\n\n**H2H ANALYSIS**: {player1} vs {player2}"
                        narrative += f"\nMatch Score: {comparison['match_score']:.1f}%"
                        narrative += "\nThis comparison identifies the stylistic overlap and performance gaps between these two profiles."
                        
                        # Create temp PDF file
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            pdf_path = tmp.name
                            
                        try:
                            # Generate radar image (with BOTH players)
                            radar_path = save_radar_chart_image(p1_data, p2_data)
                            
                            # Generate PDF
                            generate_dossier(p1_data, narrative, pdf_path, radar_image_path=radar_path)
                            
                            # Read PDF for download
                            with open(pdf_path, 'rb') as f:
                                pdf_data = f.read()
                            
                            st.download_button(
                                label="Save Comparison PDF",
                                data=pdf_data,
                                file_name=f"comparison_{player1.replace(' ', '_')}_vs_{player2.replace(' ', '_')}.pdf",
                                mime="application/pdf",
                                key='save_comparison_pdf'
                            )
                            
                            st.success("Comparison PDF generated successfully!")
                        finally:
                            # Cleanup
                            if os.path.exists(pdf_path):
                                try:
                                    os.unlink(pdf_path)
                                except:
                                    pass
                            if 'radar_path' in locals() and os.path.exists(radar_path):
                                try:
                                    os.unlink(radar_path)
                                except:
                                    pass
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")



# ============================================================================
# PAGE 3: HIDDEN GEMS
# ============================================================================

elif st.session_state.page == 'Hidden Gems':
    st.header("Hidden Gems Discovery")
    st.write("Discover high-efficiency outliers and unique profiles.")
    
    # Mode Selection
    search_mode = st.radio(
        "Search Mode:",
        ["Discovery (Filters)", "Benchmark (Player Match)"], 
        key='gems_search_mode',
        horizontal=True,
        help="Choose between filtering by metrics or finding players similar to a benchmark player."
    )
    
    st.divider()

    if search_mode == "Discovery (Filters)":
        # -------------------------------------------------------------------------
        # EXISTING DISCOVERY LOGIC
        # -------------------------------------------------------------------------
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            exclude_pl = st.checkbox("Exclude Premier League", value=True, help="Focus on lower leagues/abroad.")
        with col2:
            use_step_up = st.toggle("Step-Up Projection", help="Apply penalty to non-PL stats to estimate PL quality.")
        with col3:
            pass # spacer

        # Base Filtering
        gems = st.session_state.df_clustered.copy()
        if exclude_pl:
            gems = gems[gems['League'] != 'Premier League']
            
        if use_step_up:
            # PROJECT TO PREMIER LEAGUE
            gems = project_to_tier(gems, target_tier='Premier League')
            # Swap projected columns for display/filtering
            cols_to_swap = ['Gls/90', 'Ast/90', 'xG90', 'xA90', 'Sh/90', 'SoT/90']
            for col in cols_to_swap:
                if f'Projected_{col}' in gems.columns:
                    gems[col] = gems[f'Projected_{col}']
            st.info("Stats have been discounted to reflect projected Premier League output.")

        # -------------------------------------------------------------------------
        # 2. FILTERS
        # -------------------------------------------------------------------------
        st.subheader("Discovery Filters")
        
        # Tabs reordered: Basics first, then Efficiency
        tab_basics, tab_efficiency = st.tabs(["Basics", "Efficiency"])
        
        with tab_basics:
            col1, col2, col3 = st.columns(3)
            with col1:
                # Use AI-set age if available, otherwise default to 24
                default_max_age = st.session_state.get('gems_max_age', 24)
                max_age = st.slider("Max Age:", 16, 35, default_max_age)
                # Sync back to session state
                st.session_state.gems_max_age = max_age
            with col2:
                min_90s = st.slider("Min 90s Played:", 0, 30, 5, step=1)
            with col3:
                # Ensure percentile cols exist
                if 'Gls/90_pct' in gems.columns:
                     min_percentile = st.slider("Min Gls/90 Percentile:", 0, 100, 0)
                else:
                     min_percentile = 0

        with tab_efficiency:
            st.caption("Find players who overperform their expected metrics (Clinical Finishing & Creativity).")
            col1, col2 = st.columns(2)
            with col1:
                min_fin_eff = st.slider("Min Finishing Efficiency (Gls - xG):", -0.5, 1.0, 0.0, step=0.05, 
                                        help="Positive = Scoring more than expected.")
            with col2:
                min_creat_eff = st.slider("Min Creative Efficiency (Ast - xA):", -0.5, 1.0, 0.0, step=0.05,
                                          help="Positive = Assisting more than expected.")

        # Apply Filters
        # Ensure columns exist (patch for cached data)
        if 'Finishing_Efficiency' not in gems.columns and 'Gls/90' in gems.columns: 
            gems['Finishing_Efficiency'] = gems['Gls/90'] - gems.get('xG90', 0)
        if 'Creative_Efficiency' not in gems.columns and 'Ast/90' in gems.columns: 
            gems['Creative_Efficiency'] = gems['Ast/90'] - gems.get('xA90', 0)

        filtered_gems = gems[
            (gems.get('Finishing_Efficiency', 0) >= min_fin_eff) &
            (gems.get('Creative_Efficiency', 0) >= min_creat_eff) &
            (gems['Age'] <= max_age) &
            (gems['90s'] >= min_90s)
        ]
        
        if min_percentile > 0 and 'Gls/90_pct' in filtered_gems.columns:
            filtered_gems = filtered_gems[filtered_gems['Gls/90_pct'] >= min_percentile]
        
        # -------------------------------------------------------------------------
        # 3. RESULTS
        # -------------------------------------------------------------------------
        st.divider()
        st.subheader(f"Results ({len(filtered_gems)} players)")
        
        if not filtered_gems.empty:
            # Sort by Finishing Efficiency
            filtered_gems = filtered_gems.sort_values('Finishing_Efficiency', ascending=False)
                
            cols = ['Player', 'Age', 'League', 'Squad', 'Archetype', 'Finishing_Efficiency', 'Creative_Efficiency']
            # Add Gls/90 to view context
            if 'Gls/90' in filtered_gems.columns: cols.append('Gls/90')

            st.dataframe(
                filtered_gems[cols].head(50), 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Player": st.column_config.TextColumn("Player", pinned=True),
                    "Finishing_Efficiency": st.column_config.NumberColumn("Finishing Eff.", format="%+.2f"),
                    "Creative_Efficiency": st.column_config.NumberColumn("Creative Eff.", format="%+.2f"),
                    "Gls/90": st.column_config.ProgressColumn("Goals/90 (Scaled)", min_value=0, max_value=1, format="%.2f"),
                }
            )
            
            # Export
            csv = filtered_gems.to_csv(index=False)
            st.download_button("Download Data", csv, "hidden_gems.csv", "text/csv")
        else:
            st.warning("No players found. Try relaxing the filters.")
    
    elif search_mode == "Benchmark (Player Match)":
        # -------------------------------------------------------------------------
        # BENCHMARK SEARCH LOGIC
        # -------------------------------------------------------------------------
        st.subheader("Find the 'Next'...")
        st.caption("Search for players across lower leagues who statically resemble a top-tier star.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Benchmark Player Selector
            gems_bench_search = st.text_input("Search benchmark player:", placeholder="e.g. 'Odegaard'", key='gems_bench_search')
            
            bench_suggestions = engine.get_player_suggestions(gems_bench_search or "", limit=10)
            bench_options = [s[0] for s in bench_suggestions]
            
            if not bench_options:
                curr_bench = st.session_state.get('benchmark_player')
                if curr_bench:
                    bench_options = [curr_bench]
                else:
                    bench_options = ["Search to see players..."]

            benchmark_player_entry = st.selectbox(
                "Select benchmark:",
                options=bench_options,
                key='benchmark_player'
            )
            
            benchmark_name = benchmark_player_entry.split(" (")[0] if benchmark_player_entry and " (" in benchmark_player_entry else None
        
        with col2:
            curr_league = st.session_state.get('benchmark_target_league', 'Championship')
            league_idx = LEAGUES.index(curr_league) if curr_league in LEAGUES else 0
            
            target_league = st.selectbox(
                "Target League:",
                options=LEAGUES,
                index=league_idx,
                key='benchmark_target_league'
            )
            
        with col3:
            curr_priority = st.session_state.get('benchmark_priority', 'Standard')
            priority_options = list(SCOUTING_PRIORITIES.keys())
            priority_idx = priority_options.index(curr_priority) if curr_priority in priority_options else 0
            
            priority = st.selectbox(
                "Metric Priority:",
                options=priority_options,
                index=priority_idx,
                key='benchmark_priority'
            )
            
        if benchmark_name and target_league:
            st.divider()
            
            # Run similarity search
            st.write(f"Finding **{priority}** profiles in **{target_league}** similar to **{benchmark_name}**...")
            
            try:
                results = None
                
                # Try API first if available
                if API_CLIENT_AVAILABLE and st.session_state.backend_available:
                    # Get benchmark player's database ID
                    search_response = api_search_players(
                        query=benchmark_name,
                        league='all',
                        limit=1
                    )
                    if search_response.success and search_response.data:
                        benchmark_id = search_response.data[0].get('id')
                        if benchmark_id:
                            api_response = api_get_similar(
                                player_id=benchmark_id,
                                league=target_league,
                                top_n=20,
                                use_position_weights=True,
                                scouting_priority=priority,
                                target_league_tier=LEAGUE_TO_TIER.get(target_league)
                            )
                            if api_response.success and api_response.data:
                                matches = api_response.data.get('matches', [])
                                if matches:
                                    results = pd.DataFrame([{
                                        'Player': m['name'],
                                        'Squad': m['squad'],
                                        'League': m['league'],
                                        'Age': m['stats'].get('Age', 0),
                                        'Match_Score': m['match_score'],
                                        'Primary_Drivers': m.get('primary_drivers', ''),
                                        'Proxy_Warnings': api_response.data.get('proxy_warnings', ''),
                                    } for m in matches])
                
                # Fallback to local engine
                if results is None:
                    results = engine.find_similar_players(
                        target_player=benchmark_name, 
                        league=target_league,
                        top_n=20,
                        use_position_weights=True,
                        scouting_priority=priority,
                        target_league_tier=LEAGUE_TO_TIER.get(target_league)
                    )
                
                if results is not None and not results.empty:
                    st.success(f"Found {len(results)} matches!")
                    
                    # Display columns
                    display_cols = ['Player', 'Squad', 'Age', 'Match_Score', 'Primary_Drivers']
                    
                    # Add Proxy Warning column if it exists and has content
                    if 'Proxy_Warnings' in results.columns and results['Proxy_Warnings'].str.len().sum() > 0:
                         display_cols.append('Proxy_Warnings')
                    
                    st.dataframe(
                        results[display_cols],
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Match_Score": st.column_config.ProgressColumn(
                                "Similarity Score",
                                help="How similar the player is to the benchmark",
                                format="%.1f%%",
                                min_value=0,
                                max_value=100,
                            ),
                            "Proxy_Warnings": st.column_config.TextColumn(
                                "Data Confidence",
                                help="When stats are missing in lower leagues, we use these proxies.",
                            )
                        }
                    )
                else:
                    st.warning("No matches found. Try a different league or benchmark player.")
                    
            except Exception as e:
                st.error(f"Error during search: {e}")
                # Optional: print stack trace for debugging
                # st.exception(e)
    


# ============================================================================
# PAGE 4: LEADERBOARDS
# ============================================================================

elif st.session_state.page == 'Leaderboards':
    st.header("League Leaderboards")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        position_filter = st.selectbox(
            "Filter by position:",
            options=['all'] + PRIMARY_POSITIONS,
            key='leaderboard_position'
        )
        # Note: We need to re-run if position changes to update metric list
        if 'prev_pos' not in st.session_state:
            st.session_state.prev_pos = position_filter
        if st.session_state.prev_pos != position_filter:
            st.session_state.prev_pos = position_filter
            st.rerun()

    with col2:
        league = st.selectbox(
            "Select league:",
            options=['all'] + LEAGUES,
            key='leaderboard_league'
        )
        
    with col3:
        from utils.constants import FEATURE_COLUMNS, GK_FEATURE_COLUMNS
        # Dynamically switch options based on position filter
        if position_filter == 'GK':
            metric_options = GK_FEATURE_COLUMNS
        else:
            metric_options = FEATURE_COLUMNS
            
        metric = st.selectbox(
            "Select metric:",
            options=metric_options,
            index=0,
            key='leaderboard_metric'
        )
    
    # Apply filters
    board_df = df_filtered.copy()
    
    if league != 'all':
        board_df = board_df[board_df['League'] == league]
    
    if position_filter != 'all':
        board_df = board_df[board_df['Primary_Pos'] == position_filter]
    
    # Sort by metric
    board_df = board_df.sort_values(metric, ascending=False)
    
    st.divider()
    
    # Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Players", len(board_df))
    with col2:
        st.metric("Avg " + metric, f"{board_df[metric].mean():.2f}")
    with col3:
        st.metric("Max " + metric, f"{board_df[metric].max():.2f}")
    with col4:
        st.metric("Min " + metric, f"{board_df[metric].min():.2f}")
    
    st.divider()
    
    # Leaderboard
    st.subheader(f"Top Players - {metric}")
    
    display_cols = ['Player', 'Squad', 'League', 'Primary_Pos', 'Age', metric, f'{metric}_pct', 'Archetype']
    st.dataframe(
        board_df[display_cols].head(25),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Player": st.column_config.TextColumn("Player", pinned=True),
            metric: st.column_config.NumberColumn(metric, format="%.2f"),
            f'{metric}_pct': st.column_config.ProgressColumn(
                "Percentile",
                help=f"Rank for {metric}",
                min_value=0,
                max_value=100,
                format="%d%%"
            ),
            "Archetype": st.column_config.TextColumn("Archetype", width="small")
        }
    )
    
    
    # Distribution visualization - IMPROVED
    st.divider()
    st.subheader(f"Distribution Analysis - {metric}")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Histogram", "League Comparison", "Top Performers", "Tactical Style Map"])
    
    with tab1:
        # Histogram with percentile markers
        import plotly.graph_objects as go
        
        fig = go.Figure()
        
        # Add histogram
        fig.add_trace(go.Histogram(
            x=board_df[metric],
            nbinsx=30,
            name='Distribution',
            marker=dict(
                color='#3498DB',
                line=dict(color='white', width=1)
            ),
            opacity=0.75,
        ))
        
        # Add mean line
        mean_val = board_df[metric].mean()
        fig.add_vline(
            x=mean_val,
            line_dash='dash',
            line_color='#E74C3C',
            line_width=2,
            annotation_text=f'Mean: {mean_val:.2f}',
            annotation_position='top'
        )
        
        # Add median line
        median_val = board_df[metric].median()
        fig.add_vline(
            x=median_val,
            line_dash='dot',
            line_color='#F39C12',
            line_width=2,
            annotation_text=f'Median: {median_val:.2f}',
            annotation_position='bottom'
        )
        
        fig.update_layout(
            title=f'{metric} Distribution - {league if league != "all" else "All Leagues"}',
            xaxis_title=metric,
            yaxis_title='Player Count',
            height=450,
            template='plotly_dark',
            showlegend=False,
            hovermode='x',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("25th Percentile", f"{board_df[metric].quantile(0.25):.2f}")
        with col2:
            st.metric("50th Percentile", f"{board_df[metric].quantile(0.50):.2f}")
        with col3:
            st.metric("75th Percentile", f"{board_df[metric].quantile(0.75):.2f}")
        with col4:
            st.metric("Std Dev", f"{board_df[metric].std():.2f}")
    
    with tab2:
        # Box plot by league
        if league == 'all' and len(board_df['League'].unique()) > 1:
            league_box = PlotlyVisualizations.league_comparison(
                board_df,
                metric,
                height=450
            )
            st.plotly_chart(league_box, use_container_width=True)
            
            # League stats table
            st.write("**League Statistics:**")
            league_stats = board_df.groupby('League')[metric].agg(['mean', 'median', 'std', 'count']).round(2)
            league_stats.columns = ['Mean', 'Median', 'Std Dev', 'Players']
            league_stats = league_stats.sort_values('Mean', ascending=False)
            st.dataframe(league_stats, use_container_width=True)
        else:
            st.info("Select 'all' leagues to see league comparison")
    
    with tab3:
        # Scatter plot of top performers
        import plotly.express as px
        
        top_50 = board_df.head(50)
        
        fig = px.scatter(
            top_50,
            x='Age',
            y=metric,
            color='League',
            size=f'{metric}_pct',
            hover_data=['Player', 'Squad', 'Primary_Pos'],
            title=f'Top 50 Players - {metric} vs Age',
            color_discrete_map=LEAGUE_COLORS,
            labels={
                'Age': 'Age',
                metric: metric,
                f'{metric}_pct': 'Percentile'
            }
        )
        
        fig.update_layout(
            height=450,
            template='plotly_dark',
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Tactical Style Map (Archetype Universe)
        st.write("**Tactical Galaxy**: Players positioned by stylistic similarity via PCA (Principal Component Analysis)")
        st.caption("Each dot is a player. Proximity = Stylistic Similarity. Color = Assigned Archetype.")
        
        # Model Confidence Metric (Silhouette Score)
        clusterer = st.session_state.clusterer
        if clusterer is not None:
            try:
                silhouette = clusterer.get_silhouette_score()
                
                # Determine confidence label and color
                if silhouette >= 0.50:
                    confidence_label = "Excellent"
                    confidence_color = "success"
                elif silhouette >= 0.35:
                    confidence_label = "Good"
                    confidence_color = "warning"
                else:
                    confidence_label = "Overlap Warning"
                    confidence_color = "error"
                
                # Display as compact metric row
                col_metric, col_status, col_spacer = st.columns([1, 1, 2])
                with col_metric:
                    st.metric(
                        "Model Confidence",
                        f"{silhouette:.3f}",
                        help="Silhouette Score: -1 to 1. Higher = better cluster separation. <0.35 indicates significant overlap."
                    )
                with col_status:
                    st.markdown(f"**Status:** {confidence_label}")
                    if silhouette < 0.35:
                        st.caption("Clusters may overlap. Archetype assignments less reliable.")
                
                st.divider()
            except Exception as e:
                st.warning(f"Model confidence unavailable: {e}")

        
        # Filter by archetype
        selected_universe_archs = st.multiselect(
            "Filter by Archetype:",
            options=ARCHETYPE_NAMES,
            default=ARCHETYPE_NAMES[:4], # Show some by default to avoid clutter
            key='universe_arch_filter'
        )
        
        # Universe visualization
        # Universe visualization
        col_viz_sets, _ = st.columns([1, 2])
        with col_viz_sets:
            show_centroids = st.checkbox("Show Archetype Ideals (Centroids)", value=False, help="Show the 'perfect' version of each archetype.")

        universe_fig = PlotlyVisualizations.archetype_universe_filter(
            st.session_state.df_clustered,
            selected_archetypes=selected_universe_archs,
            height=700
        )
        
        if show_centroids:
             centroid_fig = PlotlyVisualizations.plot_archetype_centroids(st.session_state.clusterer)
             if centroid_fig and hasattr(centroid_fig, 'data'):
                 for trace in centroid_fig.data:
                     universe_fig.add_trace(trace)
        
        st.plotly_chart(universe_fig, use_container_width=True)
        
        st.info("**How to read this**: The axes represent the primary stylistic variances in the dataset. "
                "Forward-thinking players tend to cluster on one side, while defensive stalwarts occupy the other. "
                "Hybrid players appeared in the 'gravity' between defined archetypes.")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Player highlighting option
            highlight_player_name = st.selectbox(
                "Highlight specific player (optional):",
                options=['None'] + sorted(df_filtered['Player'].unique()),
                index=0,
                key='highlight_player_universe'
            )
        
        with col2:
            # Get unique archetypes
            all_archetypes = sorted(df_filtered['Archetype'].unique())
            filter_archetype = st.selectbox(
                "Filter by archetype:",
                options=['All'] + all_archetypes,
                index=0,
                key='filter_archetype_universe'
            )
        
        # Apply archetype filter if selected
        if filter_archetype != 'All':
            universe_df = df_filtered[df_filtered['Archetype'] == filter_archetype]
        else:
            universe_df = df_filtered
        
        # Create the universe map
        highlight = None if highlight_player_name == 'None' else highlight_player_name
        
        universe_fig = PlotlyVisualizations.archetype_universe(
            universe_df,
            height=700,
            highlight_player=highlight
        )
        
        # Interactive chart with selection
        selection = st.plotly_chart(
            universe_fig,
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="universe_plot_selection"
        )
        
        # Handle selection
        if selection and selection['selection']['points']:
            try:
                # Extract player name from customdata (index 0)
                # Streamlit returns points as a list of dicts. We take the first one.
                point_index = selection['selection']['points'][0]['point_index']
                # We need to map point_index back to the dataframe row to get the name reliably
                # Or trust customdata if available in the selection event (Streamlit 1.35+ supports this)
                
                # Robust approach: Get the row from the filtered df used for plotting
                # Note: universe_df might be shuffled or filtered, so we need to be careful.
                # However, Plotly returns point_index relative to the trace.
                # Simplest way if customdata is passed: use it.
                # But st.plotly_chart selection output structure:
                # {'selection': {'points': [{'curve_number': 0, 'point_index': 123, 'point_number': 123, 'x': ..., 'y': ...}]}}
                # It does NOT guaranteed return customdata in the event yet (depends on version).
                
                # Fallback: Get name from universe_df by index
                clicked_player = universe_df.iloc[point_index]['Player']
                
                # Direct update and rerun
                st.session_state.page = 'Player Search'
                st.session_state.player_search = clicked_player
                # Clear previous selection to force new lookup based on search
                if 'selected_player' in st.session_state:
                     del st.session_state['selected_player']
                st.rerun()
                    
            except Exception as e:
                # Fail silently or log if index mismatch (e.g. during filtering updates)
                pass
        
        # Explanation
        with st.expander("How to read the Tactical Style Map"):
            st.markdown("""
            **What is this?**
            - Each dot represents a player, positioned based on their playing style (PCA analysis)
            - Players close together have similar statistical profiles
            - Colors represent the assigned archetype
            
            **How to use it:**
            - Hover over points to see player details
            - Use the highlight feature to find a specific player
            - Filter by archetype to focus on specific playing styles
            - Look for players between clusters to find tactical hybrids
            
            **Scouting insights:**
            - Players on the edge of their archetype cluster may be versatile
            - Isolated players have unique statistical profiles
            - Dense clusters indicate common playing styles in the dataset
            """)

    
    # Archetype Distribution Statistics
    st.divider()
    st.subheader("Archetype Distribution")
    
    archetype_counts = df_filtered['Archetype'].value_counts().reset_index()
    archetype_counts.columns = ['Archetype', 'Count']
    archetype_counts['Percentage'] = (archetype_counts['Count'] / len(df_filtered) * 100).round(1)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create pie chart
        archetype_pie = PlotlyVisualizations.archetype_distribution(
            df_filtered,
            league=league if league != 'all' else 'all',
            height=400
        )
        st.plotly_chart(archetype_pie, use_container_width=True)
    
    with col2:
        st.dataframe(
            archetype_counts,
            use_container_width=True,
            hide_index=True
        )

# ============================================================================
# PAGE 5: SQUAD ANALYSIS
# ============================================================================

elif st.session_state.page == 'Squad Analysis':
    st.header("Squad Analysis")
    st.caption("Team-level tactical analysis and composition insights")
    
    # Use global df
    squad_df = df  # df is already st.session_state.df_clustered
    
    # ============================================================================
    # SQUAD SELECTOR
    # ============================================================================

    st.subheader("Select Squad")

    # Get all unique squads with their leagues
    squad_options = squad_df.groupby(['Squad', 'League']).size().reset_index(name='Players')
    squad_options['Display'] = squad_options['Squad'] + " (" + squad_options['League'] + ")"
    squad_options = squad_options.sort_values('Display')

    # Search box
    col1, col2 = st.columns([3, 1])

    with col1:
        search_term = st.text_input(
            "Search squad:",
            placeholder="Type to search (e.g., 'Manchester', 'Arsenal')...",
            key='squad_search'
        )

    with col2:
        league_filter = st.selectbox(
            "Filter by league:",
            options=['All Leagues'] + LEAGUES,
            key='squad_league_filter'
        )

    # Filter squad options
    filtered_squads = squad_options.copy()
    if search_term:
        term_norm = normalize_name(search_term)
        filtered_squads = filtered_squads[
            filtered_squads['Squad'].apply(normalize_name).str.contains(term_norm, regex=False)
        ]
    if league_filter != 'All Leagues':
        filtered_squads = filtered_squads[filtered_squads['League'] == league_filter]

    if len(filtered_squads) > 0:
        selected_display = st.selectbox(
            "Select squad:",
            options=filtered_squads['Display'].tolist(),
            key='selected_squad'
        )
        
        # Extract squad name from display
        selected_squad = selected_display.rsplit(' (', 1)[0]
        selected_league = filtered_squads[filtered_squads['Display'] == selected_display]['League'].iloc[0]
        
        # Get squad roster
        squad_roster = get_squad_roster(squad_df, selected_squad)
        league_roster = squad_df[squad_df['League'] == selected_league]
        
        st.divider()
        
        # ========================================================================
        # SQUAD OVERVIEW
        # ========================================================================
        
        st.subheader("Squad Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Players", len(squad_roster))
        with col2:
            st.metric("Avg Age", f"{squad_roster['Age'].mean():.1f}")
        with col3:
            st.metric("Total 90s", f"{squad_roster['90s'].sum():.0f}")
        with col4:
            st.metric("Avg Gls/90", f"{squad_roster['Gls/90'].mean():.2f}")
        with col5:
            # Check if dominance column exists
            if 'Gls/90_Dominance' in squad_roster.columns:
                avg_dom = squad_roster['Gls/90_Dominance'].mean()
                st.metric("Gls Dominance", f"{avg_dom:+.2f}")
            else:
                st.metric("Avg Ast/90", f"{squad_roster['Ast/90'].mean():.2f}")
        
        st.divider()
        
        # ========================================================================
        # TACTICAL DNA (ARCHETYPE DISTRIBUTION)
        # ========================================================================
        
        st.subheader("Tactical DNA")
        st.write("Distribution of player archetypes within the squad")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            archetype_dist = get_archetype_distribution(squad_roster)
            if archetype_dist:
                pie_fig = create_archetype_pie_chart(archetype_dist, selected_squad)
                if pie_fig:
                    st.plotly_chart(pie_fig, use_container_width=True)
            else:
                st.info("Archetype data not available for this squad")
        
        with col2:
            st.write("**Archetype Breakdown:**")
            if archetype_dist:
                for archetype, count in sorted(archetype_dist.items(), key=lambda x: -x[1]):
                    pct = (count / len(squad_roster)) * 100
                    st.write(f"- **{archetype}**: {count} ({pct:.0f}%)")
            
            # Tactical balance analysis
            st.write("---")
            st.write("**Tactical Balance:**")
            
            # Count by position group
            pos_counts = squad_roster['Primary_Pos'].value_counts()
            for pos in ['FW', 'MF', 'DF', 'GK']:
                count = pos_counts.get(pos, 0)
                st.write(f"- {pos}: {count} players")
        
        st.divider()
        
        # ========================================================================
        # ROSTER AGE CURVE
        # ========================================================================
        
        st.subheader("Roster Age Curve")
        st.write("Compare squad age distribution to league average")
        
        age_fig = create_age_histogram(squad_roster, league_roster, selected_squad)
        st.plotly_chart(age_fig, use_container_width=True)
        
        # Age analysis
        col1, col2, col3 = st.columns(3)
        
        squad_mean_age = squad_roster['Age'].mean()
        league_mean_age = league_roster['Age'].mean()
        age_diff = squad_mean_age - league_mean_age
        
        with col1:
            st.metric(
                "Squad Avg Age",
                f"{squad_mean_age:.1f}",
                delta=f"{age_diff:+.1f} vs league"
            )
        with col2:
            young_players = len(squad_roster[squad_roster['Age'] <= 23])
            st.metric("U-23 Players", young_players)
        with col3:
            peak_players = len(squad_roster[(squad_roster['Age'] >= 24) & (squad_roster['Age'] <= 29)])
            st.metric("Peak Age (24-29)", peak_players)
        
        # Rebuild indicator
        if age_diff > 2:
            st.warning("This squad is OLDER than league average. Consider youth investment.")
        elif age_diff < -2:
            st.success("This squad is YOUNGER than league average. Strong development pipeline.")
        
        st.divider()
        
        # ========================================================================
        # SQUAD RADAR
        # ========================================================================
        
        st.subheader("Squad Profile (Top 11)")
        st.write("Mean percentile scores of starting XI vs league average")
        
        squad_pcts = calculate_squad_mean_percentiles(squad_roster)
        league_pcts = calculate_league_mean_percentiles(squad_df, selected_league)
        
        if squad_pcts and league_pcts:
            radar_fig = create_squad_radar(squad_pcts, league_pcts, selected_squad)
            st.plotly_chart(radar_fig, use_container_width=True)
        else:
            st.info("Insufficient percentile data for radar chart")
        
        st.divider()
        
        # ========================================================================
        # LEAGUE DOMINANCE
        # ========================================================================
        
        st.subheader("League Dominance Scores")
        st.write("How much the squad out-performs their league peers (Z-scores)")
        
        dominance_summary = calculate_squad_dominance_summary(squad_roster)
        
        if dominance_summary:
            dom_fig = create_dominance_bar_chart(dominance_summary, selected_squad)
            if dom_fig:
                st.plotly_chart(dom_fig, use_container_width=True)
            
            # Interpretation
            top_dominance = max(dominance_summary.items(), key=lambda x: x[1])
            weak_dominance = min(dominance_summary.items(), key=lambda x: x[1])
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Strongest Area:** {top_dominance[0]} ({top_dominance[1]:+.2f} Z)")
            with col2:
                st.error(f"**Weakest Area:** {weak_dominance[0]} ({weak_dominance[1]:+.2f} Z)")
        else:
            st.info("Dominance scores not available. Run ETL to calculate.")
        
        st.divider()
        
        # ========================================================================
        # FULL ROSTER TABLE
        # ========================================================================
        
        with st.expander("View Full Roster", expanded=False):
            display_cols = ['Player', 'Primary_Pos', 'Age', '90s', 'Gls/90', 'Ast/90']
            if 'Archetype' in squad_roster.columns:
                display_cols.append('Archetype')
            
            # Add dominance if available
            dominance_cols = [col for col in squad_roster.columns if '_Dominance' in col]
            if dominance_cols:
                display_cols.extend(dominance_cols[:3])  # Top 3 dominance cols
            
            st.dataframe(
                squad_roster[display_cols].sort_values('90s', ascending=False),
                use_container_width=True,
                hide_index=True
            )

    else:
        st.warning("No squads found matching your search criteria.")

# ============================================================================
# PAGE 6: SQUAD PLANNER
# ============================================================================

elif st.session_state.page == 'Squad Planner':
    st.header("Shadow Board Planner")
    st.caption("Plan your squad and identify recruitment targets.")
    
    # Initialize Shadow Squad if not exists
    if 'shadow_squad' not in st.session_state:
        st.session_state.shadow_squad = {
            'GK': None,
            'RB': None, 'RCB': None, 'LCB': None, 'LB': None,
            'CDM': None, 'RCM': None, 'LCM': None,
            'RW': None, 'ST': None, 'LW': None
        }

    # Layout: Left = Settings/Add Player, Right = Pitch
    col_control, col_pitch = st.columns([1, 2])

    with col_control:
        st.subheader("Manage Roster")
        
        # Position Selector to Add/Remove
        selected_pos = st.selectbox("Select Position Slot:", list(PITCH_COORDS.keys()))
        
        current_player = st.session_state.shadow_squad[selected_pos]
        
        if current_player:
            st.info(f"Current: **{current_player['Player']}**")
            if st.button("Remove Player", key='remove_btn'):
                st.session_state.shadow_squad[selected_pos] = None
                st.rerun()
        else:
            st.write("Slot Empty")
            
        st.divider()
        
        st.subheader("Add Player")
        # Player Search
        search_term = st.text_input("Search Player:", key='planner_search')
        
        if search_term:
            # Use global df and name normalization
            term_norm = normalize_name(search_term)
            results = df[df['Player'].apply(normalize_name).str.contains(term_norm, regex=False)].head(10)
            
            if not results.empty:
                player_name_list = results['Player'].tolist()
                # Try to add squad context to keys to make them unique
                player_options = {f"{row['Player']} ({row['Squad']})": row for _, row in results.iterrows()}
                
                selected_option = st.selectbox("Select Result:", list(player_options.keys()))
                
                if st.button("Add to Squad"):
                    # Architectural Requirement: Store Full Player JSON
                    # Converting Series to Dict for serialization
                    player_obj = player_options[selected_option].to_dict()
                    st.session_state.shadow_squad[selected_pos] = player_obj
                    st.success(f"Added {player_obj['Player']} to {selected_pos}")
                    st.rerun()
        
        st.divider()
        
        # Recommendation Engine (Mini)
        if not current_player:
            st.subheader("Recommendations")
            st.caption(f"Find players for {selected_pos}")
            
            target_pos_map = {
                'GK': 'GK', 'RB': 'DF', 'LB': 'DF', 'RCB': 'DF', 'LCB': 'DF',
                'CDM': 'MF', 'RCM': 'MF', 'LCM': 'MF',
                'RW': 'FW', 'LW': 'FW', 'ST': 'FW'
            }
            
            if st.button("Find Candidates"):
                # Simple logic: Top rated players in league for that position
                # Ideally use SimilarityEngine if we had a "Template Player"
                pos_code = target_pos_map.get(selected_pos, 'MF')
                
                # Check metrics existence
                sort_metric = 'Gls/90_Dominance' if pos_code=='FW' and 'Gls/90_Dominance' in df.columns else 'Ast/90'
                if pos_code != 'FW' and 'Int/90_Dominance' in df.columns:
                     sort_metric = 'Int/90_Dominance'
                
                candidates = df[
                    (df['Primary_Pos'] == pos_code) & 
                    (df['Age'] <= 28)
                ].sort_values(sort_metric, ascending=False).head(5)
                
                st.write("Top Available:")
                for _, p in candidates.iterrows():
                    if st.button(f"Add {p['Player']} ({p['Squad']})", key=f"rec_{p['Player']}"):
                         st.session_state.shadow_squad[selected_pos] = p.to_dict()
                         st.rerun()

    with col_pitch:
        st.subheader("Formation: 4-3-3")
        pitch_fig = create_pitch_visualization(st.session_state.shadow_squad)
        st.plotly_chart(pitch_fig, use_container_width=True)
        
        # Squad Metrics
        st.subheader("Shadow Squad DNA")
        
        # Calculate aggregate stats
        active_players = [p for p in st.session_state.shadow_squad.values() if p]
        
        if active_players:
            col1, col2, col3 = st.columns(3)
            ages = [p['Age'] for p in active_players]
            avg_age = sum(ages) / len(ages)
            
            with col1:
                 st.metric("Avg Age", f"{avg_age:.1f}")
            with col2:
                 st.metric("Roster Size", len(active_players))
            with col3:
                 # Example Dominance aggregations
                 doms = [p.get('Gls/90_Dominance', 0) for p in active_players if 'Gls/90_Dominance' in p]
                 avg_dom = sum(doms)/len(doms) if doms else 0
                 st.metric("Avg Attack Dominance", f"{avg_dom:+.2f}")
        else:
            st.info("Add players to see squad metrics.")
            
        # Clear Squad Button (from original sidebar)
        if st.button("Clear Squad", key='clear_shadow_squad'):
            for k in st.session_state.shadow_squad:
                 st.session_state.shadow_squad[k] = None
            st.rerun()

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.caption("Football Scouting Dashboard | Global League Coverage 2024-2025 | Built with Streamlit")
