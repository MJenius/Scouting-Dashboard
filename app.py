"""
app.py - Main Streamlit application entry point for Football Scouting Dashboard.

Multi-page app with global filters (Age, League, Position, Minutes) persisted via st.session_state.
Pages:
  - 🔍 Player Search: Fuzzy search + similar players
  - ⚔️  Head-to-Head: Player comparison + radar charts
  - 💎 Hidden Gems: Metric filters + transfer recommendations
  - 🏆 Leaderboards: Rankings by metric + archetype filters
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

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
    page_title="⚽ Football Scouting Dashboard",
    page_icon="⚽",
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
        'min_90s': 0.5,
    }

if 'page' not in st.session_state:
    st.session_state.page = '🔍 Player Search'

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
    result = process_all_data('english_football_pyramid_master.csv', min_90s=0.5)
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
        with st.spinner("🔄 Loading and processing data..."):
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

    st.title("⚙️ Filters & Settings")
    
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
    
    # Minutes filter
    st.subheader("Minimum Minutes Played")
    min_90s = st.slider(
        "Min 90s:",
        min_value=0.0,
        max_value=30.0,
        value=float(st.session_state.filters['min_90s']),
        step=0.5,
        key='min_90s_slider'
    )
    st.session_state.filters['min_90s'] = min_90s
    
    st.divider()
    
    # Info panel
    st.subheader("📊 Dataset Info")
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
        
        st.metric("Players (Filtered)", len(filtered))
        st.metric("Leagues", len(selected_leagues))
        st.metric("Positions", len(selected_positions))
        
        if len(filtered) > 0:
            st.metric("Avg Age", f"{filtered['Age'].mean():.1f}")
            st.metric("Avg Goals/90", f"{filtered['Gls/90'].mean():.2f}")

    st.divider()
    
    # Backend connection status indicator
    st.subheader("🔌 API Status")
    if API_CLIENT_AVAILABLE and st.session_state.backend_available:
        st.success("✅ FastAPI Backend Connected")
        try:
            health_ok, health_data = check_backend_health()
            if health_ok:
                st.caption(f"👥 {health_data.get('player_count', '?')} players in DB")
                if health_data.get('engine_loaded'):
                    st.caption("🚀 Similarity Engine: Active")
        except:
            pass
    elif API_CLIENT_AVAILABLE:
        st.warning("⚠️ Backend Unavailable")
        st.caption("Using local fallback mode")
        if st.button("🔄 Retry Connection", key='retry_backend'):
            try:
                st.session_state.backend_available = is_backend_available()
                st.rerun()
            except:
                pass
    else:
        st.info("📦 Running in Local Mode")
        st.caption("API client not installed")
    
    st.divider()
    if st.button("🔄 Reset Cache & Reload Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.data_loaded = False
        st.rerun()

# ============================================================================
# MAIN APP
# ============================================================================

def apply_filters(df):
    """Apply global filters to dataframe."""
    filters = st.session_state.filters
    return df[
        (df['Age'] >= filters['age_min']) &
        (df['Age'] <= filters['age_max']) &
        (df['League'].isin(filters['leagues'])) &
        (df['Primary_Pos'].isin(filters['positions'])) &
        (df['90s'] >= filters['min_90s'])
    ]

# Ensure data is loaded
ensure_data_loaded()

# Get data
df = st.session_state.df_clustered
engine = st.session_state.engine

# Apply filters
df_filtered = apply_filters(df)

# Header
st.title("⚽ Football Scouting Dashboard")
st.caption(f"Multi-League Global Scouting Dashboard | {len(df_filtered):,} players after filters")

# Page selection
col1, col2, col3, col4 = st.columns(4)
pages = {
    '🔍 Player Search': col1,
    '⚔️ Head-to-Head': col2,
    '💎 Hidden Gems': col3,
    '🏆 Leaderboards': col4,
}

for page_name, col in pages.items():
    if col.button(page_name, key=f"page_{page_name}", use_container_width=True):
        st.session_state.page = page_name

st.divider()

# ============================================================================
# PAGE 1: PLAYER SEARCH
# ============================================================================

if st.session_state.page == '🔍 Player Search':
    st.header("🔍 Player Search")
    
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
                    st.error(f"⚠️ **Data Alignment Issue**: Could not find data for '{selected_player}' in the current tactical map.")
                    st.info("This can happen if the cache is out of sync. Please click **'Reset Cache & Reload Data'** in the sidebar.")
                    st.stop()
                
                # Player card
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Age", int(player_data['Age']))
                with col2:
                    st.metric("League", player_data['League'])
                with col3:
                    st.metric("Position", player_data['Primary_Pos'])
                with col4:
                    st.metric("90s", f"{player_data['90s']:.1f}")
                with col5:
                    archetype = player_data.get('Archetype', 'Unknown')
                    st.metric("Archetype", archetype[:15])
                
                st.divider()
                
                # Completeness score with professional confidence labels
                completeness = player_data['Completeness_Score']
                player_league = player_data['League']
                
                # Special handling for limited data leagues (capped at 33% by design)
                if player_league in LOW_DATA_LEAGUES:
                    st.write(f"**📋 Data Availability**: 🔵 Limited Data Tier ({player_league})")
                    st.caption(
                        f"_{player_league} players have limited statistical coverage in our dataset. "
                        "This does not reflect player quality—further manual scouting recommended._"
                    )
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.info(
                            f"⚠️ **Scouting Note**: {player_league} data is capped at 33% completeness by design. "
                            "Use this profile for directional insights only."
                        )
                    with col2:
                        st.metric("Data Coverage", f"{completeness:.0f}%")
                else:
                    # Standard confidence labels for other leagues
                    # Determine confidence label and color
                    if completeness >= 90:
                        confidence_label = "🟢 Verified Elite Data"
                        confidence_desc = "Full scouting confidence - all key metrics available"
                    elif completeness >= 70:
                        confidence_label = "🟡 Good Scouting Data"
                        confidence_desc = "Sufficient data for reliable assessment"
                    elif completeness >= 40:
                        confidence_label = "🟠 Directional Data"
                        confidence_desc = "Further vetting required - use with caution"
                    else:
                        confidence_label = "🔴 Incomplete Data"
                        confidence_desc = "Caution advised - limited metrics available"

                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Scouting Confidence**: {confidence_label}")
                        st.caption(f"_{confidence_desc}_ ({completeness:.0f}% complete)")
                    with col2:
                        st.metric("Completeness", f"{completeness:.0f}%")

                

                
                st.divider()
                
                # Scout's Take - ENHANCED WITH LLM
                st.subheader("📝 Scout's Take")
                
                # Toggle for LLM vs rule-based
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write("Automated scouting report")
                with col2:
                    use_llm = st.checkbox(
                        "🤖 Use AI",
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
                            
                            st.success("🤖 AI-Generated Report (Google Gemini)")
                            st.markdown(narrative)
                        else:
                            # Use rule-based generation
                            narrative = generate_narrative_for_player(player_data)
                            st.info("📋 Rule-Based Report")
                            st.markdown(narrative)
                    except RuntimeError as e:
                        # AI not available or failed
                        st.error(f"❌ AI Generation Failed: {e}")
                        st.warning("💡 **AI is not integrated.** Please check your GEMINI_API_KEY in the .env file.")
                        st.info("Uncheck '🤖 Use AI' to see the rule-based report instead.")
                    except Exception as e:
                        st.error(f"❌ Error generating narrative: {e}")

                
                # PDF Export button - ENHANCED
                st.divider()
                st.subheader("📄 Export Scouting Dossier")
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
                        label="📥 Download PDF Scouting Dossier",
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
                st.subheader("📊 Key Statistics")
                
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
                        st.dataframe(pct_df, use_container_width=True, hide_index=True)
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
                    st.dataframe(pct_df, use_container_width=True, hide_index=True)
                
                
                st.divider()
                
                # NEW: Age-Curve Anomaly Detection (High-Ceiling Prospects)
                st.subheader("🌟 Age-Curve Analysis")
                
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
                        with st.expander(f"📊 View Age Cohort Comparison ({key_metric})"):
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
                st.subheader("👥 Top 5 Similar Players")

                
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
                                        bar = "█" * bar_length + "░" * (20 - bar_length)
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
                    st.subheader("🔍 Explainable Similarity - What Makes Them Similar?")
                    
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
                                        st.write(f"• {feat}: {similarity_pct}% similar")
                                    
                                    st.write("**Key Differences:**")
                                    for feat, dist in most_different_features:
                                        similarity_pct = int(max(0, (1 - (dist / 2.0))) * 100)
                                        st.write(f"• {feat}: {similarity_pct}% similar (Lower match here)")
                                    
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
                    st.subheader("📥 Export Scouting Dossier")
                    
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write("Download a comprehensive one-page scouting report with radar chart and AI analysis")
                    
                    with col2:
                        if st.button("📥 Download PDF Dossier", key='download_pdf_player_search'):
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
                                    
                                    st.success("✅ Dossier generated successfully! Click below to download.")
                                    
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
                                label="📥 Download PDF Dossier",
                                data=st.session_state['pdf_bytes'],
                                file_name=st.session_state['pdf_filename'],
                                mime="application/pdf",
                                use_container_width=True
                            )
        else:
            st.info("No players found. Try a different search term.")
    else:
        # Show Trending Prospects when search is empty
        st.subheader("🌟 Trending Prospects (U23, Elite Stats)")
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

elif st.session_state.page == '⚔️ Head-to-Head':
    st.header("⚔️ Head-to-Head Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create unique display names for the selectbox
        player_options = [f"{row['Player']} ({row['Squad']})" for _, row in df.iterrows()]
        player_options = sorted(list(set(player_options)))  # Unique and sorted

        player1 = st.selectbox(
            "Select first player:",
            options=player_options,
            key='player1_select'
        )
    
    with col2:
        player2 = st.selectbox(
            "Select second player:",
            options=player_options,
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
                st.metric(f"📍 {p1['name']}", f"{p1['league']} | {p1['position']}")
            with col2:
                p1_age = p1['age']
                p2_age = p2['age']
                p1_age_display = int(p1_age) if not pd.isna(p1_age) else "??"
                p2_age_display = int(p2_age) if not pd.isna(p2_age) else "??"
                st.metric("Age", f"{p1_age_display} vs {p2_age_display}")
            with col3:
                st.metric(f"⚡ {p2['name']}", f"{p2['league']} | {p2['position']}")
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
                
                st.subheader("📝 Smart Analysis")
                st.info(comp_narrative)
            
            st.divider()
            
            # Check if both players are goalkeepers
            is_both_gk = p1['position'] == 'GK' and p2['position'] == 'GK'
            
            # Radar chart
            st.subheader("📊 Radar Comparison")
            
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
            st.subheader("🔄 Feature-by-Feature Comparison")
            
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
                    winner = "✅ P1" if data['player1_better'] else "✅ P2"
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
            st.subheader("📥 Export Comparison Dossier")
            
            if st.button("📥 Download Comparison PDF", key='download_pdf_h2h'):
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
                                label="💾 Save Comparison PDF",
                                data=pdf_data,
                                file_name=f"comparison_{player1.replace(' ', '_')}_vs_{player2.replace(' ', '_')}.pdf",
                                mime="application/pdf",
                                key='save_comparison_pdf'
                            )
                            
                            st.success("✓ Comparison PDF generated successfully!")
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

elif st.session_state.page == '💎 Hidden Gems':
    st.header("💎 Hidden Gems Discovery")
    st.write("Discover high-efficiency outliers and unique profiles.")
    
    # Mode Selection
    search_mode = st.radio(
        "Search Mode:",
        ["🚀 Discovery (Filters)", "🎯 Benchmark (Player Match)"], 
        horizontal=True,
        help="Choose between filtering by metrics or finding players similar to a benchmark player."
    )
    
    st.divider()

    if search_mode == "🚀 Discovery (Filters)":
        # -------------------------------------------------------------------------
        # EXISTING DISCOVERY LOGIC
        # -------------------------------------------------------------------------
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            exclude_pl = st.checkbox("Exclude Premier League", value=True, help="Focus on lower leagues/abroad.")
        with col2:
            use_step_up = st.toggle("🚀 Step-Up Projection", help="Apply penalty to non-PL stats to estimate PL quality.")
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
            st.info("📉 Stats have been discounted to reflect projected Premier League output.")

        # -------------------------------------------------------------------------
        # 2. FILTERS
        # -------------------------------------------------------------------------
        st.subheader("🔍 Discovery Filters")
        
        tab_efficiency, tab_style, tab_basics = st.tabs(["🚀 Efficiency", "🦄 The Unicorn Finder", "📊 Basics"])
        
        with tab_efficiency:
            st.caption("Find players who overperform their expected metrics (Clinical Finishing & Creativity).")
            col1, col2 = st.columns(2)
            with col1:
                min_fin_eff = st.slider("Min Finishing Efficiency (Gls - xG):", -0.5, 1.0, 0.0, step=0.05, 
                                        help="Positive = Scoring more than expected.")
            with col2:
                min_creat_eff = st.slider("Min Creative Efficiency (Ast - xA):", -0.5, 1.0, 0.0, step=0.05,
                                          help="Positive = Assisting more than expected.")

        with tab_style:
            st.caption("Find players with unique, hybrid profiles that defy standard categorization.")
            unicorn_score = st.slider("🦄 Stylistic Uniqueness (Unicorn Score):", 0, 100, 0,
                                      help="0 = Generic Archetype, 100 = Unique Stylistic Outlier.")
            
        with tab_basics:
            col1, col2, col3 = st.columns(3)
            with col1:
                max_age = st.slider("Max Age:", 16, 35, 24)
            with col2:
                min_90s = st.slider("Min 90s Played:", 0.0, 30.0, 5.0)
            with col3:
                # Ensure percentile cols exist
                if 'Gls/90_pct' in gems.columns:
                     min_percentile = st.slider("Min Gls/90 Percentile:", 0, 100, 0)
                else:
                     min_percentile = 0

        # Apply Filters
        # Ensure columns exist (patch for cached data)
        if 'Finishing_Efficiency' not in gems.columns and 'Gls/90' in gems.columns: 
            gems['Finishing_Efficiency'] = gems['Gls/90'] - gems.get('xG90', 0)
        if 'Creative_Efficiency' not in gems.columns and 'Ast/90' in gems.columns: 
            gems['Creative_Efficiency'] = gems['Ast/90'] - gems.get('xA90', 0)
        if 'Outlier_Score' not in gems.columns: 
            gems['Outlier_Score'] = 0.0

        filtered_gems = gems[
            (gems.get('Finishing_Efficiency', 0) >= min_fin_eff) &
            (gems.get('Creative_Efficiency', 0) >= min_creat_eff) &
            (gems.get('Outlier_Score', 0) >= unicorn_score) &
            (gems['Age'] <= max_age) &
            (gems['90s'] >= min_90s)
        ]
        
        if min_percentile > 0 and 'Gls/90_pct' in filtered_gems.columns:
            filtered_gems = filtered_gems[filtered_gems['Gls/90_pct'] >= min_percentile]
        
        # -------------------------------------------------------------------------
        # 3. RESULTS
        # -------------------------------------------------------------------------
        st.divider()
        st.subheader(f"🎯 Results ({len(filtered_gems)} players)")
        
        if not filtered_gems.empty:
            # Sort
            if unicorn_score > 0:
                filtered_gems = filtered_gems.sort_values('Outlier_Score', ascending=False)
            else:
                filtered_gems = filtered_gems.sort_values('Finishing_Efficiency', ascending=False)
                
            cols = ['Player', 'Age', 'League', 'Squad', 'Archetype', 'Finishing_Efficiency', 'Creative_Efficiency', 'Outlier_Score']
            # Add Gls/90 to view context
            if 'Gls/90' in filtered_gems.columns: cols.append('Gls/90')

            st.dataframe(
                filtered_gems[cols].head(50), 
                use_container_width=True, 
                hide_index=True
            )
            
            # Export
            csv = filtered_gems.to_csv(index=False)
            st.download_button("📥 Download Data", csv, "hidden_gems.csv", "text/csv")
        else:
            st.warning("No players found. Try relaxing the filters.")
    
    elif search_mode == "🎯 Benchmark (Player Match)":
        # -------------------------------------------------------------------------
        # BENCHMARK SEARCH LOGIC
        # -------------------------------------------------------------------------
        st.subheader("🎯 Find the 'Next'...")
        st.caption("Search for players across lower leagues who statically resemble a top-tier star.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Benchmark Player Selector
            # Use 'Player (Squad)' format for uniqueness
            player_options = [f"{row['Player']} ({row['Squad']})" for _, row in df.iterrows()]
            player_options = sorted(list(set(player_options))) 
            
            benchmark_player_entry = st.selectbox(
                "Benchmark Player (The Ideal):",
                options=player_options,
                index=None,
                placeholder="e.g. Bukayo Saka",
                key='benchmark_player'
            )
            
            benchmark_name = benchmark_player_entry.split(" (")[0] if benchmark_player_entry else None
        
        with col2:
            target_league = st.selectbox(
                "Target League:",
                options=LEAGUES,
                key='benchmark_target_league'
            )
            
        with col3:
            priority = st.selectbox(
                "Metric Priority:",
                options=list(SCOUTING_PRIORITIES.keys()),
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
                                "⚠️ Data Confidence",
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

elif st.session_state.page == '🏆 Leaderboards':
    st.header("🏆 League Leaderboards")
    
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
    st.subheader(f"🏅 Top Players - {metric}")
    
    display_cols = ['Player', 'Squad', 'League', 'Primary_Pos', 'Age', metric, f'{metric}_pct', 'Archetype']
    st.dataframe(
        board_df[display_cols].head(25),
        use_container_width=True,
        hide_index=True
    )
    
    
    # Distribution visualization - IMPROVED
    st.divider()
    st.subheader(f"📊 Distribution Analysis - {metric}")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Histogram", "📦 League Comparison", "🎯 Top Performers", "🌌 Tactical Style Map"])
    
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
                    confidence_label = "🟢 Excellent"
                    confidence_color = "success"
                elif silhouette >= 0.35:
                    confidence_label = "🟡 Good"
                    confidence_color = "warning"
                else:
                    confidence_label = "🔴 Overlap Warning"
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
                        st.caption("⚠️ Clusters may overlap. Archetype assignments less reliable.")
                
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
        
        st.info("🔍 **How to read this**: The axes represent the primary stylistic variances in the dataset. "
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
                st.session_state.page = '🔍 Player Search'
                st.session_state.player_search = clicked_player
                # Clear previous selection to force new lookup based on search
                if 'selected_player' in st.session_state:
                     del st.session_state['selected_player']
                st.rerun()
                    
            except Exception as e:
                # Fail silently or log if index mismatch (e.g. during filtering updates)
                pass
        
        # Explanation
        with st.expander("ℹ️ How to read the Tactical Style Map"):
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
    st.subheader("📊 Archetype Distribution")
    
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
# FOOTER
# ============================================================================

st.divider()
st.caption("⚽ Football Scouting Dashboard | Global League Coverage 2024-2025 | Built with Streamlit")
