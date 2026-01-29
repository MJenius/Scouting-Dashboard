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
    add_market_value_to_dataframe,
    generate_narrative_for_player,
)
from utils.visualizations import PlotlyVisualizations

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
        'min_90s': 10,
    }

if 'page' not in st.session_state:
    st.session_state.page = '🔍 Player Search'

# ============================================================================
# DATA LOADING (CACHED)
# ============================================================================


@st.cache_data
def load_all_data():
    """Load and process all data once per session."""
    result = process_all_data('english_football_pyramid_master.csv')
    df = result['dataframe']
    scaled = result['scaled_features']
    scaler = result['scaler']
    engine = SimilarityEngine(df, scaled, scaler)
    return result, engine, df, scaled

# Separate clustering resource
@st.cache_resource
def get_clustered_players(df, scaled):
    df_clustered, clusterer = cluster_players(df, scaled)
    df_clustered = add_market_value_to_dataframe(df_clustered)
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
    st.subheader("Scout Bias Settings")
    bias_options = ["Conservative", "Neutral", "Aggressive"]
    selected_bias = st.selectbox(
        "Market Value Bias:",
        options=bias_options,
        index=bias_options.index("Neutral"),
        help="Adjusts the multiplier for market value estimation."
    )
    st.session_state['scout_bias'] = selected_bias
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
        min_value=0,
        max_value=30,
        value=st.session_state.filters['min_90s'],
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
st.caption(f"English Football Pyramid (PL → National League) | {len(df_filtered):,} players after filters")

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
        # Get suggestions
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
                # Get player data
                player_data = df[df['Player'] == selected_player].iloc[0]
                
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
                
                # Special handling for National League players (capped at 33% by design)
                if player_league == 'National League':
                    st.write("**📋 Data Availability**: 🔵 Limited Data Tier (National League)")
                    st.caption(
                        "_National League players have limited statistical coverage in our dataset. "
                        "This does not reflect player quality—further manual scouting recommended._"
                    )
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.info(
                            "⚠️ **Scouting Note**: National League data is capped at 33% completeness by design. "
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

                
                # Market Value
                if 'Estimated_Value_£M' in player_data.index:
                    est_value = player_data['Estimated_Value_£M']
                    value_tier = player_data.get('Value_Tier', 'Unknown')
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("💰 Estimated Value", f"£{est_value:.2f}M")
                    with col2:
                        st.metric("📊 Value Rating", value_tier)
                
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
                                    include_value=True,
                                    use_llm=True
                                )
                            
                            st.success("🤖 AI-Generated Report (Google Gemini)")
                            st.markdown(narrative)
                        else:
                            # Use rule-based generation
                            narrative = generate_narrative_for_player(player_data, include_value=True)
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
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpfile:
                        export_scouting_pdf(player_data, narrative, tmpfile.name)
                        tmpfile.flush()
                        
                        with open(tmpfile.name, "rb") as f:
                            pdf_data = f.read()
                        
                        os.unlink(tmpfile.name)
                    
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
                    percentiles = {}
                    for feat in ['Gls/90', 'Ast/90', 'Sh/90', 'SoT/90', 'Crs/90', 'Int/90', 'TklW/90']:
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
                    
                    # NEW: Add similarity driver for each player
                    st.write("**Top 5 Similar Players** (with primary similarity drivers)")
                    
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
                                # Get top 2 most similar features (primary drivers)
                                top_drivers = list(attribution.items())[:2]
                                driver_text = ", ".join([f"{feat}" for feat, _ in top_drivers])
                                
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
                                    if 'Gls/90' in row.index:
                                        stat_cols[0].metric("Gls/90", f"{row['Gls/90']:.2f}")
                                    if 'Ast/90' in row.index:
                                        stat_cols[1].metric("Ast/90", f"{row['Ast/90']:.2f}")
                                    if 'Age' in row.index:
                                        stat_cols[2].metric("Age", int(row['Age']))
                                    
                                    # Show similarity breakdown
                                    st.write("**Similarity Breakdown:**")
                                    for feat, dist in list(attribution.items())[:5]:
                                        similarity_pct = int((1 - dist) * 100)
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
                                        similarity_pct = int((1 - dist) * 100)
                                        st.write(f"• {feat}: {similarity_pct}% similar")
                                    
                                    st.write("**Key Differences:**")
                                    for feat, dist in most_different_features:
                                        similarity_pct = int((1 - dist) * 100)
                                        st.write(f"• {feat}: {similarity_pct}% similar (Lower match here)")
                                    
                                    # Summary
                                    st.write(f"**Overall Match**: {top_match_score:.1f}% - " +
                                           "Strong profile alignment with key differences in " +
                                           f"{most_different_features[0][0]}")
                            except Exception as e:
                                st.warning(f"Could not calculate similarity breakdown: {e}")
        else:
            st.info("No players found. Try a different search term.")

# ============================================================================
# PAGE 2: HEAD-TO-HEAD
# ============================================================================

elif st.session_state.page == '⚔️ Head-to-Head':
    st.header("⚔️ Head-to-Head Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        player1 = st.selectbox(
            "Select first player:",
            options=df['Player'].unique(),
            key='player1_select'
        )
    
    with col2:
        player2 = st.selectbox(
            "Select second player:",
            options=df['Player'].unique(),
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
                st.metric("Age", f"{int(p1['age'])} vs {int(p2['age'])}")
            with col3:
                st.metric(f"⚡ {p2['name']}", f"{p2['league']} | {p2['position']}")
            with col4:
                st.metric("Match Score", f"{comparison['match_score']:.1f}%")

            st.divider()
            
            # Check if both players are goalkeepers
            is_both_gk = p1['position'] == 'GK' and p2['position'] == 'GK'
            
            # Radar chart
            st.subheader("📊 Radar Comparison")
            
            profile1 = engine.get_player_profile(player1, use_percentiles=False, is_goalkeeper=is_both_gk)
            profile2 = engine.get_player_profile(player2, use_percentiles=False, is_goalkeeper=is_both_gk)
            
            from utils.similarity import RadarChartGenerator
            generator = RadarChartGenerator()
            radar_fig = generator.generate_plotly_radar(
                profile1,
                profile2,
                player1,
                player2,
                use_percentiles=False,
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


# ============================================================================
# PAGE 3: HIDDEN GEMS
# ============================================================================

elif st.session_state.page == '💎 Hidden Gems':
    st.header("💎 Hidden Gems Discovery")
    st.write("Discover young, high-performing players with excellent price-to-performance ratios")
    
    # Add tab selection
    tab1, tab2 = st.tabs(["🔍 Performance Filters", "💰 Best Value Players"])
    
    with tab1:
        # Metric sliders
        st.subheader("⚙️ Filter Criteria")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_goals = st.slider(
                "Min Goals/90:",
                0.0, 2.0, 0.3, step=0.1, key='gems_goals',
                help=METRIC_TOOLTIPS.get('Gls/90', "Minimum goals per 90 minutes.")
            )
        
        with col2:
            min_assists = st.slider(
                "Min Assists/90:",
                0.0, 1.0, 0.1, step=0.05, key='gems_assists',
                help=METRIC_TOOLTIPS.get('Ast/90', "Minimum assists per 90 minutes.")
            )
        
        with col3:
            max_age = st.slider(
                "Max Age:",
                15, 30, 23, key='gems_age',
                help="Maximum player age for hidden gems search."
            )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_percentile = st.slider(
                "Min Percentile:",
                0, 100, 75, step=5, key='gems_percentile',
                help="Minimum percentile for goals/90 (relative to position/league)."
            )
        
        with col2:
            exclude_pl = st.checkbox(
                "Exclude Premier League",
                value=True, key='gems_exclude_pl',
                help="Exclude Premier League players from hidden gems (focus on undervalued leagues)."
            )
        
        with col3:
            min_games = st.slider(
                "Min 90s:",
                0, 30, 10, key='gems_games',
                help=METRIC_TOOLTIPS.get('90s', "Minimum full matches played (90s) for reliability.")
            )
        
        # Apply filters
        gems = df[
            (df['Gls/90'] >= min_goals) &
            (df['Ast/90'] >= min_assists) &
            (df['Age'] <= max_age) &
            (df['Gls/90_pct'] >= min_percentile) &
            (df['90s'] >= min_games)
        ]
        
        if exclude_pl:
            gems = gems[gems['League'] != 'Premier League']
        
        # Results
        st.divider()
        st.subheader(f"🎯 Results ({len(gems)} players found)")
        
        if len(gems) > 0:
            # Sort by goals/90 percentile descending
            gems_sorted = gems.sort_values('Gls/90_pct', ascending=False)
            
            display_cols = ['Player', 'Squad', 'League', 'Age', 'Primary_Pos', 'Gls/90', 'Gls/90_pct', 
                           'Ast/90', 'Estimated_Value_£M', 'Value_Tier', 'Archetype']
            st.dataframe(
                gems_sorted[display_cols].head(50),
                use_container_width=True,
                hide_index=True
            )
            
            # Export button
            csv = gems_sorted.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name=f"hidden_gems_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No players match the selected criteria. Try adjusting the filters.")
    
    with tab2:
        # Price vs Performance Analysis
        st.subheader("💰 Best Value Players - Price vs Performance")
        st.write("Find the biggest bargains: high performance relative to estimated market value")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            value_league = st.selectbox(
                "Filter by league:",
                options=['all'] + LEAGUES,
                key='value_league'
            )
        
        with col2:
            value_position = st.selectbox(
                "Filter by position:",
                options=['all'] + PRIMARY_POSITIONS,
                key='value_position'
            )
        
        with col3:
            max_value_filter = st.slider(
                "Max Value (£M):",
                min_value=0.0,
                max_value=50.0,
                value=10.0,
                step=0.5,
                key='max_value'
            )
        
        # Filter data
        value_df = df_filtered.copy()
        
        if value_league != 'all':
            value_df = value_df[value_df['League'] == value_league]
        
        if value_position != 'all':
            value_df = value_df[value_df['Primary_Pos'] == value_position]
        
        if max_value_filter > 0:
            value_df = value_df[value_df['Estimated_Value_£M'] <= max_value_filter]
        
        # Sort by Value_Score
        value_df = value_df.sort_values('Value_Score', ascending=False)
        
        st.divider()
        
        # Show top bargains
        st.subheader(f"🏆 Top 25 Best Value Players")
        
        if len(value_df) > 0:
            # Calculate average percentile for display
            percentile_cols = [col for col in value_df.columns if col.endswith('_pct')]
            value_df['Avg_Percentile'] = value_df[percentile_cols].mean(axis=1)
            
            display_cols = [
                'Player', 'Squad', 'League', 'Age', 'Primary_Pos',
                'Estimated_Value_£M', 'Avg_Percentile', 'Value_Score', 'Value_Tier',
                'Gls/90', 'Ast/90', 'Archetype'
            ]
            
            # Format for display
            top_value = value_df.head(25).copy()
            top_value['Avg_Percentile'] = top_value['Avg_Percentile'].round(1)
            top_value['Value_Score'] = top_value['Value_Score'].round(2)
            
            st.dataframe(
                top_value[display_cols],
                use_container_width=True,
                hide_index=True
            )
            
            # Value distribution chart
            st.divider()
            st.subheader("📊 Value Score Distribution")
            
            import plotly.express as px
            
            fig = px.scatter(
                value_df.head(100),
                x='Estimated_Value_£M',
                y='Avg_Percentile',
                size='Value_Score',
                color='League',
                hover_data=['Player', 'Squad', 'Age', 'Primary_Pos'],
                title='Price vs Performance (Larger bubbles = Better value)',
                color_discrete_map=LEAGUE_COLORS,
                labels={
                    'Estimated_Value_£M': 'Estimated Value (£M)',
                    'Avg_Percentile': 'Average Percentile',
                }
            )
            
            fig.update_layout(
                height=500,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Export
            csv = value_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Value Analysis CSV",
                data=csv,
                file_name=f"value_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No players match the selected criteria.")

# ============================================================================
# PAGE 4: LEADERBOARDS
# ============================================================================

elif st.session_state.page == '🏆 Leaderboards':
    st.header("🏆 League Leaderboards")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        metric = st.selectbox(
            "Select metric:",
            options=['Gls/90', 'Ast/90', 'Sh/90', 'SoT/90', 'Crs/90', 'Int/90', 'TklW/90'],
            key='leaderboard_metric'
        )
    
    with col2:
        league = st.selectbox(
            "Select league:",
            options=['all'] + LEAGUES,
            key='leaderboard_league'
        )
    
    with col3:
        position_filter = st.selectbox(
            "Filter by position:",
            options=['all'] + PRIMARY_POSITIONS,
            key='leaderboard_position'
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
    tab1, tab2, tab3 = st.tabs(["📈 Histogram", "📦 League Comparison", "🎯 Top Performers"])
    
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

    
    # NEW: PCA Archetype Universe Map
    st.divider()
    st.subheader("🌌 Archetype Universe - Player Style Map")
    st.write("**Tactical Hybrids**: Players positioned between archetypes represent hybrid playing styles")
    
    # Filter options for the universe map
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Get unique archetypes
        all_archetypes = sorted(df_filtered['Archetype'].unique())
        selected_archetypes = st.multiselect(
            "Filter by archetype (leave empty for all):",
            options=all_archetypes,
            default=[],
            key='archetype_filter'
        )
    
    with col2:
        show_all = st.checkbox("Show all players", value=True, key='show_all_universe')
    
    # Create the universe map
    if show_all or not selected_archetypes:
        # Show all players
        universe_fig = PlotlyVisualizations.archetype_universe(
            df_filtered,
            height=700,
        )
    else:
        # Show filtered archetypes
        universe_fig = PlotlyVisualizations.archetype_universe_filter(
            df_filtered,
            selected_archetypes=selected_archetypes,
            height=700,
        )
    
    st.plotly_chart(universe_fig, use_container_width=True)
    
    # Explanation
    with st.expander("ℹ️ How to read the Archetype Universe"):
        st.markdown("""
        **What is this?**
        - Each dot represents a player, positioned based on their playing style (PCA analysis)
        - Players close together have similar statistical profiles
        - Colors represent the assigned archetype
        
        **Finding Tactical Hybrids:**
        - Look for players positioned **between** two archetype clusters
        - These players combine characteristics from multiple playing styles
        - Useful for finding versatile players or unique profiles
        
        **Technical Details:**
        - Built using Principal Component Analysis (PCA) with 67.4% variance explained
        - 2D projection of 9-dimensional feature space
        - K-Means clustering (k=8) assigns archetype labels
        """)
    
    # Show archetype statistics
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
st.caption("⚽ Football Scouting Dashboard | Data: English Football Pyramid 2024-2025 | Built with Streamlit")
