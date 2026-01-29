"""
visualizations.py - Plotly-based interactive visualizations for Streamlit dashboard.

This module provides reusable chart components:
- Beeswarm plots (player distribution by metric and league)
- Age-curve analysis (performance trends by age)
- Percentile bar charts (progress bars for player rankings)
- League comparison charts
- Archetype distribution charts
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, List, Dict, Tuple

from .constants import FEATURE_COLUMNS, LEAGUE_COLORS, ARCHETYPES


class PlotlyVisualizations:
    """Reusable Plotly chart components for Streamlit."""
    
    @staticmethod
    def beeswarm_by_metric(
        df: pd.DataFrame,
        metric: str,
        league: str = 'all',
        target_player: Optional[str] = None,
        highlight_archetype: Optional[str] = None,
        height: int = 600,
    ) -> go.Figure:
        """
        Create beeswarm plot showing player distribution by metric.
        
        Args:
            df: Player DataFrame with metric columns
            metric: Metric to plot (e.g., 'Gls/90')
            league: Filter by league ('all' for all)
            target_player: Highlight specific player
            highlight_archetype: Color points by archetype
            height: Chart height in pixels
            
        Returns:
            Plotly Figure
        """
        # Filter data
        if league.lower() != 'all':
            plot_df = df[df['League'] == league].copy()
        else:
            plot_df = df.copy()
        
        # Create figure
        fig = go.Figure()
        
        # Get league average
        league_avg = plot_df[metric].mean()
        
        # Create points
        colors = []
        if highlight_archetype:
            for idx, row in plot_df.iterrows():
                if row.get('Archetype') == highlight_archetype:
                    colors.append('#E74C3C')  # Red
                else:
                    colors.append('#95A5A6')  # Gray
        else:
            # Color by league
            colors = [LEAGUE_COLORS.get(league, '#7F8C8D') for league in plot_df['League']]
        
        # Add scatter
        fig.add_trace(go.Scatter(
            x=[metric] * len(plot_df),
            y=plot_df[metric],
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                opacity=0.6,
                line=dict(width=0),
            ),
            text=plot_df['Player'],
            hovertemplate='<b>%{text}</b><br>' + metric + ': %{y:.2f}<extra></extra>',
            name='Players',
        ))
        
        # Add league average line
        fig.add_hline(
            y=league_avg,
            line_dash='dash',
            line_color='blue',
            annotation_text=f'Avg: {league_avg:.2f}',
            annotation_position='right',
        )
        
        # Highlight target player if specified
        if target_player:
            target = plot_df[plot_df['Player'] == target_player]
            if len(target) > 0:
                fig.add_trace(go.Scatter(
                    x=[metric],
                    y=[target.iloc[0][metric]],
                    mode='markers',
                    marker=dict(size=15, color='gold', symbol='star'),
                    name=target_player,
                    hovertemplate=f'<b>{target_player}</b><br>' + metric + ': %{y:.2f}<extra></extra>',
                ))
        
        # Layout
        fig.update_layout(
            title=f'{metric} Distribution - {league if league != "all" else "All Leagues"}',
            xaxis=dict(showticklabels=False),
            yaxis_title=metric,
            height=height,
            hovermode='closest',
            showlegend=False,
            template='plotly_dark',
        )
        
        return fig
    
    @staticmethod
    def age_curve(
        df: pd.DataFrame,
        metric: str,
        position: Optional[str] = None,
        target_player: Optional[str] = None,
        height: int = 500,
    ) -> go.Figure:
        """
        Create age-curve chart showing performance trend by age.
        
        Args:
            df: Player DataFrame
            metric: Metric to analyze
            position: Filter by position (FW, MF, DF, GK)
            target_player: Highlight specific player
            height: Chart height
            
        Returns:
            Plotly Figure
        """
        # Filter by position
        if position:
            plot_df = df[df['Primary_Pos'] == position].copy()
        else:
            plot_df = df.copy()
        
        # Group by age and compute mean
        age_stats = plot_df.groupby('Age').agg({
            metric: ['mean', 'median', 'count'],
        }).reset_index()
        age_stats.columns = ['Age', 'Mean', 'Median', 'Count']
        
        fig = go.Figure()
        
        # Add mean line
        fig.add_trace(go.Scatter(
            x=age_stats['Age'],
            y=age_stats['Mean'],
            mode='lines+markers',
            name='Mean',
            line=dict(color='#3498DB', width=3),
            marker=dict(size=8),
            hovertemplate='Age %{x}<br>' + metric + ' Mean: %{y:.2f}<extra></extra>',
        ))
        
        # Add median line
        fig.add_trace(go.Scatter(
            x=age_stats['Age'],
            y=age_stats['Median'],
            mode='lines',
            name='Median',
            line=dict(color='#E67E22', width=2, dash='dash'),
            hovertemplate='Age %{x}<br>' + metric + ' Median: %{y:.2f}<extra></extra>',
        ))
        
        # Add target player scatter
        if target_player:
            target = plot_df[plot_df['Player'] == target_player]
            if len(target) > 0:
                fig.add_trace(go.Scatter(
                    x=target['Age'],
                    y=target[metric],
                    mode='markers',
                    marker=dict(size=12, color='gold', symbol='star'),
                    name=target_player,
                    hovertemplate=f'<b>{target_player}</b><br>Age: %{{x}}<br>' + metric + ': %{y:.2f}<extra></extra>',
                ))
        
        title_suffix = f' ({position})' if position else ''
        fig.update_layout(
            title=f'{metric} by Age{title_suffix}',
            xaxis_title='Age',
            yaxis_title=metric,
            height=height,
            hovermode='x unified',
            template='plotly_dark',
            legend=dict(x=0.01, y=0.99),
        )
        
        return fig
    
    @staticmethod
    def percentile_progress_bars(
        percentiles: Dict[str, float],
        max_cols: int = 3,
    ) -> pd.DataFrame:
        """
        Create dataframe for percentile progress bars (for st.dataframe).
        
        Args:
            percentiles: Dict mapping feature → percentile rank (0-100)
            max_cols: Max columns per row
            
        Returns:
            DataFrame with styled data
        """
        data = []
        for feature, pct in percentiles.items():
            # Create bar representation
            bar_length = 20
            filled = int(pct / 5)
            bar = "█" * filled + "░" * (bar_length - filled)
            
            # Color coding
            if pct >= 80:
                color = "🟢"  # Green
            elif pct >= 50:
                color = "🟡"  # Yellow
            else:
                color = "🔴"  # Red
            
            data.append({
                'Feature': feature,
                'Percentile': f"{pct:.1f}%",
                'Bar': bar,
                'Ranking': color,
            })
        
        return pd.DataFrame(data)
    
    @staticmethod
    def archetype_distribution(
        df: pd.DataFrame,
        league: str = 'all',
        height: int = 500,
    ) -> go.Figure:
        """
        Create pie/bar chart of archetype distribution.
        
        Args:
            df: Player DataFrame with 'Archetype' column
            league: Filter by league ('all' for all)
            height: Chart height
            
        Returns:
            Plotly Figure
        """
        # Filter
        if league.lower() != 'all':
            plot_df = df[df['League'] == league]
        else:
            plot_df = df
        
        # Count archetypes
        archetype_counts = plot_df['Archetype'].value_counts()
        
        # Get colors
        colors = [ARCHETYPES.get(arch, {}).get('color', '#7F8C8D') 
                 for arch in archetype_counts.index]
        
        # Create figure
        fig = go.Figure(data=[go.Pie(
            labels=archetype_counts.index,
            values=archetype_counts.values,
            marker=dict(colors=colors),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>',
        )])
        
        fig.update_layout(
            title=f'Archetype Distribution - {league if league != "all" else "All Leagues"}',
            height=height,
            template='plotly_dark',
        )
        
        return fig
    
    @staticmethod
    def position_comparison(
        df: pd.DataFrame,
        metric: str,
        height: int = 500,
    ) -> go.Figure:
        """
        Create box plot comparing metric across positions.
        
        Args:
            df: Player DataFrame
            metric: Metric to compare
            height: Chart height
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        for pos in ['FW', 'MF', 'DF', 'GK']:
            pos_data = df[df['Primary_Pos'] == pos][metric]
            fig.add_trace(go.Box(
                y=pos_data,
                name=pos,
                hovertemplate='Position: ' + pos + '<br>' + metric + ': %{y:.2f}<extra></extra>',
            ))
        
        fig.update_layout(
            title=f'{metric} Distribution by Position',
            yaxis_title=metric,
            height=height,
            template='plotly_dark',
            showlegend=False,
        )
        
        return fig
    
    @staticmethod
    def league_comparison(
        df: pd.DataFrame,
        metric: str,
        height: int = 500,
    ) -> go.Figure:
        """
        Create box plot comparing metric across leagues.
        
        Args:
            df: Player DataFrame
            metric: Metric to compare
            height: Chart height
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        for league in df['League'].unique():
            league_data = df[df['League'] == league][metric]
            fig.add_trace(go.Box(
                y=league_data,
                name=league,
                marker=dict(color=LEAGUE_COLORS.get(league, '#7F8C8D')),
                hovertemplate='<b>' + league + '</b><br>' + metric + ': %{y:.2f}<extra></extra>',
            ))
        
        fig.update_layout(
            title=f'{metric} Distribution by League',
            yaxis_title=metric,
            height=height,
            template='plotly_dark',
        )
        
        return fig
    
    @staticmethod
    def player_comparison_bars(
        df: pd.DataFrame,
        player_names: List[str],
        metrics: Optional[List[str]] = None,
        height: int = 500,
    ) -> go.Figure:
        """
        Create grouped bar chart comparing players on multiple metrics.
        
        Args:
            df: Player DataFrame
            player_names: List of players to compare
            metrics: List of metrics to compare (default: first 5 features)
            height: Chart height
            
        Returns:
            Plotly Figure
        """
        if metrics is None:
            metrics = FEATURE_COLUMNS[:5]
        
        fig = go.Figure()
        
        for player in player_names:
            player_data = df[df['Player'] == player]
            if len(player_data) > 0:
                values = [player_data.iloc[0][m] for m in metrics]
                fig.add_trace(go.Bar(
                    name=player,
                    x=metrics,
                    y=values,
                ))
        
        fig.update_layout(
            title='Player Comparison',
            barmode='group',
            xaxis_title='Metrics',
            yaxis_title='Per-90 Value',
            height=height,
            template='plotly_dark',
            hovermode='x unified',
        )
        
        return fig
    
    @staticmethod
    def percentile_distribution(
        df: pd.DataFrame,
        percentile_column: str,
        league: str = 'all',
        height: int = 500,
    ) -> go.Figure:
        """
        Create histogram of percentile distribution.
        
        Args:
            df: Player DataFrame
            percentile_column: Percentile column name (e.g., 'Gls/90_pct')
            league: Filter by league
            height: Chart height
            
        Returns:
            Plotly Figure
        """
        if league.lower() != 'all':
            plot_df = df[df['League'] == league]
        else:
            plot_df = df
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=plot_df[percentile_column],
            nbinsx=20,
            name='Distribution',
            marker=dict(color='#3498DB'),
        ))
        
        # Add mean line
        mean_pct = plot_df[percentile_column].mean()
        fig.add_vline(x=mean_pct, line_dash='dash', annotation_text=f'Mean: {mean_pct:.0f}%')
        
        metric_name = percentile_column.replace('_pct', '')
        fig.update_layout(
            title=f'{metric_name} Percentile Distribution - {league if league != "all" else "All Leagues"}',
            xaxis_title='Percentile Rank',
            yaxis_title='Player Count',
            height=height,
            template='plotly_dark',
            showlegend=False,
        )
        
        return fig
    
    @staticmethod
    def archetype_universe(
        df: pd.DataFrame,
        height: int = 700,
        width: int = 1000,
    ) -> go.Figure:
        """
        Create interactive 2D scatter plot of players by PCA coordinates colored by archetype.
        
        This visualization shows the "player universe" where each dot is a player and their
        position reflects their overall playing style. Players close together have similar profiles.
        
        Args:
            df: Player DataFrame with PCA_X, PCA_Y, Archetype columns
            height: Chart height in pixels
            width: Chart width in pixels
            
        Returns:
            Plotly Figure with archetype universe map
        """
        # Ensure PCA coordinates exist
        if 'PCA_X' not in df.columns or 'PCA_Y' not in df.columns:
            raise ValueError("DataFrame must contain PCA_X and PCA_Y columns from clustering")
        
        plot_df = df.copy()
        
        # Create color mapping for archetypes
        archetype_colors = {}
        for archetype in plot_df['Archetype'].unique():
            # Try to get color from ARCHETYPES constant, fallback to generated color
            if archetype in ARCHETYPES:
                archetype_colors[archetype] = ARCHETYPES[archetype].get('color', '#95A5A6')
            else:
                archetype_colors[archetype] = '#95A5A6'  # Gray for unknown
        
        # Create figure
        fig = px.scatter(
            plot_df,
            x='PCA_X',
            y='PCA_Y',
            color='Archetype',
            hover_data={
                'Player': True,
                'Squad': True,
                'League': True,
                'Age': True,
                'Primary_Pos': True,
                'Gls/90': ':.2f',
                'Ast/90': ':.2f',
                'Archetype': True,
                'PCA_X': False,
                'PCA_Y': False,
            },
            color_discrete_map=archetype_colors,
            title='Archetype Universe: Player Style Map (PCA-2D Projection)',
            labels={
                'PCA_X': 'Playing Style Axis 1',
                'PCA_Y': 'Playing Style Axis 2',
            },
            height=height,
            width=width,
        )
        
        # Customize layout
        fig.update_traces(
            marker=dict(
                size=6,
                opacity=0.7,
                line=dict(width=0.5, color='white'),
            ),
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'Squad: %{customdata[1]}<br>' +
                         'League: %{customdata[2]}<br>' +
                         'Age: %{customdata[3]}<br>' +
                         'Position: %{customdata[4]}<br>' +
                         'Gls/90: %{customdata[5]}<br>' +
                         'Ast/90: %{customdata[6]}<br>' +
                         'Archetype: %{customdata[7]}<extra></extra>',
        )
        
        fig.update_layout(
            template='plotly_dark',
            hovermode='closest',
            xaxis_title='Playing Style Axis 1 (← to →)',
            yaxis_title='Playing Style Axis 2 (← to →)',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(255, 255, 255, 0.3)',
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128, 128, 128, 0.2)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(255, 255, 255, 0.3)',
            ),
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font=dict(color='white'),
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='right',
                x=0.99,
                bgcolor='rgba(0, 0, 0, 0.5)',
                bordercolor='gray',
                borderwidth=1,
            ),
        )
        
        return fig
    
    @staticmethod
    def archetype_universe_filter(
        df: pd.DataFrame,
        selected_archetypes: Optional[List[str]] = None,
        height: int = 700,
        width: int = 1000,
    ) -> go.Figure:
        """
        Create archetype universe with archetype filtering.
        
        Args:
            df: Player DataFrame
            selected_archetypes: List of archetype names to highlight (others grayed out)
            height: Chart height
            width: Chart width
            
        Returns:
            Plotly Figure
        """
        plot_df = df.copy()
        
        # Add highlight column
        if selected_archetypes:
            plot_df['Highlight'] = plot_df['Archetype'].isin(selected_archetypes)
        else:
            plot_df['Highlight'] = True
        
        # Create color mapping
        archetype_colors = {}
        for archetype in plot_df['Archetype'].unique():
            if archetype in ARCHETYPES:
                archetype_colors[archetype] = ARCHETYPES[archetype].get('color', '#95A5A6')
            else:
                archetype_colors[archetype] = '#95A5A6'
        
        # Create figure
        fig = go.Figure()
        
        # Add non-highlighted points (grayed out)
        non_selected = plot_df[~plot_df['Highlight']]
        if len(non_selected) > 0:
            fig.add_trace(go.Scatter(
                x=non_selected['PCA_X'],
                y=non_selected['PCA_Y'],
                mode='markers',
                marker=dict(
                    size=4,
                    color='rgba(150, 150, 150, 0.2)',
                    line=dict(width=0),
                ),
                hoverinfo='skip',
                name='Other Archetypes',
            ))
        
        # Add highlighted points by archetype
        if selected_archetypes:
            for archetype in selected_archetypes:
                arch_df = plot_df[plot_df['Archetype'] == archetype]
                if len(arch_df) > 0:
                    fig.add_trace(go.Scatter(
                        x=arch_df['PCA_X'],
                        y=arch_df['PCA_Y'],
                        mode='markers',
                        name=archetype,
                        marker=dict(
                            size=7,
                            color=archetype_colors.get(archetype, '#95A5A6'),
                            opacity=0.8,
                            line=dict(width=0.5, color='white'),
                        ),
                        hovertemplate='<b>%{customdata[0]}</b><br>' +
                                     'Squad: %{customdata[1]}<br>' +
                                     'League: %{customdata[2]}<br>' +
                                     'Position: %{customdata[3]}<br>' +
                                     'Archetype: ' + archetype + '<extra></extra>',
                        customdata=arch_df[['Player', 'Squad', 'League', 'Primary_Pos']],
                    ))
        
        fig.update_layout(
            title='Archetype Universe: Filtered Player Distribution',
            template='plotly_dark',
            xaxis_title='Playing Style Axis 1',
            yaxis_title='Playing Style Axis 2',
            height=height,
            width=width,
            hovermode='closest',
            plot_bgcolor='#1a1a1a',
            paper_bgcolor='#1a1a1a',
            font=dict(color='white'),
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='right',
                x=0.99,
                bgcolor='rgba(0, 0, 0, 0.5)',
                bordercolor='gray',
                borderwidth=1,
            ),
        )
        
        return fig



# Convenience functions
def create_beeswarm(df, metric, league='all', target=None):
    """Shortcut: Create beeswarm plot."""
    return PlotlyVisualizations.beeswarm_by_metric(df, metric, league, target)

def create_age_curve(df, metric, position=None, target=None):
    """Shortcut: Create age-curve."""
    return PlotlyVisualizations.age_curve(df, metric, position, target)

def create_archetype_pie(df, league='all'):
    """Shortcut: Create archetype distribution."""
    return PlotlyVisualizations.archetype_distribution(df, league)

def create_percentile_bars(percentiles):
    """Shortcut: Create percentile progress bars."""
    return PlotlyVisualizations.percentile_progress_bars(percentiles)

def create_similarity_driver_chart(feature_attribution: Dict[str, float]) -> go.Figure:
    """
    Create horizontal bar chart showing similarity drivers between two players.
    
    Args:
        feature_attribution: Dict of feature_name -> distance (0-1 scale)
        
    Returns:
        Plotly Figure
    """
    features = list(feature_attribution.keys())
    distances = list(feature_attribution.values())
    
    # Invert distances so that low distance (high similarity) shows as long bar
    similarity_scores = [1 - d for d in distances]
    
    fig = go.Figure(data=[
        go.Bar(
            y=features,
            x=similarity_scores,
            orientation='h',
            marker=dict(
                color=similarity_scores,
                colorscale='RdYlGn',
                cmin=0,
                cmax=1,
                showscale=False,
            ),
            text=[f'{s:.0%}' for s in similarity_scores],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Similarity: %{x:.0%}<extra></extra>',
        )
    ])
    
    fig.update_layout(
        title='Similarity Driver Analysis (How Similar are They?)',
        xaxis_title='Similarity Score',
        yaxis_title='Feature',
        height=400,
        template='plotly_dark',
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#1a1a1a',
        font=dict(color='white'),
        xaxis=dict(range=[0, 1]),
    )
    
    return fig

