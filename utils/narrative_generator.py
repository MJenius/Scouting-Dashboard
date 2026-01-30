"""
narrative_generator.py - Automated Scout's Take generation.

This module provides:
- Natural language narrative generation based on player archetypes and percentiles
- Context-aware descriptions using performance data
- Position-specific insights and recommendations
- Strengths/weaknesses analysis based on percentile rankings

Key components:
- ScoutNarrativeGenerator: Main class for generating narratives
- Template-based text generation with dynamic content
- Performance tier classification (Elite, Excellent, Good, Average, Below Average)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .constants import (
    ARCHETYPES,
    FEATURE_COLUMNS,
    OFFENSIVE_FEATURES,
    POSSESSION_FEATURES,
    LEAGUE_TIERS,
)


class ScoutNarrativeGenerator:
    """
    Generates automated scouting narratives for players.
    
    Creates natural language "Scout's Take" reports using:
    - Player archetype assignment
    - Percentile rankings (position + league specific)
    - Key performance metrics
    - Age and experience data
    - League tier context
    """
    
    # Performance tier thresholds
    PERCENTILE_TIERS = {
        'Elite': 90,
        'Excellent': 80,
        'Very Good': 70,
        'Good': 60,
        'Average': 40,
        'Below Average': 20,
    }
    
    # Age categories for context
    AGE_CATEGORIES = {
        'Young Prospect': (16, 21),
        'Developing': (22, 24),
        'Prime': (25, 28),
        'Experienced': (29, 32),
        'Veteran': (33, 40),
    }
    
    def __init__(self):
        """Initialize the narrative generator."""
        pass
    
    def generate_scouts_take(
        self,
        player_data: pd.Series,
        include_transfer_value: bool = False
    ) -> Dict[str, str]:
        """
        Generate a comprehensive scout's take for a player.
        
        Args:
            player_data: Player row from DataFrame with all stats
            include_transfer_value: Whether to include market value analysis
            
        Returns:
            Dict with narrative sections:
                - overview: High-level summary
                - strengths: Key strengths analysis
                - weaknesses: Areas for improvement
                - recommendation: Transfer/development recommendation
                - full_report: Complete narrative text
        """
        # Extract key data with safe defaults
        name = player_data.get('Player', 'Unknown Player')
        try:
            age = int(player_data['Age']) if not pd.isna(player_data['Age']) else 25
        except (ValueError, KeyError):
            age = 25
        
        position = player_data.get('Primary_Pos', 'MF')
        league = player_data.get('League', 'Unknown')
        archetype = player_data.get('Archetype', 'Unknown')
        minutes = player_data.get('90s', 0)
        
        # Generate sections
        overview = self._generate_overview(player_data)
        strengths = self._generate_strengths(player_data)
        weaknesses = self._generate_weaknesses(player_data)
        recommendation = self._generate_recommendation(player_data, include_transfer_value)
        
        # Compile full report
        full_report = f"{overview}\n\n{strengths}\n\n{weaknesses}\n\n{recommendation}"
        
        return {
            'overview': overview,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendation': recommendation,
            'full_report': full_report,
        }
    
    def _generate_overview(self, player_data: pd.Series) -> str:
        """Generate opening overview paragraph."""
        name = player_data.get('Player', 'Unknown Player')
        try:
            age = int(player_data['Age']) if not pd.isna(player_data['Age']) else 25
        except (ValueError, KeyError):
            age = 25
        
        position = player_data.get('Primary_Pos', 'MF')
        league = player_data.get('League', 'Unknown')
        archetype = player_data.get('Archetype', 'Unknown')
        minutes = player_data.get('90s', 0)
        squad = player_data.get('Squad', 'Unknown')
        
        # Get age category
        age_category = self._get_age_category(age)
        
        # Get archetype description
        archetype_desc = ""
        if archetype in ARCHETYPES:
            archetype_desc = ARCHETYPES[archetype]['description']
        
        # Get league tier context
        league_tier = LEAGUE_TIERS.get(league, 5)
        league_context = self._get_league_context(league, league_tier)
        
        overview = (
            f"**{name}** is a {age}-year-old {position} currently playing for {squad} in the {league}. "
            f"This {age_category.lower()} player has accumulated {minutes:.1f} full matches this season, "
            f"showing {'strong' if minutes >= 20 else 'moderate' if minutes >= 10 else 'limited'} playing time. "
        )
        
        if archetype != 'Unknown':
            overview += f"Our analysis classifies {name.split()[-1]} as a **{archetype}** archetype. "
            overview += f"{archetype_desc} "
        
        overview += league_context
        
        return overview
    
    def _generate_strengths(self, player_data: pd.Series) -> str:
        """Identify and describe key strengths based on percentile data."""
        strengths = []
        percentile_data = []
        
        # Collect percentile rankings
        for feat in FEATURE_COLUMNS:
            pct_col = f'{feat}_pct'
            if pct_col in player_data.index and not pd.isna(player_data[pct_col]):
                percentile_data.append((feat, player_data[pct_col], player_data[feat]))
        
        # Sort by percentile descending
        percentile_data.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 3 strengths (percentile >= 70)
        elite_metrics = [item for item in percentile_data if item[1] >= 70][:3]
        
        if not elite_metrics:
            return "**Strengths:** Limited standout metrics in current league/position comparison. Player shows room for development across most statistical categories."
        
        strength_text = "**Strengths:** "
        strength_bullets = []
        
        for feat, pct, value in elite_metrics:
            tier = self._get_percentile_tier(pct)
            metric_context = self._get_metric_context(feat, value, pct)
            strength_bullets.append(f"- {metric_context}")
        
        strength_text += "\n" + "\n".join(strength_bullets)
        
        return strength_text
    
    def _generate_weaknesses(self, player_data: pd.Series) -> str:
        """Identify areas for improvement based on low percentile rankings."""
        weaknesses = []
        percentile_data = []
        
        # Collect percentile rankings
        for feat in FEATURE_COLUMNS:
            pct_col = f'{feat}_pct'
            if pct_col in player_data.index and not pd.isna(player_data[pct_col]):
                percentile_data.append((feat, player_data[pct_col], player_data[feat]))
        
        # Sort by percentile ascending
        percentile_data.sort(key=lambda x: x[1])
        
        # Get bottom 3 weaknesses (percentile < 40, but only if they're relevant to position)
        position = player_data['Primary_Pos']
        weak_metrics = []
        
        for feat, pct, value in percentile_data:
            if pct < 40 and self._is_metric_relevant(feat, position):
                weak_metrics.append((feat, pct, value))
                if len(weak_metrics) >= 3:
                    break
        
        if not weak_metrics:
            return "**Areas for Development:** Overall well-rounded profile with no significant statistical weaknesses relative to position peers."
        
        weakness_text = "**Areas for Development:** "
        weakness_bullets = []
        
        for feat, pct, value in weak_metrics:
            metric_context = self._get_weakness_context(feat, value, pct)
            weakness_bullets.append(f"- {metric_context}")
        
        weakness_text += "\n" + "\n".join(weakness_bullets)
        
        return weakness_text
    
    def _generate_recommendation(self, player_data: pd.Series, include_value: bool = False) -> str:
        """Generate final recommendation based on all data."""
        name = player_data.get('Player', 'Unknown Player')
        try:
            age = int(player_data['Age']) if not pd.isna(player_data['Age']) else 25
        except (ValueError, KeyError):
            age = 25
        
        position = player_data.get('Primary_Pos', 'MF')
        league = player_data.get('League', 'Unknown')
        archetype = player_data.get('Archetype', 'Unknown')
        
        # Calculate average percentile across key metrics
        avg_percentile = self._get_avg_percentile(player_data)
        
        # Get age category
        age_category = self._get_age_category(age)
        
        rec_text = "**Scout's Recommendation:** "
        
        # Determine recommendation tier
        if avg_percentile >= 80 and age <= 26:
            rec_text += f"🌟 **HIGH PRIORITY TARGET** - {name.split()[-1]} demonstrates elite-level performance "
            rec_text += f"in the {league} with {age_category.lower()} age profile. Strong candidate for immediate first-team impact "
            rec_text += "at higher tier. Recommend comprehensive scouting package with live match observations."
            
        elif avg_percentile >= 70 and age <= 28:
            rec_text += f"⭐ **PROMISING PROSPECT** - Above-average metrics suggest {name.split()[-1]} could contribute "
            rec_text += f"effectively with proper development. {age_category} player with clear pathway to progression. "
            rec_text += "Consider for squad depth or development loan."
            
        elif avg_percentile >= 60:
            rec_text += f"✓ **DEPTH OPTION** - Solid statistical profile indicates reliable squad player potential. "
            rec_text += f"Could provide cover in {position} position. Monitor for 6-12 months before final decision."
            
        else:
            rec_text += f"ℹ️ **MONITOR STATUS** - Current metrics don't suggest immediate transfer priority. "
            rec_text += "Track development trajectory before revisiting."
        
        if include_value and 'Estimated_Value_£M' in player_data.index:
            value = player_data['Estimated_Value_£M']
            if not pd.isna(value):
                rec_text += f" Estimated market value: £{value:.1f}M."
        
        return rec_text
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _get_age_category(self, age: int) -> str:
        """Categorize player by age."""
        for category, (min_age, max_age) in self.AGE_CATEGORIES.items():
            if min_age <= age <= max_age:
                return category
        return "Player"
    
    def _get_percentile_tier(self, percentile: float) -> str:
        """Get performance tier label from percentile."""
        for tier, threshold in self.PERCENTILE_TIERS.items():
            if percentile >= threshold:
                return tier
        return "Below Average"
    
    def _get_league_context(self, league: str, tier: int) -> str:
        """Generate contextual description of league level."""
        # League-specific overrides
        league_specific = {
            'Premier League': "Competing at the highest level of English football provides exceptional defensive organization and tactical discipline.",
            'Championship': "The Championship is a demanding second tier known for its intensity and physicality, ideal for player development.",
            'League One': "League One offers competitive third-tier football where consistent performers can showcase progression potential.",
            'League Two': "Playing in League Two demonstrates ability against professional competition with room for advancement.",
            'National League': "The National League represents fifth-tier football with limited defensive statistics available.",
            'Bundesliga': "The Bundesliga is renowned for its high-pressing systems and tactical innovation at the top level of German football.",
            'La Liga': "La Liga represents the technical peak of Spanish football, emphasizing ball retention and tactical intelligence.",
            'Serie A': "Serie A offers elite Italian football known for its sophisticated defensive structures and tactical discipline.",
            'Ligue 1': "Ligue 1 is a highly competitive top tier in France, becoming increasingly known for its physical profile and technical development.",
        }
        
        if league in league_specific:
            return league_specific[league]
            
        # Generic tier-based fallbacks
        generic_contexts = {
            1: f"Competing at the highest level of {league} provides elite exposure and tactical testing.",
            2: f"Playing in a second-tier league like {league} offers a demanding environment ideal for development.",
            3: f"A mid-tier professional environment like {league} requires consistency and physical resilience.",
            4: f"Competing in {league} provides essential professional experience in a competitive environment.",
            5: f"Playing in {league} offers a platform for progression from a semi-professional or lower-tier professional level.",
        }
        
        return generic_contexts.get(tier, f"Competing in the {league} requires consistent tactical discipline.")
    
    def _get_metric_context(self, metric: str, value: float, percentile: float) -> str:
        """Generate natural language description of a metric strength."""
        tier = self._get_percentile_tier(percentile)
        
        metric_names = {
            'Gls/90': 'Goal-scoring',
            'Ast/90': 'Creativity',
            'Sh/90': 'Shot volume',
            'SoT/90': 'Shooting accuracy',
            'Crs/90': 'Crossing',
            'Int/90': 'Reading of the game',
            'TklW/90': 'Tackling',
            'Fls/90': 'Physical approach',
            'Fld/90': 'Drawing fouls',
        }
        
        metric_insights = {
            'Gls/90': f"{tier} finishing ability ({value:.2f} goals/90, {percentile:.0f}th percentile) marks them as a genuine goal threat",
            'Ast/90': f"{tier} creative output ({value:.2f} assists/90, {percentile:.0f}th percentile) demonstrates excellent vision and final-ball quality",
            'Sh/90': f"{tier} shot frequency ({value:.2f} shots/90, {percentile:.0f}th percentile) shows willingness to test goalkeepers",
            'SoT/90': f"{tier} accuracy ({value:.2f} shots on target/90, {percentile:.0f}th percentile) reflects clinical finishing",
            'Crs/90': f"{tier} crossing ability ({value:.2f} crosses/90, {percentile:.0f}th percentile) provides excellent width and delivery",
            'Int/90': f"{tier} anticipation ({value:.2f} interceptions/90, {percentile:.0f}th percentile) shows strong positional awareness",
            'TklW/90': f"{tier} tackle success rate ({value:.2f} tackles won/90, {percentile:.0f}th percentile) indicates defensive reliability",
            'Fls/90': f"Aggressive approach ({value:.2f} fouls/90, {percentile:.0f}th percentile) demonstrates physical commitment",
            'Fld/90': f"Ability to draw fouls ({value:.2f} fouls drawn/90, {percentile:.0f}th percentile) suggests good ball retention under pressure",
        }
        
        return metric_insights.get(metric, f"{metric}: {value:.2f} ({percentile:.0f}th percentile)")
    
    def _get_weakness_context(self, metric: str, value: float, percentile: float) -> str:
        """Generate natural language description of a weakness."""
        metric_weaknesses = {
            'Gls/90': f"Goal output ({value:.2f}/90, {percentile:.0f}th percentile) below position average - needs to improve finishing",
            'Ast/90': f"Creative contribution ({value:.2f}/90, {percentile:.0f}th percentile) limited - final ball decision-making requires work",
            'Sh/90': f"Shot frequency ({value:.2f}/90, {percentile:.0f}th percentile) below peers - could be more aggressive in attacking areas",
            'SoT/90': f"Shooting accuracy ({value:.2f}/90, {percentile:.0f}th percentile) needs improvement for consistent goal threat",
            'Crs/90': f"Crossing numbers ({value:.2f}/90, {percentile:.0f}th percentile) suggest limited wide delivery",
            'Int/90': f"Interception rate ({value:.2f}/90, {percentile:.0f}th percentile) indicates positioning could improve",
            'TklW/90': f"Tackle success ({value:.2f}/90, {percentile:.0f}th percentile) below average - defensive technique needs refinement",
            'Fls/90': f"Foul frequency ({value:.2f}/90, {percentile:.0f}th percentile) may indicate discipline issues",
            'Fld/90': f"Fouls drawn ({value:.2f}/90, {percentile:.0f}th percentile) suggests difficulty retaining possession under pressure",
        }
        
        return metric_weaknesses.get(metric, f"{metric}: {value:.2f} ({percentile:.0f}th percentile)")
    
    def _is_metric_relevant(self, metric: str, position: str) -> bool:
        """Check if a metric is relevant for a position."""
        relevant_metrics = {
            'FW': ['Gls/90', 'Ast/90', 'Sh/90', 'SoT/90', 'Fld/90'],
            'MF': ['Ast/90', 'Crs/90', 'Int/90', 'TklW/90', 'Gls/90'],
            'DF': ['Int/90', 'TklW/90', 'Crs/90', 'Fls/90'],
            'GK': [],  # Goalkeeper metrics handled separately
        }
        
        return metric in relevant_metrics.get(position, FEATURE_COLUMNS)
    
    def _get_avg_percentile(self, player_data: pd.Series) -> float:
        """Calculate average percentile across all available metrics."""
        percentiles = []
        
        for feat in FEATURE_COLUMNS:
            pct_col = f'{feat}_pct'
            if pct_col in player_data.index and not pd.isna(player_data[pct_col]):
                percentiles.append(player_data[pct_col])
        
        return np.mean(percentiles) if percentiles else 0.0
    
    def analyze_statistical_variance(self, player_data: pd.Series) -> Dict[str, str]:
        """
        Analyze variance in player performance across metrics.
        
        High variance = inconsistent performance (e.g., elite shooter but poor passer)
        Low variance = balanced player
        
        Returns:
            Dict with variance analysis insights
        """
        # Collect percentile data
        percentiles = []
        position = player_data.get('Primary_Pos', 'MF')
        
        for feat in FEATURE_COLUMNS:
            pct_col = f'{feat}_pct'
            if pct_col in player_data.index and not pd.isna(player_data[pct_col]):
                percentiles.append(player_data[pct_col])
        
        if not percentiles:
            return {
                'variance_type': 'Unknown',
                'analysis': 'Insufficient percentile data for variance analysis.',
            }
        
        percentiles = np.array(percentiles)
        variance = np.var(percentiles)
        std_dev = np.std(percentiles)
        min_pct = np.min(percentiles)
        max_pct = np.max(percentiles)
        avg_pct = np.mean(percentiles)
        
        # Classify variance
        if variance > 300:  # High variance threshold
            variance_type = "High Variance"
            interpretation = (
                f"This player shows **highly inconsistent performance** across metrics. "
                f"Elite in some areas (top {100 - max_pct:.0f}%) but struggles in others (bottom {min_pct:.0f}%). "
                f"Typical of specialists or players with a clear strength/weakness profile."
            )
        elif variance > 100:
            variance_type = "Moderate Variance"
            interpretation = (
                f"This player shows **good specialization**. Stronger in some areas "
                f"({100 - max_pct:.0f}th percentile) than others ({min_pct:.0f}th percentile). "
                f"Suitable for tactical roles that leverage strengths."
            )
        else:
            variance_type = "Low Variance"
            interpretation = (
                f"This player is **well-rounded**, showing consistent performance across metrics "
                f"(range: {min_pct:.0f}-{100 - max_pct:.0f}th percentile). "
                f"Valuable for flexible tactical systems and positional versatility."
            )
        
        # Risk assessment for high performers
        if max_pct >= 90 and variance > 200:
            risk = (
                f"⚠️  **Risk Alert**: Elite shooting ({100 - max_pct:.0f}th percentile) but low volume "
                f"({min_pct:.0f}th percentile on Sh/90). May indicate overperformance or "
                f"limited playing time. Monitor sustainability."
            )
        else:
            risk = None
        
        return {
            'variance_type': variance_type,
            'variance_score': float(variance),
            'std_dev': float(std_dev),
            'interpretation': interpretation,
            'range': f"{min_pct:.0f}-{100 - max_pct:.0f}th percentile",
            'risk_note': risk,
        }


def generate_narrative_for_player(player_data: pd.Series, include_value: bool = False, include_variance: bool = True) -> str:
    """
    Convenience function to generate a full scout's take narrative.
    
    Args:
        player_data: Player row from DataFrame
        include_value: Whether to include market value in recommendation
        include_variance: Whether to include variance analysis
        
    Returns:
        Complete narrative text
    """
    generator = ScoutNarrativeGenerator()
    report = generator.generate_scouts_take(player_data, include_value)
    
    if include_variance:
        # Add variance analysis
        variance_analysis = generator.analyze_statistical_variance(player_data)
        variance_section = (
            f"\n\n**Performance Consistency**: {variance_analysis['variance_type']}\n"
            f"{variance_analysis['interpretation']}"
        )
        if variance_analysis.get('risk_note'):
            variance_section += f"\n{variance_analysis['risk_note']}"
        
        report['full_report'] += variance_section
    
    return report['full_report']

