"""
llm_integration.py - LLM-powered scouting narrative generation using Google Gemini.

This module provides:
- Context-aware scouting summaries using Gemini API
- Fallback to rule-based generator if API unavailable
- Prompt engineering for professional scouting language
- Caching to reduce API calls
"""

import os
from typing import Optional, Dict
import pandas as pd

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from .narrative_generator import ScoutNarrativeGenerator


class LLMScoutNarrativeGenerator:
    """
    Generate context-aware scouting narratives using Google Gemini API.
    
    Features:
    - Professional scouting language
    - Context-aware analysis (league, age, archetype)
    - Strengths/weaknesses identification
    - Tactical insights
    - Fallback to rule-based generator
    """
    
    def __init__(self, api_key: Optional[str] = None, use_llm: bool = True):
        """
        Initialize LLM narrative generator.
        
        Args:
            api_key: Google Gemini API key (or set GEMINI_API_KEY env var)
            use_llm: Whether to use LLM (False = always use rule-based)
        """
        self.use_llm = use_llm and GEMINI_AVAILABLE
        self.fallback_generator = ScoutNarrativeGenerator()
        self.model = None
        
        if not GEMINI_AVAILABLE:
            print("⚠ google-generativeai package not installed. AI features disabled.")
            self.use_llm = False
            return
        
        if self.use_llm:
            # Get API key from parameter or environment
            api_key = api_key or os.getenv('GEMINI_API_KEY')
            
            if api_key:
                print(f"✓ API key found (starts with: {api_key[:10]}...)")
                try:
                    genai.configure(api_key=api_key)
                    # Use Gemini 2.5 Flash (latest stable model)
                    # Note: Must use full path 'models/gemini-2.5-flash'
                    self.model = genai.GenerativeModel('models/gemini-2.5-flash')
                    print("✓ Gemini API initialized successfully with gemini-2.5-flash")
                except Exception as e:
                    print(f"⚠ Gemini API initialization failed: {e}")
                    print("⚠ Falling back to rule-based narrative generation")
                    self.use_llm = False
            else:
                print("⚠ No Gemini API key found in environment or parameters.")
                print("⚠ Please set GEMINI_API_KEY in your .env file.")
                self.use_llm = False
    
    def _build_prompt(
        self,
        player_data: pd.Series,
        top_strengths: list,
        weaknesses: list,
        archetype: str,
        league: str,
        age: int,
    ) -> str:
        """
        Build prompt for Gemini API.
        
        Args:
            player_data: Player statistics
            top_strengths: Top 3 statistical strengths
            weaknesses: Top 2 statistical weaknesses
            archetype: Player archetype
            league: League name
            age: Player age
            include_value: Include market value context
            
        Returns:
            Formatted prompt string
        """
        # Extract key stats
        position = player_data.get('Primary_Pos', 'Unknown')
        squad = player_data.get('Squad', 'Unknown')
        gls_90 = player_data.get('Gls/90', 0)
        ast_90 = player_data.get('Ast/90', 0)
        
        # Extract efficiency metrics
        fin_eff = player_data.get('Finishing_Efficiency', 0.0)
        creat_eff = player_data.get('Creative_Efficiency', 0.0)
        age_z = player_data.get('Age_Z_Score_GA90', 0.0)
        
        # Build context
        prompt = f"""You are a professional football scout writing a concise scouting report.

**Player Profile:**
- Position: {position}
- Age: {age}
- Club: {squad}
- League: {league}
- Archetype: {archetype}

**Performance Indicators:**
- Goals/90: {gls_90:.2f}
- Assists/90: {ast_90:.2f}
- Finishing Efficiency (Goals - xG): {fin_eff:+.2f}
- Creative Efficiency (Ast - xA): {creat_eff:+.2f}
- Age Z-Score (G+A/90 vs cohort): {age_z:+.2f}

**Statistical Strengths:**
{chr(10).join([f'- {s}' for s in top_strengths])}

**Areas for Development:**
{chr(10).join([f'- {w}' for w in weaknesses])}

**Instructions:**
Write a 3 sentence professional scouting summary.
1. Analyze if this player is a high-potential outlier or a high-variance risk based on their xG overperformance (Efficiency metrics).
2. Contextualize their performance relative to their age (using Age Z-Score).
3. Mention their tactical fit based on archetype.

**Tone:** Professional, analytical, balanced.
**Length:** Around 50-60 words (strictly 3 sentences).
**Format:** Plain text paragraph.

**Example Style:**
"This advanced playmaker shows unsustainable finishing efficiency (+0.45), suggesting regression risk despite elite creative numbers. However, his Age Z-Score of +2.1 ranks him as a standout generational talent amongst U23s in the Bundesliga. Defensively, he requires a high-pressing system to mask his limited recovery pace."

Write the scouting summary:"""
        
        return prompt
    
    def generate_narrative(
        self,
        player_data: pd.Series,
        max_retries: int = 2,
    ) -> str:
        """
        Generate scouting narrative using LLM.
        
        Args:
            player_data: Player statistics
            include_value: Include market value context
            max_retries: Number of API retry attempts
            
        Returns:
            Scouting narrative string
            
        Raises:
            RuntimeError: If LLM is not available or API call fails
        """
        # Check if LLM is available
        if not self.use_llm or self.model is None:
            raise RuntimeError(
                "AI narrative generation is not available. "
                "Please ensure GEMINI_API_KEY is set in your .env file and google-generativeai is installed."
            )
        
        try:
            # Extract player info
            archetype = player_data.get('Archetype', 'Unknown')
            league = player_data.get('League', 'Unknown')
            age = int(player_data.get('Age', 0))
            position = player_data.get('Primary_Pos', 'Unknown')
            
            # Identify strengths and weaknesses
            strengths, weaknesses = self._identify_strengths_weaknesses(player_data, position)
            
            # Build prompt
            prompt = self._build_prompt(
                player_data,
                strengths,
                weaknesses,
                archetype,
                league,
                age,
            )
            
            # Call Gemini API
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                narrative = response.text.strip()
                return narrative
            else:
                raise ValueError("Empty response from Gemini API")
        
        except Exception as e:
            # Re-raise with more context
            raise RuntimeError(f"AI narrative generation failed: {str(e)}")
    
    def _identify_strengths_weaknesses(
        self,
        player_data: pd.Series,
        position: str,
    ) -> tuple:
        """
        Identify top strengths and weaknesses from percentile data.
        
        Args:
            player_data: Player statistics
            position: Player position
            
        Returns:
            Tuple of (strengths_list, weaknesses_list)
        """
        # Get percentile columns
        pct_cols = [col for col in player_data.index if col.endswith('_pct')]
        
        if len(pct_cols) == 0:
            return (["Statistical data limited"], ["Further analysis required"])
        
        # Get percentile values
        percentiles = {col.replace('_pct', ''): player_data[col] for col in pct_cols}
        
        # Sort by percentile
        sorted_pcts = sorted(percentiles.items(), key=lambda x: x[1], reverse=True)
        
        # Top 3 strengths (>75th percentile)
        strengths = []
        for metric, pct in sorted_pcts[:5]:
            if pct >= 75:
                strengths.append(f"{metric} ({pct:.0f}th percentile)")
        
        # Top 2 weaknesses (<40th percentile)
        weaknesses = []
        for metric, pct in sorted_pcts[-5:]:
            if pct < 40:
                weaknesses.append(f"{metric} ({pct:.0f}th percentile)")
        
        # Defaults if none found
        if len(strengths) == 0:
            strengths = ["Balanced statistical profile"]
        
        if len(weaknesses) == 0:
            weaknesses = ["No significant weaknesses identified"]
        
        return (strengths[:3], weaknesses[:2])


def generate_llm_narrative(
    player_data: pd.Series,
    api_key: Optional[str] = None,
    use_llm: bool = True,
) -> str:
    """
    Convenience function to generate LLM narrative.
    
    Args:
        player_data: Player statistics
        include_value: Include market value context
        api_key: Google Gemini API key
        use_llm: Whether to use LLM (False = rule-based only)
        
    Returns:
        Scouting narrative string
    """
    generator = LLMScoutNarrativeGenerator(api_key=api_key, use_llm=use_llm)
    return generator.generate_narrative(player_data)
