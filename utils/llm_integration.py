"""
llm_integration.py - Local LLM integration using Ollama (OpenAI-compatible API).

This module provides:
- Agentic Chat implementation (Text-to-JSON filters)
- Context-aware scouting narratives using local models (Llama 3, Mistral)
- Fallback to rule-based generation if Ollama is offline

Configuration:
- Base URL: http://localhost:11434/v1
- Model: llama3 (default), mistral, or any pulled Ollama model
"""

import os
import json
import logging
import requests
import pandas as pd
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime

# Import OpenAI client for Ollama compatibility
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from .narrative_generator import ScoutNarrativeGenerator

# Setup logging
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS & MAPPINGS
# ============================================================================

OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3"

# Strict mapping: Natural language terms → Database stat keys
STAT_KEY_MAPPING = {
    # Offensive terms
    'goalscorer': {'scouting': 'Gls/90_Dominance', 'filter': 'Gls/90', 'display': 'Goals/90'},
    'striker': {'scouting': 'Gls/90_Dominance', 'filter': 'Gls/90', 'display': 'Goals/90'},
    'finisher': {'scouting': 'Gls/90_Dominance', 'filter': 'Gls/90', 'display': 'Goals/90'},
    'clinical': {'scouting': 'Finishing_Efficiency', 'filter': 'Finishing_Efficiency', 'display': 'Finishing Efficiency'},
    
    # Creative terms
    'creative': {'scouting': 'xA90_Dominance', 'filter': 'xA90', 'display': 'xA/90'},
    'playmaker': {'scouting': 'Ast/90_Dominance', 'filter': 'Ast/90', 'display': 'Assists/90'},
    'creator': {'scouting': 'xA90_Dominance', 'filter': 'xA90', 'display': 'xA/90'},
    
    # Defensive terms
    'defender': {'scouting': 'TklW/90_Dominance', 'filter': 'TklW/90', 'display': 'Tackles Won/90'},
    'ball winner': {'scouting': 'TklW/90_Dominance', 'filter': 'TklW/90', 'display': 'Tackles Won/90'},
    'interceptor': {'scouting': 'Int/90_Dominance', 'filter': 'Int/90', 'display': 'Interceptions/90'},
    
    # Expected metrics
    'xg': {'scouting': 'xG90_Dominance', 'filter': 'xG90', 'display': 'xG/90'},
    'xa': {'scouting': 'xA90_Dominance', 'filter': 'xA90', 'display': 'xA/90'},
}

# Database schema description for System Prompt
SYSTEM_PROMPT_FILTER = """You are a football scouting assistant. 
Output ONLY valid JSON. No markdown, no explanations, no chat.

Your task is to convert the user's scouting query into a JSON object with an ACTION and FILTERS.

AVAILABLE SCHEMA (Use only these keys):
{
    "action": str (REQUIRED: "leaderboard" | "compare" | "search" | "find_similar" | "hidden_gems"),
    "target_page": str (Auto-mapped from action: "Leaderboards" | "Head-to-Head" | "Player Search" | "Hidden Gems"),
    "metric": str (For leaderboards: "Gls/90", "Ast/90", "xG90", "xA90", "Sh/90", "SoT/90", "TklW/90", "Int/90"),
    "player_name": str (For search/compare - the main player),
    "compare_player": str (For compare - the second player),
    "min_age": int,
    "max_age": int,
    "league": str (Exact match: "Premier League", "Championship", "League One", "League Two", "National League", "Bundesliga", "La Liga", "Serie A", "Ligue 1"),
    "position": str (Exact match: "FW", "MF", "DF", "GK"),
    "min_goals": float,
    "min_assists": float,
    "min_dominance": float (Z-score: 1.0 = good, 2.0 = elite)
}

ACTION ROUTING RULES:
- "best X" / "top scorers" / "leaders" / "ranking"           -> {"action": "leaderboard", "target_page": "Leaderboards"}
- "compare X with Y" / "X vs Y"                               -> {"action": "compare", "target_page": "Head-to-Head"}
- "find players like X" / "similar to X"                      -> {"action": "find_similar", "target_page": "Player Search"}
- "hidden gems" / "undervalued" / "underrated" / "bargain"    -> {"action": "hidden_gems", "target_page": "Hidden Gems"}
- "search for X" / "find X" / "show me X" (specific player)   -> {"action": "search", "target_page": "Player Search"}

METRIC INFERENCE RULES:
- "scorer", "goals", "prolific", "striker"   -> metric: "Gls/90"
- "creator", "assists", "playmaker"          -> metric: "Ast/90"
- "xg", "expected goals"                     -> metric: "xG90"
- "interceptor", "defensive mid"             -> metric: "Int/90"
- "tackler", "ball winner"                   -> metric: "TklW/90"
- "crosser", "winger"                        -> metric: "Crs/90"

AGE MAPPING RULES:
- "Young" -> {"max_age": 23}
- "Prime" or "Peak" -> {"min_age": 24, "max_age": 29}
- "Experienced" -> {"min_age": 30}

POSITION MAPPING RULES:
- "Striker" / "Forward" -> {"position": "FW"}
- "Midfielder" -> {"position": "MF"}
- "Defender" -> {"position": "DF"}
- "Goalkeeper" / "Keeper" -> {"position": "GK"}

Example 1: "Find me the best striker in Serie A"
Output: {"action": "leaderboard", "target_page": "Leaderboards", "position": "FW", "league": "Serie A", "metric": "Gls/90"}

Example 2: "Compare Haaland with Mbappe"
Output: {"action": "compare", "target_page": "Head-to-Head", "player_name": "Haaland", "compare_player": "Mbappe"}

Example 3: "Find young hidden gems under 21"
Output: {"action": "hidden_gems", "target_page": "Hidden Gems", "max_age": 21}

Example 4: "Show me players similar to Bukayo Saka"
Output: {"action": "find_similar", "target_page": "Player Search", "player_name": "Bukayo Saka"}
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_ollama_available() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_available_models() -> List[str]:
    """Get list of available Ollama models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        return []
    except:
        return []

# ============================================================================
# AGENTIC CHAT CLASS
# ============================================================================

class AgenticScoutChat:
    """
    Local Agentic Chat for converting natural language queries into API filters.
    Uses Ollama (Llama 3 / Mistral) via OpenAI compatible endpoint.
    """
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.client = None
        self.model = model
        self.available = False
        
        if OPENAI_AVAILABLE and is_ollama_available():
            try:
                self.client = OpenAI(
                    base_url=OLLAMA_BASE_URL,
                    api_key="ollama"  # Required but ignored
                )
                self.available = True
                
                # Check if requested model exists, fallback if not
                available_models = get_available_models()
                model_base = model.split(':')[0]
                
                # Simple fuzzy matching for model availability
                if not any(model in m or model_base in m for m in available_models):
                    if available_models:
                        logger.warning(f"Model {model} not found. Using {available_models[0]}")
                        self.model = available_models[0]
                    else:
                        logger.warning("No models found in Ollama.")
            except Exception as e:
                logger.error(f"Failed to initialize AgenticChat: {e}")
                
    def parse_intent(self, user_query: str) -> Dict[str, Any]:
        """
        Convert user query to structured filter JSON.
        """
        if not self.available:
            return {"error": "Local AI is offline"}
            
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_FILTER},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.1,  # Low temperature for deterministic JSON
                max_tokens=150,
                response_format={"type": "json_object"}  # Structured Output
            )
            
            content = response.choices[0].message.content
            
            # Additional JSON cleaning if model outputs markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].strip()
                
            return json.loads(content)
            
        except Exception as e:
            logger.error(f"Intent parsing failed: {e}")
            return {"error": str(e)}

    def get_api_params(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert internal JSON filters to actual API parameter names.
        Now includes navigation actions for page routing.
        """
        params = {}
        
        # Navigation actions (NEW)
        if 'action' in filters: 
            params['action'] = filters['action']
        if 'target_page' in filters: 
            params['target_page'] = filters['target_page']
        if 'metric' in filters: 
            params['metric'] = filters['metric']
        if 'player_name' in filters:
            params['player_name'] = filters['player_name']
        if 'compare_player' in filters:
            params['compare_player'] = filters['compare_player']
        
        # Direct filter mappings
        if 'min_age' in filters: params['min_age'] = filters['min_age']
        if 'max_age' in filters: params['max_age'] = filters['max_age']
        if 'league' in filters: params['league'] = filters['league']
        if 'position' in filters: params['position'] = filters['position']
        
        # Stat mappings
        if 'min_goals' in filters: params['min_gls_90'] = filters['min_goals']
        if 'min_assists' in filters: params['min_ast_90'] = filters['min_assists']
        if 'min_xg' in filters: params['min_xg_90'] = filters['min_xg']
        if 'min_xa' in filters: params['min_xa_90'] = filters['min_xa']
        
        # Dominance mapping (advanced)
        if 'min_dominance' in filters:
            params['min_dominance'] = filters['min_dominance']
            
        return params


# ============================================================================
# NARRATIVE GENERATOR CLASS
# ============================================================================

class LLMScoutNarrativeGenerator:
    """
    Generate scouting narratives using Local LLM (Ollama).
    Falls back to rule-based generator if offline.
    """
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.fallback = ScoutNarrativeGenerator()
        self.client = None
        self.model = model
        self.available = False
        
        if OPENAI_AVAILABLE and is_ollama_available():
            self.client = OpenAI(
                base_url=OLLAMA_BASE_URL,
                api_key="ollama"
            )
            self.available = True
            
            # Ensure model selection logic (same as AgenticChat)
            available_models = get_available_models()
            model_base = model.split(':')[0]
            if not any(model in m or model_base in m for m in available_models):
                if available_models:
                    self.model = available_models[0]

    def _build_prompt(self, player_data: pd.Series, strengths: List[str], weaknesses: List[str]) -> str:
        """Create the scouting report prompt."""
        
        name = player_data.get('Player', 'The player')
        pos = player_data.get('Primary_Pos', 'Unknown')
        league = player_data.get('League', 'Unknown')
        age = player_data.get('Age', 'Unknown')
        archetype = player_data.get('Archetype', 'Unknown')
        
        # Dominance stats if available
        dom_context = ""
        dom_cols = [c for c in player_data.index if '_Dominance' in c]
        if dom_cols:
             top_dom = player_data[dom_cols].astype(float).sort_values(ascending=False).head(1)
             if not top_dom.empty:
                 metric = top_dom.index[0].replace('_Dominance', '')
                 z_score = top_dom.iloc[0]
                 dom_context = f"\n- Dominance: {z_score:+.2f} Z-score for {metric} (League Context)"

        prompt = f"""You are a professional football scout. Write a 3-sentence scouting summary for:
        
Name: {name}
Age: {age}
Position: {pos}
Club/League: {player_data.get('Squad', '')} ({league})
Archetype: {archetype}

Key Stats:
- Goals/90: {player_data.get('Gls/90', 0):.2f}
- Assists/90: {player_data.get('Ast/90', 0):.2f}
{dom_context}

Strengths: {', '.join(strengths)}
Weaknesses: {', '.join(weaknesses)}

Instructions:
1. Synthesize the strengths/weaknesses and archetype into a cohesive narrative.
2. Mention if they are over/under-performing based on age or dominance.
3. Keep it strictly 3 sentences. Professional tone. No fluff.

Summary:"""
        return prompt

    def generate_narrative(self, player_data: pd.Series) -> str:
        """Generate narrative with fallback."""
        if not self.available:
            return self.fallback.generate_narrative(player_data)
        
        try:
            # Get simple strengths/weaknesses for prompt
            # (Reusing logic would be better, but quick implementation here)
            # Simulating basic strength finding:
            dict_data = player_data.to_dict()
            strengths = [k for k,v in dict_data.items() if '_pct' in k and isinstance(v, (int, float)) and v > 80][:3]
            weaknesses = [k for k,v in dict_data.items() if '_pct' in k and isinstance(v, (int, float)) and v < 30][:2]
            
            prompt = self._build_prompt(player_data, strengths, weaknesses)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Local AI Narrative Failed: {e}")
            return self.fallback.generate_narrative(player_data)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def generate_llm_narrative(player_data: pd.Series, use_llm: bool = True) -> str:
    """Wrapper for external calls."""
    if not use_llm:
        return ScoutNarrativeGenerator().generate_narrative(player_data)
    
    generator = LLMScoutNarrativeGenerator()
    return generator.generate_narrative(player_data)
