"""
llm_integration.py - Local LLM integration using Ollama (OpenAI-compatible API).

This module provides:
- Agentic Chat implementation (Text-to-JSON filters)
- Context-aware scouting narratives using local models (Llama 3, Mistral)
- Fallback to rule-based generation if Ollama is offline
- Response caching for query/intent performance optimization
- Timeout & retry logic with fallback filter extraction

Configuration:
- Base URL: http://localhost:11434/v1
- Model: mistral (recommended for speed), llama3, or any pulled Ollama model
- Cache TTL: 30 minutes for intent queries
- Timeout: 5 seconds for API calls
"""

import os
import json
import logging
import requests
import pandas as pd
import time
import hashlib
import re
from functools import lru_cache
from typing import Optional, Dict, List, Any, Tuple
from datetime import datetime, timedelta

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
# PERFORMANCE CONSTANTS
# ============================================================================
API_TIMEOUT = 5.0  # Aggressive timeout to fail-fast
CACHE_TTL_SECONDS = 1800  # 30 minutes cache for intent queries
DEFAULT_MODEL = "mistral"  # Faster than llama3 for intent parsing

# ============================================================================
# CONSTANTS & MAPPINGS
# ============================================================================

OLLAMA_BASE_URL = "http://localhost:11434/v1"

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

# ============================================================================
# QUERY CACHE FOR PERFORMANCE OPTIMIZATION
# ============================================================================

class IntentQueryCache:
    """Simple TTL-based cache for intent parsing results."""
    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS):
        self.cache: Dict[str, Tuple[Dict, float]] = {}
        self.ttl = ttl_seconds
    
    def _hash_query(self, query: str) -> str:
        """Create hash of normalized query for cache key."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, query: str) -> Optional[Dict]:
        """Get cached result if exists and not expired."""
        key = self._hash_query(query)
        if key in self.cache:
            result, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                logger.debug(f"Cache HIT for query: {query[:50]}...")
                return result
            else:
                # Expired, remove
                del self.cache[key]
                logger.debug(f"Cache EXPIRED for query: {query[:50]}...")
        return None
    
    def set(self, query: str, result: Dict) -> None:
        """Store result in cache."""
        key = self._hash_query(query)
        self.cache[key] = (result, time.time())
        logger.debug(f"Cache MISS → STORED for query: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()

# Global cache instance
_intent_cache = IntentQueryCache()

# ============================================================================
# FALLBACK FILTER EXTRACTION (Rule-based, Fast)
# ============================================================================

def extract_filters_fallback(query: str) -> Dict[str, Any]:
    """
    Fallback rule-based filter extraction when Ollama is slow/offline.
    Uses regex patterns and keyword matching. ~50ms latency vs 1-2s for LLM.
    """
    filters = {}
    q = query.lower()
    
    # Determine action
    if any(x in q for x in ['compare', 'vs', 'head to head', 'versus']):
        filters['action'] = 'compare'
        filters['target_page'] = 'Head-to-Head'
    elif any(x in q for x in ['best', 'top', 'ranking', 'leaderboard']):
        filters['action'] = 'leaderboard'
        filters['target_page'] = 'Leaderboards'
    elif any(x in q for x in ['hidden gem', 'undervalued', 'bargain', 'cheap']):
        filters['action'] = 'hidden_gems'
        filters['target_page'] = 'Hidden Gems'
    elif any(x in q for x in ['find', 'search', 'show me']) and 'like' not in q:
        filters['action'] = 'search'
        filters['target_page'] = 'Player Search'
    elif any(x in q for x in ['like', 'similar', 'next']):
        filters['action'] = 'find_similar'
        filters['target_page'] = 'Hidden Gems'
    elif any(x in q for x in ['squad plan', 'build squad', 'build team']):
        filters['action'] = 'squad_planner'
        filters['target_page'] = 'Squad Planner'
    elif any(x in q for x in ['squad analysis', 'analyze']):
        filters['action'] = 'squad_analysis'
        filters['target_page'] = 'Squad Analysis'
    else:
        filters['action'] = 'search'
        filters['target_page'] = 'Player Search'
    
    # Extract age ranges
    age_match = re.search(r'(under|younger than)\s+(\d+)', q)
    if age_match:
        filters['max_age'] = int(age_match.group(2))
    age_match = re.search(r'(over|older than)\s+(\d+)', q)
    if age_match:
        filters['min_age'] = int(age_match.group(2))
    
    # Extract positions
    pos_map = {'striker': 'FW', 'forward': 'FW', 'winger': 'FW', 'midfielder': 'MF', 
               'defender': 'DF', 'goalkeeper': 'GK', 'keeper': 'GK'}
    for term, pos in pos_map.items():
        if term in q:
            filters['position'] = pos
            break
    
    # Extract metrics
    metric_map = {'goal': 'Gls/90', 'score': 'Gls/90', 'assist': 'Ast/90', 
                  'xg': 'xG90', 'xa': 'xA90', 'tackle': 'TklW/90'}
    for term, metric in metric_map.items():
        if term in q:
            filters['metric'] = metric
            break
    
    # Extract league names (simplified)
    leagues = ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1', 
               'Championship', 'League One', 'League Two', 'National League']
    for league in leagues:
        if league.lower() in q:
            filters['league'] = league
            break
    
    # Try to extract player names (simple heuristic: capitalized words)
    words = query.split()
    cap_words = [w.strip('.,;:') for w in words if w and w[0].isupper() and len(w) > 2]
    
    if cap_words:
        if 'compare' in q or 'vs' in q:
            # For compare: extract first two capitalized names
            if len(cap_words) >= 2:
                filters['player_name'] = cap_words[0]
                filters['compare_player'] = cap_words[1]
        elif 'squad plan' in q or 'build squad' in q or 'build team' in q or 'create team' in q:
            # For squad planner: extract ALL capitalized names as squad players
            filters['squad_players'] = cap_words
            logger.info(f"Extracted squad players: {cap_words}")
        else:
            # For other actions: just first name
            filters['player_name'] = cap_words[0]
    
    logger.info(f"Fallback extraction: {filters}")
    return filters

# Database schema description for System Prompt
# OPTIMIZED: Reduced from 500+ to ~250 tokens, kept only critical mappings
SYSTEM_PROMPT_FILTER = """You are a football scouting assistant. Output ONLY valid JSON. No markdown.

TASK: Convert user query to JSON with ACTION, TARGET_PAGE, and optional FILTERS.

ACTIONS & PAGES:
- "best" / "top" / "ranking" → action: "leaderboard", target_page: "Leaderboards"
- "compare X vs Y" → action: "compare", target_page: "Head-to-Head", player_name: "X", compare_player: "Y"
- "find like X" / "similar" → action: "find_similar", target_page: "Hidden Gems", player_name: "X"
- "hidden gems" / "undervalued" → action: "hidden_gems", target_page: "Hidden Gems"
- "search X" / "find X" → action: "search", target_page: "Player Search", player_name: "X"
- "squad analysis X" → action: "squad_analysis", target_page: "Squad Analysis", team_name: "X"
- "build squad" / "make squad" / "create team" with player list → action: "squad_planner", target_page: "Squad Planner", squad_players: [list of all player names]

OPTIONAL FILTERS: league, position, min_age, max_age, metric, priority

IMPORTANT for squad_planner:
- Extract ALL player names mentioned (separated by commas, "and", or "with")
- Return as array: squad_players: ["Bruno Fernandes", "Haaland", "Rudiger"]
- Do NOT abbreviate names

Example 1: "Compare Haaland with Mbappe"
{"action": "compare", "target_page": "Head-to-Head", "player_name": "Haaland", "compare_player": "Mbappe"}

Example 2: "Show hidden gems under 21 in Serie A"
{"action": "hidden_gems", "target_page": "Hidden Gems", "max_age": 21, "league": "Serie A"}

Example 3: "Make a squad with Bruno Fernandes, Haaland, Rudiger"
{"action": "squad_planner", "target_page": "Squad Planner", "squad_players": ["Bruno Fernandes", "Haaland", "Rudiger"]}

ALWAYS include action and target_page. Return valid JSON only."""

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
        Convert user query to structured filter JSON with caching and fallback.
        
        Performance optimizations:
        1. Checks cache first (~0ms if hit)
        2. Uses timeout to fail-fast on slow Ollama (~5s max)
        3. Falls back to rule-based extraction if LLM fails (~50ms)
        4. Caches successful results for future identical queries
        """
        if not self.available:
            logger.info("Ollama offline, using fallback extraction")
            return extract_filters_fallback(user_query)
        
        # 1. CHECK CACHE FIRST
        cached = _intent_cache.get(user_query)
        if cached is not None:
            return cached
        
        # 2. TRY LLM WITH TIMEOUT & RETRY
        max_retries = 2
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_FILTER},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.1,
                    max_tokens=100,  # Reduced from 150
                    timeout=API_TIMEOUT,
                    response_format={"type": "json_object"}
                )
                
                elapsed = time.time() - start_time
                logger.info(f"Intent parsing via LLM ({self.model}): {elapsed:.2f}s")
                
                content = response.choices[0].message.content
                
                # Clean JSON if wrapped in markdown
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()
                
                result = json.loads(content)
                
                # Cache successful result
                _intent_cache.set(user_query, result)
                return result
                
            except requests.exceptions.Timeout:
                logger.warning(f"Ollama timeout on attempt {attempt+1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)  # Brief backoff before retry
                    continue
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parse error: {e}")
                if attempt < max_retries - 1:
                    continue
            except Exception as e:
                logger.warning(f"LLM error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(0.5)
                    continue
        
        # 3. FALLBACK TO RULE-BASED EXTRACTION
        logger.info("LLM failed/timed out, using fallback extraction")
        result = extract_filters_fallback(user_query)
        _intent_cache.set(user_query, result)
        return result

    def get_api_params(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert internal JSON filters to actual API parameter names.
        Now includes navigation actions for page routing.
        Automatically maps action → target_page if not explicitly provided.
        """
        params = {}
        
        # Auto-map action to target_page if not present
        action_to_page_map = {
            'leaderboard': 'Leaderboards',
            'compare': 'Head-to-Head',
            'search': 'Player Search',
            'find_similar': 'Hidden Gems',
            'hidden_gems': 'Hidden Gems',
            'squad_analysis': 'Squad Analysis',
            'squad_planner': 'Squad Planner'
        }
        
        # Navigation actions
        if 'action' in filters:
            params['action'] = filters['action']
            # Auto-map action to target_page if not explicitly set
            if 'target_page' not in filters:
                params['target_page'] = action_to_page_map.get(filters['action'])
        
        if 'target_page' in filters:
            params['target_page'] = filters['target_page']
        
        if 'metric' in filters: 
            params['metric'] = filters['metric']
        if 'player_name' in filters:
            params['player_name'] = filters['player_name']
        if 'compare_player' in filters:
            params['compare_player'] = filters['compare_player']
        
        # Squad actions
        if 'team_name' in filters:
            params['team_name'] = filters['team_name']
        if 'squad_players' in filters:
            params['squad_players'] = filters['squad_players']
        
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
        
        # Benchmark specific
        if 'priority' in filters:
            params['priority'] = filters['priority']
        if 'target_league' in filters:
            params['target_league'] = filters['target_league']
        
        # Debug logging
        logger.info(f"API params: {params}")
        return params


# ============================================================================
# NARRATIVE GENERATOR CLASS
# ============================================================================

class LLMScoutNarrativeGenerator:
    """
    Generate scouting narratives using Local LLM (Ollama).
    Falls back to rule-based generator if offline.
    """
    
    def __init__(self, model: str = DEFAULT_MODEL, use_llm: bool = True, **kwargs):
        self.fallback = ScoutNarrativeGenerator()
        self.client = None
        self.model = model
        self.available = False
        self.use_llm = use_llm
        
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

        prompt = f"""You are a professional football scout. Write a comprehensive, deep-dive scouting report (3-4 paragraphs, approx 250 words) for the following player:
        
PLAYER PROFILE:
- Name: {name}
- Age: {age}
- Position: {pos}
- Club/League: {player_data.get('Squad', '')} ({league})
- Archetype: {archetype}

PERFORMANCE CONTEXT:
- Goals/90: {player_data.get('Gls/90', 0):.2f}
- Assists/90: {player_data.get('Ast/90', 0):.2f}
{dom_context}

STATISTICAL PROFILE (Percentiles vs Peers):
- Strengths: {', '.join(strengths)}
- Weaknesses: {', '.join(weaknesses)}

INSTRUCTIONS:
1. Provide a highly unique, engaging, and in-depth scouting perspective that goes beyond basic stats.
2. Formulate a strong opinion on their playstyle and potential tactical roles, comparing them to well-known player archetypes if applicable.
3. Highlight any hidden value or critical flaws derived from their statistical profile.
4. Use professional, evocative football scouting terminology (e.g., 'half-spaces', 'progressive carries', 'rest defense').
5. Do not just list the stats provided above; synthesize them into a coherent profile of what it's like to watch this player.
6. Target a length of 3 engaging paragraphs (approx 200-250 words) that feel distinctly human and perceptive.

REPORT:"""
        return prompt

    def generate_narrative(self, player_data: pd.Series) -> str:
        """Generate narrative with timeout and fallback."""
        if not self.available or not self.use_llm:
            return self.fallback.generate_scouts_take(player_data)['full_report']
        
        try:
            dict_data = player_data.to_dict()
            strengths = [k for k,v in dict_data.items() if '_pct' in k and isinstance(v, (int, float)) and v > 80][:3]
            weaknesses = [k for k,v in dict_data.items() if '_pct' in k and isinstance(v, (int, float)) and v < 30][:2]
            
            prompt = self._build_prompt(player_data, strengths, weaknesses)
            
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=800,  # Reduced from 1000
                timeout=API_TIMEOUT * 2  # Allow longer for narrative generation
            )
            elapsed = time.time() - start_time
            logger.info(f"Narrative generation: {elapsed:.2f}s")
            
            return response.choices[0].message.content.strip()
            
        except requests.exceptions.Timeout:
            logger.warning("Narrative generation timeout, using fallback")
            return self.fallback.generate_scouts_take(player_data)['full_report']
        except Exception as e:
            logger.warning(f"LLM Narrative failed ({e}), using fallback")
            return self.fallback.generate_scouts_take(player_data)['full_report']


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def generate_llm_narrative(player_data: pd.Series, use_llm: bool = True) -> str:
    """Wrapper for external calls."""
    if not use_llm:
        return ScoutNarrativeGenerator().generate_scouts_take(player_data)['full_report']
    
    generator = LLMScoutNarrativeGenerator()
    return generator.generate_narrative(player_data)
