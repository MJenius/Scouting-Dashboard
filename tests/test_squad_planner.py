"""
test_squad_planner.py - Test squad planner intent parsing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.llm_integration import (
    AgenticScoutChat,
    extract_filters_fallback
)
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

test_queries = [
    "make a squad plan with bruno fernandes, haaland, rudiger",
    "build a squad with De Gea, Nuno Mendes, Trent Alexander Arnold, Mbappe",
    "create team with Lewandowski, Modric, Van Dijk"
]

print("=" * 80)
print("🧪 SQUAD PLANNER INTENT PARSING TEST")
print("=" * 80)

chat = AgenticScoutChat()

for query in test_queries:
    print(f"\n📝 Query: {query}")
    print("-" * 80)
    
    # Test fallback extraction
    fallback_result = extract_filters_fallback(query)
    print(f"Fallback Result:")
    print(f"  action: {fallback_result.get('action')}")
    print(f"  target_page: {fallback_result.get('target_page')}")
    print(f"  squad_players: {fallback_result.get('squad_players')}")
    
    # Test LLM
    if chat.available:
        try:
            llm_result = chat.parse_intent(query)
            api_params = chat.get_api_params(llm_result)
            print(f"\nLLM Result:")
            print(f"  action: {api_params.get('action')}")
            print(f"  target_page: {api_params.get('target_page')}")
            print(f"  squad_players: {api_params.get('squad_players')}")
        except Exception as e:
            print(f"⚠️ LLM Error: {e}")
    else:
        print("⚠️ Ollama offline - using fallback only")

print("\n" + "=" * 80)
