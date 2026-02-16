"""
test_intent_navigation.py - Quick test to verify intent parsing includes target_page
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.llm_integration import (
    AgenticScoutChat,
    extract_filters_fallback
)
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger(__name__)

test_queries = [
    "Compare Haaland with Mbappe",
    "Show me hidden gems in Serie A",
    "Find the best strikers in Premier League"
]

print("=" * 70)
print("🧪 INTENT PARSING & NAVIGATION TEST")
print("=" * 70)

chat = AgenticScoutChat()

for query in test_queries:
    print(f"\n📝 Query: {query}")
    print("-" * 70)
    
    # Test fallback extraction
    fallback_result = extract_filters_fallback(query)
    print(f"✅ Fallback: action={fallback_result.get('action')}, target_page={fallback_result.get('target_page')}")
    
    # Test LLM + auto-mapping
    if chat.available:
        llm_result = chat.parse_intent(query)
        api_params = chat.get_api_params(llm_result)
        print(f"✅ LLM: action={api_params.get('action')}, target_page={api_params.get('target_page')}")
    else:
        print("⚠️  Ollama offline - using fallback")
        api_params = chat.get_api_params(fallback_result)
        print(f"✅ Fallback→API: action={api_params.get('action')}, target_page={api_params.get('target_page')}")
    
    # Verify target_page is present
    if api_params.get('target_page'):
        print(f"✅ PASS: target_page is present: {api_params.get('target_page')}")
    else:
        print(f"❌ FAIL: target_page is missing!")

print("\n" + "=" * 70)
print("✅ Test complete!")
print("=" * 70)
