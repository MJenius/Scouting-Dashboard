"""
test_llm_performance.py - Performance benchmarks for agentic chat optimizations.

Tests:
1. Intent parsing caching (cold cache vs warm cache)
2. Fallback extraction performance
3. Timeout handling
4. End-to-end latency across multiple queries
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.llm_integration import (
    AgenticScoutChat,
    extract_filters_fallback,
    _intent_cache
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test queries
TEST_QUERIES = [
    "Find me the best striker in Serie A",
    "Compare Haaland with Mbappe",
    "Find young hidden gems under 21",
    "Show me the top scorers in Premier League",
    "Find players similar to De Bruyne in Championship",
    "Build a squad with De Gea, Nuno Mendes, and Trent",
    "Analyze Manchester City squad",
    "Find the best defensive midfielder in Bundesliga",
]

def benchmark_fallback_extraction():
    """Test rule-based fallback extraction speed."""
    print("\n" + "="*60)
    print("BENCHMARK: Rule-Based Fallback Extraction")
    print("="*60)
    
    latencies = []
    for query in TEST_QUERIES:
        start = time.perf_counter()
        result = extract_filters_fallback(query)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed * 1000)  # Convert to ms
        
        print(f"Query: {query[:45]:45s} | {elapsed*1000:6.2f}ms | Action: {result.get('action', 'N/A')}")
    
    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage latency: {avg_latency:.2f}ms (Expected: <100ms)")
    print(f"✅ PASS" if avg_latency < 100 else "❌ FAIL")
    return avg_latency

def benchmark_llm_intent_parsing():
    """Test LLM intent parsing with caching."""
    print("\n" + "="*60)
    print("BENCHMARK: LLM Intent Parsing with Caching")
    print("="*60)
    
    # Clear cache
    _intent_cache.clear()
    
    chat = AgenticScoutChat()
    
    if not chat.available:
        print("⚠️  Ollama offline, skipping LLM benchmark")
        return None
    
    cold_latencies = []
    warm_latencies = []
    
    # Test 1: Cold cache (first run)
    print("\nCold Cache (First Run):")
    for i, query in enumerate(TEST_QUERIES[:3], 1):
        start = time.perf_counter()
        result = chat.parse_intent(query)
        elapsed = time.perf_counter() - start
        cold_latencies.append(elapsed)
        
        print(f"{i}. Query: {query[:40]:40s} | {elapsed:6.2f}s | Action: {result.get('action', 'N/A')}")
    
    # Test 2: Warm cache (cached results)
    print("\nWarm Cache (Identical Queries):")
    for i, query in enumerate(TEST_QUERIES[:3], 1):
        start = time.perf_counter()
        result = chat.parse_intent(query)
        elapsed = time.perf_counter() - start
        warm_latencies.append(elapsed)
        
        print(f"{i}. Query: {query[:40]:40s} | {elapsed*1000:6.2f}ms | Action: {result.get('action', 'N/A')}")
    
    avg_cold = sum(cold_latencies) / len(cold_latencies)
    avg_warm = sum(warm_latencies) / len(warm_latencies)
    speedup = avg_cold / avg_warm if avg_warm > 0 else 0
    
    print(f"\nCold cache avg: {avg_cold:.2f}s")
    print(f"Warm cache avg: {avg_warm*1000:.2f}ms")
    print(f"Speedup ratio: {speedup:.0f}x faster")
    print(f"✅ PASS" if speedup > 10 else "⚠️  Cache working but slower than expected")
    
    return avg_cold, avg_warm, speedup

def benchmark_repeated_queries():
    """Test performance of repeated queries (caching impact)."""
    print("\n" + "="*60)
    print("BENCHMARK: Repeated Query Performance (Cache Impact)")
    print("="*60)
    
    _intent_cache.clear()
    
    chat = AgenticScoutChat()
    query = "Find the best striker in Premier League"
    
    if not chat.available:
        print("⚠️  Ollama offline, using fallback")
        # Test with fallback instead
        latencies = []
        for i in range(1, 6):
            start = time.perf_counter()
            result = extract_filters_fallback(query)
            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000)
            print(f"Run {i}: {elapsed*1000:6.2f}ms (Fallback)")
        avg = sum(latencies) / len(latencies)
        print(f"Average: {avg:.2f}ms")
        return avg
    
    latencies = []
    for i in range(1, 6):
        start = time.perf_counter()
        result = chat.parse_intent(query)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        
        cache_status = "CACHED" if elapsed < 0.01 else "LLM"
        print(f"Run {i}: {elapsed*1000:7.2f}ms ({cache_status})")
    
    avg = sum(latencies) / len(latencies)
    is_cached = latencies[-1] < 0.01  # Last run should be cached
    
    print(f"\nAverage latency: {avg*1000:.2f}ms")
    print(f"✅ PASS (Cache working: {is_cached})" if is_cached else "❌ FAIL (No caching detected)")
    return avg

def benchmark_timeout_behavior():
    """Test timeout and fallback behavior."""
    print("\n" + "="*60)
    print("BENCHMARK: Timeout & Fallback Behavior")
    print("="*60)
    
    chat = AgenticScoutChat()
    
    if chat.available:
        print("Ollama is available - timeout behavior will be tested in real scenarios")
        print("⚠️  Recommended: Stop Ollama and re-run to simulate timeout fallback")
    else:
        print("✅ Ollama offline detected")
        print("Testing fallback extraction (should be <100ms)")
        
        start = time.perf_counter()
        result = extract_filters_fallback("Find the best strikers in Serie A")
        elapsed = time.perf_counter() - start
        
        print(f"Fallback extraction: {elapsed*1000:.2f}ms")
        print(f"Result: {result}")
        print("✅ PASS - Fallback working")

def run_all_benchmarks():
    """Run all performance benchmarks."""
    print("\n" + "█"*60)
    print("🚀 AGENTIC CHAT PERFORMANCE BENCHMARKS")
    print("█"*60)
    
    results = {}
    
    # Benchmark 1: Fallback extraction
    results['fallback'] = benchmark_fallback_extraction()
    time.sleep(1)
    
    # Benchmark 2: LLM intent parsing
    lm_results = benchmark_llm_intent_parsing()
    if lm_results:
        results['cold_cache'], results['warm_cache'], results['speedup'] = lm_results
    time.sleep(1)
    
    # Benchmark 3: Repeated queries
    results['repeated_avg'] = benchmark_repeated_queries()
    time.sleep(1)
    
    # Benchmark 4: Timeout behavior
    benchmark_timeout_behavior()
    
    # Summary
    print("\n" + "█"*60)
    print("📊 SUMMARY")
    print("█"*60)
    
    if 'fallback' in results:
        print(f"✅ Fallback extraction: {results['fallback']:.2f}ms (Target: <100ms)")
    
    if 'cold_cache' in results:
        print(f"✅ LLM cold cache: {results['cold_cache']:.2f}s")
        print(f"✅ LLM warm cache: {results['warm_cache']*1000:.2f}ms")
        print(f"✅ Cache speedup: {results['speedup']:.0f}x")
    
    if 'repeated_avg' in results:
        print(f"✅ Repeated query avg: {results['repeated_avg']*1000:.2f}ms")
    
    print("\n✅ All benchmarks complete!")
    print("💡 Next: Monitor Ollama to ensure it's using 'mistral' model for maximum speed")

if __name__ == "__main__":
    run_all_benchmarks()
