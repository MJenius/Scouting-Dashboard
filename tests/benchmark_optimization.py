
import time
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_integration import AgenticScoutChat, LLMScoutNarrativeGenerator
from backend.app.services.evaluator import ModelEvaluator

def benchmark_llm():
    print("\n--- LLM Performance Benchmark (Llama 3.2) ---")
    chat = AgenticScoutChat()
    if not chat.available:
        print("Error: LLM not available. Ensure Ollama is running and Llama 3.2 is pulled.")
        return

    queries = [
        "Find me the best striker in Serie A",
        "Compare Haaland with Mbappe",
        "Find young hidden gems under 21"
    ]
    
    for query in queries:
        start_time = time.time()
        result = chat.parse_intent(query)
        end_time = time.time()
        print(f"Query: '{query}'")
        print(f"Response Time: {end_time - start_time:.2f}s")
        print(f"Result: {result}")
        print("-" * 20)

def benchmark_random_forest():
    print("\n--- Random Forest Performance Benchmark (Pruned) ---")
    evaluator = ModelEvaluator()
    
    # Generate synthetic data for training
    n_samples = 1000
    data = {
        'Player': [f'Player {i}' for i in range(n_samples)],
        'Age': np.random.randint(18, 35, n_samples),
        'Primary_Pos': np.random.choice(['FW', 'MF', 'DF', 'GK'], n_samples),
        'League': np.random.choice(['Premier League', 'Championship', 'League One'], n_samples),
        '90s': np.random.uniform(5, 38, n_samples),
        'Gls/90': np.random.uniform(0, 1, n_samples),
        'Ast/90': np.random.uniform(0, 1, n_samples),
        'Gls/90_pct': np.random.uniform(0, 100, n_samples),
        'Ast/90_pct': np.random.uniform(0, 100, n_samples),
        'Transfermarkt_Value_£M': np.random.uniform(1, 100, n_samples)
    }
    df = pd.DataFrame(data)
    
    start_time = time.time()
    metrics = evaluator.train_and_evaluate(df)
    end_time = time.time()
    
    print(f"Training Time: {end_time - start_time:.2f}s")
    print(f"R2 Score: {metrics.get('r2_score', 'N/A'):.4f}")
    print(f"Model Parameters: n_estimators={evaluator.model.n_estimators}, max_depth={evaluator.model.max_depth}")
    
    # Benchmark SHAP
    if evaluator.explainer:
        start_time = time.time()
        evaluator.get_shap_explanation('Player 0')
        end_time = time.time()
        print(f"SHAP Explanation Time: {end_time - start_time:.2f}s")
    
if __name__ == "__main__":
    benchmark_llm()
    benchmark_random_forest()
