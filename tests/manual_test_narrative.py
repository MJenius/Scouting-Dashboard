
import pandas as pd
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.narrative_generator import generate_comparison_narrative

def test_comparison():
    # Mock data for Player 1 (Elite Forward)
    p1 = pd.Series({
        'Player': 'Erling Haaland',
        'Primary_Pos': 'FW',
        'Archetype': 'Goalscoring Forward',
        'Gls/90': 1.2,
        'Gls/90_pct': 99.0,
        'Ast/90': 0.1,
        'Ast/90_pct': 40.0,
        'Sh/90_pct': 95.0,
        'SoT/90_pct': 96.0,
        'Fld/90_pct': 50.0,
    })

    # Mock data for Player 2 (Creative Forward)
    p2 = pd.Series({
        'Player': 'Harry Kane',
        'Primary_Pos': 'FW',
        'Archetype': 'Complete Forward',
        'Gls/90': 0.8,
        'Gls/90_pct': 85.0,
        'Ast/90': 0.5,
        'Ast/90_pct': 95.0,
        'Sh/90_pct': 80.0,
        'SoT/90_pct': 85.0,
        'Fld/90_pct': 60.0,
    })

    print("--- Testing Comparison Narrative ---")
    narrative = generate_comparison_narrative(p1, p2)
    print(narrative)
    
    with open("test_output.txt", "w", encoding="utf-8") as f:
        f.write("--- Testing Comparison Narrative ---\n")
        f.write(narrative)
        f.write("\n\n")

    print("------------------------------------")

    # Test 2: Similar Players
    p3 = pd.Series({
        'Player': 'Ollie Watkins',
        'Primary_Pos': 'FW',
        'Archetype': 'Goalscoring Forward',
        'Gls/90_pct': 90.0,
        'Ast/90_pct': 50.0,
        'Sh/90_pct': 90.0,
        'SoT/90_pct': 90.0,
        'Fld/90_pct': 55.0,
    })
    
    print("\n--- Testing Similar Narrative ---")
    narrative_sim = generate_comparison_narrative(p1, p3)
    print(narrative_sim)
    
    with open("test_output.txt", "a", encoding="utf-8") as f:
        f.write("--- Testing Similar Narrative ---\n")
        f.write(narrative_sim)
        f.write("\n")
    print("------------------------------------")

if __name__ == "__main__":
    test_comparison()
