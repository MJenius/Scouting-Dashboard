
import pandas as pd
import numpy as np
import os
import sys

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import recruitment_logic, data_engine

def empirical_backtest(current_df, historical_df):
    """
    Compare actual performance delta vs predicted delta for players who moved up.
    """
    print("\nSTARTING EMPIRICAL BACKTEST (Historical Data Found)")
    
    # 1. Identify movers (Heuristic: Same Name+Year, diff League)
    # Since we have separate files, we match strictly on Name + Birthday (if avail) or just Name.
    # Simplifying to Name for this exercise.
    
    # TODO: Implement strict matching if real historical data provided.
    pass

def theoretical_consistency_check(df):
    """
    Verify internal consistency of the projection logic.
    Check: Does Projecting Down then Up return the original value?
    """
    print("\nSTARTING THEORETICAL CONSISTENCY CHECK (No Historical Data)")
    print("Rationale: Verifying that Step-Up logic is mathematically consistent across tiers.")
    
    # Filter for PL players
    pl_players = df[df['League'] == 'Premier League'].copy()
    
    if len(pl_players) == 0:
        print("No Premier League players found for consistency check.")
        return

    # Select numeric columns
    stat_cols = [c for c in pl_players.columns if c in data_engine.FEATURE_COLUMNS]
    
    results = {}
    
    # Test for each position group
    for pos_group, poses in {'FW': ['FW'], 'MF': ['MF'], 'DF': ['DF']}.items():
        group = pl_players[pl_players['Primary_Pos'].isin(poses)]
        
        if len(group) == 0:
            continue
            
        print(f"\nAnalyzing {pos_group} Group ({len(group)} players)...")
        
        errors = []
        
        for idx, row in group.iterrows():
            # 1. Simulate Move DOWN to Championship
            # To go PL -> Champ, we divide by the PL factor and multiply by Champ factor?
            # Or simply: Project(PL, Target='Championship')
            
            # Logic: PL (1.0) -> Champ (0.7)
            # Factor = 1.0 / 0.7 = 1.42 (Premiums apply going down)
            
            # Let's use the function directly
            # We create a single-row DF
            p_df = pd.DataFrame([row])
            
            # Project DOWN (Simulating them in Championship)
            down_df = recruitment_logic.project_to_tier(p_df, 'Championship')
            
            # Extract Projected Stats
            simulated_champ_stats = {}
            for col in stat_cols:
                if f'Projected_{col}' in down_df.columns:
                    simulated_champ_stats[col] = down_df.iloc[0][f'Projected_{col}']
                    
            # 2. Project UP (Simulating return to PL)
            # Create a DF mimicking the championship stats
            champ_row = row.copy()
            for col, val in simulated_champ_stats.items():
                champ_row[col] = val
            champ_row['League'] = 'Championship'
            
            p_back_df = pd.DataFrame([champ_row])
            
            # Project UP
            up_df = recruitment_logic.project_to_tier(p_back_df, 'Premier League')
            
            # 3. Calculate Error (Original vs Round-Trip)
            # Should be close to 0 if logic is reversible
            # Current Logic:
            # Down: val * (1.0 / 0.7) -> 1.42x
            # Up:   val * (0.7 / 1.0) -> 0.7x
            # 1.42 * 0.7 = 0.994... (Close to 1.0)
            
            for col in stat_cols:
                if f'Projected_{col}' in up_df.columns:
                    final_val = up_df.iloc[0][f'Projected_{col}']
                    original_val = row[col]
                    
                    if original_val > 0.1: # Avoid div/0 or tiny noise
                        error = (final_val - original_val) / original_val
                        errors.append(error)
        
        if errors:
            mbe = np.mean(errors) * 100
            mae = np.mean(np.abs(errors)) * 100
            print(f"  > MBE (Bias): {mbe:+.2f}%")
            print(f"  > MAE (Error): {mae:.2f}%")
            results[pos_group] = mbe
            
    return results

def main():
    try:
        # Load current data
        print("Loading data...")
        data_res = data_engine.process_all_data('english_football_pyramid_master.csv', min_90s=5.0)
        df_current = data_res['dataframe']
        
        # Check for history
        history_path = 'data/history/season_23_24.csv'
        if os.path.exists(history_path):
            df_history = data_engine.load_data(history_path)
            empirical_backtest(df_current, df_history)
        else:
            print(f"No historical data found at {history_path}.")
            results = theoretical_consistency_check(df_current)
            
            print("\nrecommendation:")
            if results:
                for pos, mbe in results.items():
                    if abs(mbe) > 5.0:
                        print(f"{pos} projections have high drift ({mbe:+.1f}%). Check coefficients.")
                    else:
                        print(f"{pos} projections are mathematically consistent.")
                        
    except Exception as e:
        print(f"Error: {e}")
            
if __name__ == "__main__":
    main()
