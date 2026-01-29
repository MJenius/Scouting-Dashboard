"""
train_ml_model.py - Script to train ML-based transfer value model.

This script:
1. Loads player data
2. Fetches Transfermarkt values (with caching)
3. Trains Random Forest model
4. Saves model for use in dashboard
5. Generates evaluation report
"""

import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_engine import load_data, process_all_data
from utils.transfermarkt_scraper import enrich_with_transfermarkt_values
from utils.ml_value_model import MLValuePredictor, train_and_save_model


def main():
    """Main training pipeline."""
    
    print("=" * 60)
    print("ML Transfer Value Model Training Pipeline")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1/5] Loading player data...")
    try:
        df = pd.read_csv('english_football_pyramid_master.csv')
        print(f"✓ Loaded {len(df)} players")
    except FileNotFoundError:
        print("✗ Master CSV not found. Processing raw data...")
        df = process_all_data()
        df.to_csv('english_football_pyramid_master.csv', index=False)
        print(f"✓ Processed and saved {len(df)} players")
    
    # Step 2: Filter to relevant players
    print("\n[2/5] Filtering to relevant players...")
    # Focus on players with sufficient data and reasonable age
    df_filtered = df[
        (df['Age'] >= 17) &
        (df['Age'] <= 35) &
        (df['90s'] >= 10) &  # At least 10 full matches
        (df['Completeness_Score'] >= 30)  # Reasonable data quality
    ].copy()
    print(f"✓ Filtered to {len(df_filtered)} players")
    
    # Step 3: Fetch Transfermarkt values
    print("\n[3/5] Fetching Transfermarkt values...")
    print("⚠ This may take 10-30 minutes for ~2000 players")
    print("⚠ Results are cached - subsequent runs will be much faster")
    
    # Sample for testing (remove this for full dataset)
    # df_sample = df_filtered.sample(n=min(100, len(df_filtered)), random_state=42)
    # df_enriched = enrich_with_transfermarkt_values(df_sample)
    
    # Full dataset (uncomment for production)
    df_enriched = enrich_with_transfermarkt_values(df_filtered)
    
    # Save enriched data
    df_enriched.to_csv('players_with_transfermarkt_values.csv', index=False)
    print(f"✓ Saved enriched data to 'players_with_transfermarkt_values.csv'")
    
    # Step 4: Train model
    print("\n[4/5] Training ML model...")
    
    # Check how many players have values
    n_with_values = df_enriched['Transfermarkt_Value_£M'].notna().sum()
    print(f"Training on {n_with_values} players with Transfermarkt values")
    
    if n_with_values < 50:
        print("✗ Insufficient training data. Need at least 50 players with values.")
        print("  Try:")
        print("  1. Running on full dataset (not sample)")
        print("  2. Checking Transfermarkt scraper is working")
        print("  3. Manually adding some values to CSV")
        return
    
    try:
        predictor = train_and_save_model(
            df_enriched,
            output_path='ml_value_model.pkl'
        )
    except Exception as e:
        print(f"✗ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 5: Generate evaluation report
    print("\n[5/5] Generating evaluation report...")
    
    # Analyze value premium for all players
    df_analysis = predictor.analyze_value_premium(df_enriched)
    
    # Find undervalued bargains
    bargains = predictor.find_undervalued_bargains(
        df_analysis,
        min_discount=20.0,
        max_age=26,
        top_n=20
    )
    
    print("\n" + "=" * 60)
    print("Top 20 Undervalued Bargains")
    print("=" * 60)
    
    if len(bargains) > 0:
        display_cols = [
            'Player', 'Age', 'Primary_Pos', 'League', 'Squad',
            'Transfermarkt_Value_£M', 'Predicted_Value_£M', 'Value_Premium_%'
        ]
        display_cols = [col for col in display_cols if col in bargains.columns]
        
        print(bargains[display_cols].to_string(index=False))
        
        # Save bargains report
        bargains.to_csv('undervalued_bargains.csv', index=False)
        print(f"\n✓ Saved bargains report to 'undervalued_bargains.csv'")
    else:
        print("No undervalued bargains found with current criteria")
    
    # Feature importance
    print("\n" + "=" * 60)
    print("Top 10 Most Important Features")
    print("=" * 60)
    
    for i, (feat, imp) in enumerate(list(predictor.feature_importance.items())[:10], 1):
        print(f"{i:2d}. {feat:30s} {imp:.4f}")
    
    print("\n" + "=" * 60)
    print("✓ Training Complete!")
    print("=" * 60)
    print("\nModel saved to: ml_value_model.pkl")
    print("Cache saved to: transfermarkt_cache.json")
    print("\nYou can now use the ML model in your dashboard!")
    print("Set USE_ML_VALUE_MODEL = True in app.py")


if __name__ == '__main__':
    main()
