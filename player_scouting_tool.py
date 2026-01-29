"""
Professional Player Scouting Tool
===================================
Analyzes football player statistics to find similar players using machine learning.

Features:
- Filters players with at least 10 '90s' for statistical reliability
- Uses 9 Per-90 statistics for similarity analysis
- Applies StandardScaler normalization
- Implements cosine similarity matching
- Generates radar chart visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class PlayerScoutingTool:
    """A tool for finding similar football players based on playing style."""
    
    def __init__(self, csv_path):
        """
        Initialize the scouting tool with player data.
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file containing player statistics
        """
        self.csv_path = csv_path
        self.df = None
        self.df_filtered = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'Gls/90', 'Ast/90', 'Sh/90', 'SoT/90', 
            'Crs/90', 'Int/90', 'TklW/90', 'Fls/90', 'Fld/90'
        ]
        self.scaled_features = None
        
    def load_and_preprocess_data(self):
        """Load data and apply preprocessing steps."""
        print("Loading data...")
        self.df = pd.read_csv(self.csv_path)
        
        # Filter players with at least 10 90s
        print(f"Total players in dataset: {len(self.df)}")
        self.df_filtered = self.df[self.df['90s'] >= 10].copy()
        print(f"Players with at least 10 90s: {len(self.df_filtered)}")
        
        # Handle missing values in defensive stats and other Per-90 columns
        for col in self.feature_columns:
            if col in self.df_filtered.columns:
                self.df_filtered[col] = self.df_filtered[col].fillna(0)
        
        # Check for any remaining missing values in feature columns
        missing_counts = self.df_filtered[self.feature_columns].isnull().sum()
        if missing_counts.any():
            print("\nWarning: Missing values found:")
            print(missing_counts[missing_counts > 0])
        
        print(f"\nLeagues in dataset: {self.df_filtered['League'].unique()}")
        
    def normalize_features(self):
        """Apply StandardScaler normalization to Per-90 features."""
        print("\nNormalizing features...")
        self.scaled_features = self.scaler.fit_transform(
            self.df_filtered[self.feature_columns]
        )
        
    def find_similar_players(self, target_player, search_league='all', top_n=5):
        """
        Find the most similar players to the target player.
        
        Parameters:
        -----------
        target_player : str
            Name of the target player
        search_league : str, optional
            League to search within ('all' for all leagues)
        top_n : int, optional
            Number of similar players to return (default: 5)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing similar players and their match scores
        """
        # Find the target player
        target_mask = self.df_filtered['Player'].str.contains(target_player, case=False, na=False)
        
        if not target_mask.any():
            print(f"Player '{target_player}' not found in dataset.")
            return None
        
        # Get the first match if multiple players found
        target_idx = self.df_filtered[target_mask].index[0]
        target_player_name = self.df_filtered.loc[target_idx, 'Player']
        
        print(f"\nTarget Player: {target_player_name}")
        print(f"Position: {self.df_filtered.loc[target_idx, 'Pos']}")
        print(f"Squad: {self.df_filtered.loc[target_idx, 'Squad']}")
        print(f"League: {self.df_filtered.loc[target_idx, 'League']}")
        print(f"90s Played: {self.df_filtered.loc[target_idx, '90s']}")
        
        # Filter by league if specified
        if search_league.lower() != 'all':
            search_df = self.df_filtered[self.df_filtered['League'] == search_league]
            search_indices = search_df.index
            search_features = self.scaled_features[self.df_filtered.index.isin(search_indices)]
        else:
            search_df = self.df_filtered
            search_indices = search_df.index
            search_features = self.scaled_features
        
        # Get target player's feature vector
        target_feature_idx = np.where(self.df_filtered.index == target_idx)[0][0]
        target_vector = self.scaled_features[target_feature_idx].reshape(1, -1)
        
        # Calculate cosine similarity
        similarities = cosine_similarity(target_vector, search_features)[0]
        
        # Create results dataframe
        results_df = search_df.copy()
        results_df['Match_Score'] = similarities * 100  # Convert to percentage
        
        # Exclude the target player and sort by similarity
        results_df = results_df[results_df.index != target_idx]
        results_df = results_df.sort_values('Match_Score', ascending=False)
        
        # Return top N results
        return results_df.head(top_n)
    
    def display_scouting_report(self, target_player, similar_players):
        """
        Display a formatted scouting report comparing players.
        
        Parameters:
        -----------
        target_player : str
            Name of the target player
        similar_players : pd.DataFrame
            DataFrame of similar players
        """
        if similar_players is None or len(similar_players) == 0:
            print("No similar players found.")
            return
        
        # Get target player info
        target_mask = self.df_filtered['Player'].str.contains(target_player, case=False, na=False)
        target_idx = self.df_filtered[target_mask].index[0]
        
        print("\n" + "="*100)
        print("SCOUTING REPORT - SIMILAR PLAYERS")
        print("="*100)
        
        # Prepare display columns (without Match_Score first)
        base_cols = ['Player', 'Pos', 'Squad', 'League'] + self.feature_columns
        
        # Create comparison dataframe
        target_row = self.df_filtered.loc[target_idx:target_idx, base_cols].copy()
        target_row.insert(4, 'Match_Score', 100.0)  # Insert Match_Score after League
        
        # Get similar players data
        similar_data = similar_players[base_cols].copy()
        similar_data.insert(4, 'Match_Score', similar_players['Match_Score'].values)
        
        comparison_df = pd.concat([target_row, similar_data])
        
        # Format the output
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 20)
        
        print(comparison_df.to_string(index=False))
        print("="*100)
        
    def create_radar_chart(self, target_player, similar_players, save_path='player_comparison_radar.png'):
        """
        Create a radar chart comparing the target player with the top match.
        
        Parameters:
        -----------
        target_player : str
            Name of the target player
        similar_players : pd.DataFrame
            DataFrame of similar players
        save_path : str, optional
            Path to save the radar chart image
        """
        if similar_players is None or len(similar_players) == 0:
            print("Cannot create radar chart: No similar players found.")
            return
        
        # Get target player and top match
        target_mask = self.df_filtered['Player'].str.contains(target_player, case=False, na=False)
        target_idx = self.df_filtered[target_mask].index[0]
        target_data = self.df_filtered.loc[target_idx, self.feature_columns].values
        
        top_match_idx = similar_players.index[0]
        top_match_data = self.df_filtered.loc[top_match_idx, self.feature_columns].values
        
        # Get player names
        target_name = self.df_filtered.loc[target_idx, 'Player']
        top_match_name = self.df_filtered.loc[top_match_idx, 'Player']
        match_score = similar_players.iloc[0]['Match_Score']
        
        # Setup radar chart
        categories = ['Goals', 'Assists', 'Shots', 'SoT', 'Crosses', 'Int', 'Tackles', 'Fouls', 'Fouled']
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Append first value to close the plot
        target_data = np.concatenate((target_data, [target_data[0]]))
        top_match_data = np.concatenate((top_match_data, [top_match_data[0]]))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Plot data
        ax.plot(angles, target_data, 'o-', linewidth=2, label=target_name, color='#1f77b4')
        ax.fill(angles, target_data, alpha=0.25, color='#1f77b4')
        
        ax.plot(angles, top_match_data, 'o-', linewidth=2, label=f'{top_match_name} ({match_score:.1f}%)', color='#ff7f0e')
        ax.fill(angles, top_match_data, alpha=0.25, color='#ff7f0e')
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12)
        
        # Add title and legend
        plt.title(f'Player Comparison: Playing Style Overlap\n{target_name} vs {top_match_name}', 
                  size=16, weight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nRadar chart saved to: {save_path}")
        plt.close()


def main():
    """Main function to demonstrate the scouting tool."""
    
    # Initialize the tool
    tool = PlayerScoutingTool('english_football_pyramid_master.csv')
    
    # Load and preprocess data
    tool.load_and_preprocess_data()
    
    # Normalize features
    tool.normalize_features()
    
    # Example 1: Find similar players to a Premier League striker
    print("\n" + "="*100)
    print("EXAMPLE 1: Finding similar players to Erling Haaland")
    print("="*100)
    
    similar = tool.find_similar_players('Haaland', search_league='all', top_n=5)
    tool.display_scouting_report('Haaland', similar)
    tool.create_radar_chart('Haaland', similar, 'haaland_comparison.png')
    
    # Example 2: Search within a specific league
    print("\n\n" + "="*100)
    print("EXAMPLE 2: Finding similar players to Chris Wood (within Premier League only)")
    print("="*100)
    
    similar_pl = tool.find_similar_players('Chris Wood', search_league='Premier League', top_n=5)
    tool.display_scouting_report('Chris Wood', similar_pl)
    
    print("\n\nScouting tool demonstration complete!")
    print("You can now use the PlayerScoutingTool class to analyze any player in the dataset.")


if __name__ == "__main__":
    main()
