"""
transfermarkt_scraper.py - Web scraper for Transfermarkt player valuations.

This module provides:
- Player name matching and search
- Market value extraction from Transfermarkt
- Batch processing for multiple players
- Caching to avoid repeated requests
- Rate limiting to respect website policies
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from typing import Optional, Dict, List, Tuple
import re
from urllib.parse import quote
import json
import os


class TransfermarktScraper:
    """
    Scrape player market values from Transfermarkt.
    
    Features:
    - Fuzzy player name matching
    - Market value extraction
    - Batch processing with rate limiting
    - Local caching to reduce requests
    - Error handling and retries
    """
    
    BASE_URL = "https://www.transfermarkt.com"
    SEARCH_URL = f"{BASE_URL}/schnellsuche/ergebnis/schnellsuche"
    
    # Rate limiting
    REQUEST_DELAY = 2.0  # Seconds between requests
    MAX_RETRIES = 3
    
    # User agent to avoid blocking
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    def __init__(self, cache_file: str = 'transfermarkt_cache.json'):
        """
        Initialize scraper.
        
        Args:
            cache_file: Path to cache file for storing results
        """
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.last_request_time = 0
    
    def _load_cache(self) -> Dict:
        """Load cached results from file."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠ Could not load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠ Could not save cache: {e}")
    
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.REQUEST_DELAY:
            time.sleep(self.REQUEST_DELAY - elapsed)
        self.last_request_time = time.time()
    
    def _parse_market_value(self, value_str: str) -> Optional[float]:
        """
        Parse market value string to float (in millions).
        
        Args:
            value_str: Value string (e.g., "€5.00m", "€500k")
            
        Returns:
            Value in millions (£) or None if parsing fails
        """
        if not value_str or value_str == '-':
            return None
        
        try:
            # Remove currency symbols and whitespace
            value_str = value_str.replace('€', '').replace('£', '').replace('$', '').strip()
            
            # Handle millions
            if 'm' in value_str.lower():
                value = float(value_str.lower().replace('m', ''))
                return value
            
            # Handle thousands
            elif 'k' in value_str.lower():
                value = float(value_str.lower().replace('k', ''))
                return value / 1000.0
            
            # Handle billions (rare)
            elif 'bn' in value_str.lower():
                value = float(value_str.lower().replace('bn', ''))
                return value * 1000.0
            
            else:
                # Try direct conversion
                return float(value_str)
        
        except Exception as e:
            print(f"⚠ Could not parse value '{value_str}': {e}")
            return None
    
    def search_player(
        self,
        player_name: str,
        retry_count: int = 0,
    ) -> Optional[Dict]:
        """
        Search for player on Transfermarkt and get market value.
        
        Args:
            player_name: Player name to search
            retry_count: Current retry attempt
            
        Returns:
            Dict with player info and market value, or None if not found
        """
        # Check cache first
        cache_key = player_name.lower().strip()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Rate limit
        self._rate_limit()
        
        try:
            # Search for player
            search_params = {
                'query': player_name
            }
            
            response = requests.get(
                self.SEARCH_URL,
                params=search_params,
                headers=self.HEADERS,
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"⚠ Search failed for '{player_name}': HTTP {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find player results (usually in a table)
            player_rows = soup.find_all('tr', class_=['odd', 'even'])
            
            if not player_rows:
                print(f"⚠ No results found for '{player_name}'")
                self.cache[cache_key] = None
                self._save_cache()
                return None
            
            # Get first result (best match)
            first_row = player_rows[0]
            
            # Extract player info
            player_link = first_row.find('a', class_='spielprofil_tooltip')
            if not player_link:
                return None
            
            player_url = self.BASE_URL + player_link['href']
            found_name = player_link.text.strip()
            
            # Extract market value from search results
            value_cell = first_row.find('td', class_='rechts hauptlink')
            if value_cell:
                value_str = value_cell.text.strip()
                market_value = self._parse_market_value(value_str)
            else:
                # If not in search results, visit player page
                market_value = self._get_value_from_profile(player_url)
            
            result = {
                'player_name': found_name,
                'market_value_m': market_value,
                'url': player_url,
                'search_query': player_name,
            }
            
            # Cache result
            self.cache[cache_key] = result
            self._save_cache()
            
            return result
        
        except Exception as e:
            print(f"⚠ Error searching for '{player_name}': {e}")
            
            # Retry logic
            if retry_count < self.MAX_RETRIES:
                time.sleep(2 ** retry_count)  # Exponential backoff
                return self.search_player(player_name, retry_count + 1)
            
            return None
    
    def _get_value_from_profile(self, profile_url: str) -> Optional[float]:
        """
        Get market value from player profile page.
        
        Args:
            profile_url: URL to player profile
            
        Returns:
            Market value in millions or None
        """
        self._rate_limit()
        
        try:
            response = requests.get(profile_url, headers=self.HEADERS, timeout=10)
            
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find market value element
            value_elem = soup.find('a', class_='data-header__market-value-wrapper')
            if value_elem:
                value_str = value_elem.text.strip()
                return self._parse_market_value(value_str)
            
            return None
        
        except Exception as e:
            print(f"⚠ Error fetching profile: {e}")
            return None
    
    def batch_search(
        self,
        player_names: List[str],
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Search for multiple players in batch.
        
        Args:
            player_names: List of player names
            progress_callback: Optional callback function(current, total)
            
        Returns:
            DataFrame with columns: player_name, market_value_m, url
        """
        results = []
        total = len(player_names)
        
        for i, name in enumerate(player_names, 1):
            if progress_callback:
                progress_callback(i, total)
            
            result = self.search_player(name)
            
            if result:
                results.append(result)
            else:
                # Add placeholder for failed searches
                results.append({
                    'player_name': name,
                    'market_value_m': None,
                    'url': None,
                    'search_query': name,
                })
            
            # Progress update
            if i % 10 == 0:
                print(f"Processed {i}/{total} players...")
        
        return pd.DataFrame(results)


def enrich_with_transfermarkt_values(
    df: pd.DataFrame,
    player_column: str = 'Player',
    cache_file: str = 'transfermarkt_cache.json',
) -> pd.DataFrame:
    """
    Enrich DataFrame with Transfermarkt market values.
    
    Args:
        df: DataFrame with player names
        player_column: Name of column containing player names
        cache_file: Path to cache file
        
    Returns:
        DataFrame with added 'Transfermarkt_Value_£M' column
    """
    scraper = TransfermarktScraper(cache_file=cache_file)
    
    # Get unique player names
    unique_players = df[player_column].unique()
    
    print(f"Fetching Transfermarkt values for {len(unique_players)} players...")
    
    # Batch search
    results_df = scraper.batch_search(unique_players)
    
    # Merge back to original DataFrame
    results_df = results_df.rename(columns={
        'player_name': 'TM_Name',
        'market_value_m': 'Transfermarkt_Value_£M',
    })
    
    # Create mapping
    value_map = dict(zip(
        results_df['search_query'],
        results_df['Transfermarkt_Value_£M']
    ))
    
    # Add to DataFrame
    df['Transfermarkt_Value_£M'] = df[player_column].map(value_map)
    
    # Report success rate
    found = df['Transfermarkt_Value_£M'].notna().sum()
    print(f"✓ Found values for {found}/{len(df)} players ({found/len(df)*100:.1f}%)")
    
    return df
