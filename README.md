# ⚽ Football Scouting Dashboard

A professional-grade football scouting platform built with **Streamlit**, **scikit-learn**, and **Plotly**. This dashboard provides ML-powered player analysis, similarity matching, and automated scouting reports for the English Football Pyramid (Premier League → National League).

---

## 🎯 Key Features

### 1. **ML-Powered Player Clustering**
- **K-Means Clustering** (k=8) assigns players to tactical archetypes
- **PCA Dimensionality Reduction** with 67.4% variance explained
- 8 distinct player archetypes: Target Man, Creative Playmaker, Ball-Winning Midfielder, etc.

### 2. **Advanced Similarity Engine**
- **Cosine similarity** with position-specific weighting
- Fuzzy player search with typo tolerance (rapidfuzz)
- **Explainable AI**: Shows which features drive similarity matches
- Top-N similar players with detailed breakdowns

### 3. **Archetype Universe Visualization** 🌌
- Interactive 2D PCA scatter plot showing all players
- Identify **Tactical Hybrids**: players between archetype clusters
- Filter by archetype to explore player distributions
- Hover to see detailed player stats

### 4. **Automated Scouting Reports**
- **LLM-Powered Narratives**: Google Gemini AI for context-aware scouting summaries
- **Rule-Based Fallback**: Template-based generation if AI unavailable
- **PDF Export**: One-page dossiers for recruitment meetings
- Market value estimation with confidence tiers
- Position-specific percentile rankings

### 5. **Age-Curve Anomaly Detection** 🌟
- **High-Ceiling Prospect Identification**: Z-score analysis (>2σ above age cohort)
- Automatic badges for elite prospects (3σ), high-ceiling (2.5σ), ahead of curve (2σ)
- Position-specific age-curve analysis
- Age cohort comparison visualizations
- Percentile rank within same-age players

### 6. **ML-Based Transfer Value Prediction** 📊
- **Random Forest Model**: Trained on real Transfermarkt data
- Fair value vs market premium analysis
- Undervalued bargain detection
- Feature engineering (age decay, percentile boosts, league tiers)
- Model retraining pipeline with web scraping

### 7. **National League Data Handling**
- Special "Limited Data Tier" badge for National League players
- Avoids bias from capped completeness scores (max 33%)
- Clear disclaimers for data limitations

### 8. **Multi-Page Dashboard**
- 🔍 **Player Search**: Fuzzy search + similar players with similarity drivers
- ⚔️ **Head-to-Head**: Radar chart comparisons
- 💎 **Hidden Gems**: Find undervalued young talent
- 🏆 **Leaderboards**: Metric rankings + Archetype Universe map + Distribution analysis

---

## 📊 Technical Architecture

### Data Pipeline
```
Raw CSV (FBref) 
  → Data Cleaning (missing values, outliers)
  → Position-Specific Percentile Normalization
  → StandardScaler Normalization
  → K-Means Clustering (k=8)
  → PCA Projection (2D)
  → Market Value Estimation
```

### ML Components

#### 1. **Clustering Module** (`utils/clustering.py`)
- **Algorithm**: K-Means with k=8, random_state=42
- **Features**: 9 per-90 metrics (Gls/90, Ast/90, Sh/90, etc.)
- **Archetype Assignment**: Centroid analysis → semantic labels
- **PCA**: 2 components, 67.4% variance explained
- **Output**: `Archetype`, `Cluster`, `PCA_X`, `PCA_Y`, `Archetype_Confidence`

#### 2. **Similarity Engine** (`utils/similarity.py`)
- **Algorithm**: Cosine similarity on StandardScaler-normalized features
- **Position Weighting**: 
  - Attackers: Gls/90 (3.0x), Sh/90 (2.5x), SoT/90 (2.0x)
  - Midfielders: Ast/90 (2.5x), Crs/90 (2.0x), TklW/90 (1.8x)
  - Defenders: Int/90 (2.5x), TklW/90 (2.5x), Fld/90 (1.5x)
- **Fuzzy Search**: rapidfuzz with 80% threshold
- **Feature Attribution**: Calculates which metrics drive similarity

#### 3. **Market Value Estimation** (`utils/market_value.py`)
- **Inputs**: Age, percentile ranks, league tier, archetype
- **Output**: Estimated value (£M), value tier, value score
- **Bias Modes**: Conservative, Neutral, Aggressive

#### 4. **Narrative Generation** (`utils/narrative_generator.py`)
- Template-based NLG with dynamic content
- Position-specific insights
- Archetype descriptions
- Market value context

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup
```bash
# Clone repository
git clone <repo-url>
cd scouting-dashboard

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
```

### Dependencies
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
plotly>=5.17.0
rapidfuzz>=3.0.0
fpdf>=1.7.2
pyyaml>=6.0
```

---

## 📁 Project Structure

```
scouting-dashboard/
├── app.py                          # Main Streamlit app
├── config.yaml                     # Configuration (leagues, positions, etc.)
├── english_football_pyramid_master.csv  # Main dataset
├── utils/
│   ├── __init__.py
│   ├── clustering.py               # K-Means + PCA
│   ├── similarity.py               # Cosine similarity engine
│   ├── data_engine.py              # Data loading + preprocessing
│   ├── market_value.py             # Value estimation
│   ├── narrative_generator.py     # NLG for scouting reports
│   ├── visualizations.py          # Plotly charts
│   ├── pdf_export.py              # PDF dossier generation
│   ├── constants.py               # Feature columns, archetypes
│   └── config_loader.py           # YAML config loader
├── data/                          # Raw league CSVs
└── manual_data/                   # Manually downloaded data
```

---

## 🎨 UI/UX Features

### Session State Management
- Persistent filters across pages (age, league, position, minutes)
- Cached data loading for performance
- Scout bias settings (market value multiplier)

### Interactive Visualizations
- **Beeswarm plots**: Player distribution by metric
- **Radar charts**: Head-to-head comparisons
- **Archetype Universe**: PCA scatter plot with archetype colors
- **Similarity drivers**: Horizontal bar charts showing feature attribution
- **Percentile bars**: Progress bars with color coding (🟢🟡🔴)

### Professional Design
- Dark theme (`plotly_dark`)
- Color-coded leagues (Premier League: #3D195B, Championship: #0E4C92, etc.)
- Archetype-specific colors (Target Man: #E74C3C, Creative Playmaker: #3498DB, etc.)
- Responsive layout with Streamlit columns

---

## 📈 Data Quality

### Completeness Scoring
- **Formula**: `(non-null metrics / total metrics) * 100`
- **Tiers**:
  - 🟢 **90%+**: Verified Elite Data
  - 🟡 **70-89%**: Good Scouting Data
  - 🟠 **40-69%**: Directional Data (caution advised)
  - 🔴 **<40%**: Incomplete Data
  - 🔵 **National League**: Limited Data Tier (capped at 33%)

### Percentile Normalization
- **Position-specific**: Separate percentiles for FW, MF, DF, GK
- **League-aware**: Percentiles calculated within league context
- **Minimum sample**: 10 full matches (90s) for reliability

---

## 🚀 Usage Examples

### 1. Find Similar Players
```python
# In Player Search page:
1. Type player name (e.g., "Erling Haaland")
2. Select from fuzzy-matched suggestions
3. View top 5 similar players with similarity drivers
4. Expand to see feature-by-feature breakdown
```

### 2. Identify Tactical Hybrids
```python
# In Leaderboards page:
1. Scroll to "Archetype Universe" section
2. Look for players positioned between archetype clusters
3. Filter by specific archetypes to isolate hybrids
4. Hover to see player details
```

### 3. Export Scouting Dossier
```python
# In Player Search page:
1. Search for player
2. Scroll to "Export Scouting Dossier"
3. Click "Download PDF Scouting Dossier"
4. PDF includes: player info, key stats, narrative, market value
```

### 4. Discover Hidden Gems
```python
# In Hidden Gems page:
1. Set filters: Min Goals/90, Min Assists/90, Max Age
2. Exclude Premier League for undervalued players
3. Sort by percentile or value score
4. Export CSV for further analysis
```

---

## 🔬 ML Rigor Summary

| Component | Algorithm | Parameters | Validation |
|-----------|-----------|------------|------------|
| **Clustering** | K-Means | k=8, n_init=10, random_state=42 | Silhouette score, cluster cohesion |
| **Dimensionality Reduction** | PCA | n_components=2 | 67.4% variance explained |
| **Similarity** | Cosine Similarity | Position-weighted features | Manual validation vs. expert opinions |
| **Normalization** | StandardScaler | Per-position, per-league | Z-score distribution checks |
| **Market Value** | Heuristic Model | Age decay, percentile boost | Compared to Transfermarkt estimates |

---

## 📝 Business Logic

### Similarity Drivers (Explainable AI)
When showing a similar player, the dashboard calculates **feature attribution**:
- Computes absolute distance for each feature (9 metrics)
- Applies position-specific weighting
- Normalizes to 0-1 scale
- Displays top 2 features as "Primary Drivers"
- Example: *"96% Match - Driven by high Gls/90, Sh/90 overlap"*

### National League Disclaimer
- National League players have limited FBref coverage
- Completeness capped at 33% (by design, not player quality)
- UI shows **🔵 Limited Data Tier** badge instead of 🔴 red warning
- Prevents bias against lower-league talent

### PDF Dossier
- One-page format for recruitment meetings
- Includes: Player info, key stats, percentile ranks, narrative, market value
- Generated on-the-fly with `fpdf`
- Filename: `{Player_Name}_scouting_report.pdf`

---

## 🎯 Roadmap

- [ ] Add goalkeeper-specific clustering (separate k=8 for GK)
- [ ] Integrate Transfermarkt API for real market values
- [ ] Add injury history data
- [ ] Multi-season trend analysis
- [ ] Export to Excel with conditional formatting
- [ ] Add "Watch List" feature with session persistence
- [ ] Integrate video highlights (YouTube API)

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

- **Data Source**: FBref (Sports Reference)
- **Frameworks**: Streamlit, scikit-learn, Plotly
- **Inspiration**: StatsBomb, Wyscout, Opta

---

## 📧 Contact

For questions or collaboration:
- GitHub: [Your GitHub]
- Email: [Your Email]

---

**Built with ❤️ for football analytics enthusiasts**
