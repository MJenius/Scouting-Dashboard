# 🚶‍♂️ Site Walkthrough & Feature Guide

Welcome to the **Football Scouting & Recruitment Dashboard**. This guide explores the key features and workflows available in the application.

---

## 1. ⚙️ Global Filters & Sidebar
Located on the left sidebar, these settings persist across the entire application:
*   **Age Range:** Filter players between 16 and 40 years old.
*   **Leagues:** Select specific competitions (e.g., Premier League, La Liga, Championship) or view all.
*   **Positions:** Focus on Forwards (FW), Midfielders (MF), or Defenders (DF). Goalkeepers (GK) are handled separately.
*   **Minimum Minutes:** Filter out players with insufficient playing time (default > 0.5 90s) to ensure data reliability.

---

## 2. 🔍 Player Search
*The homepage for deep-dive individual analysis.*

### Key Features:
*   **Fuzzy Search:** Type a name (e.g., "Rodri") to find players. Handles typos and partial matches.
*   **Player Card:** Displays core bio-data (Age, Team, Position) and an **Archetype** (e.g., "Deep-Lying Playmaker") assigned by the unsupervised learning model.
*   **Data Confidence:** A "Scouting Confidence" score indicates how complete the statistical profile is (Green = Verified Elite, Red = Incomplete).
*   **📝 Scout's Take (AI):** Click **"🤖 Use AI"** to generate a prose-style scouting report using Gemini 1.5 Pro. It analyzes outliers and style to write a human-like summary.
*   **👥 Similarity Engine:** Finds the "Nearest Neighbors" to the selected player.
    *   **"Driven By"**: Explains *why* players are similar (e.g., "Driven by: Progressive Passes, Tackles").
    *   **Visual Logic:** A bar chart shows the exact similarity % of specific attributes.
*   **📥 PDF Dossier:** Download a one-page PDF report containing the bio, narrative, and key comparisons for offline meetings.

---

## 3. ⚔️ Head-to-Head
*Compare two players side-by-side.*

### Key Features:
*   **Direct Comparison:** Select Player A and Player B.
*   **Match Score:** A calculated percentage showing how statistically identical they are.
*   **📊 Radar Charts:** Overlays both players' percentile ranks on a spider chart. Toggle "Relative Quality" to compare them against their specific positional peers.
*   **Feature Diff:** A table showing the raw difference in key metrics (e.g., "+0.45 xG/90" for Player A).

---

## 4. 💎 Hidden Gems
*Find undervalued talent.*

### Key Features:
*   **Moneyball Filters:** Search for players performing above expectations:
    *   **Finishing Efficiency:** Players exceeding their xG (Clinical Finishers).
    *   **xA Overperformance:** Playmakers creating more than their raw stats suggest.
*   **Exclude Premier League:** A checkbox to intentionally look for talent in lower tiers or other leagues.
*   **Results Table:** Exports a CSV list of targets meeting your criteria.

---

## 5. 🏆 Leaderboards & Tactical Map
*Macro-level analysis of the football landscape.*

### Key Features:
*   **Metric Rankings:** View top 25 players for any specific stat (e.g., Interceptions/90, xGChain).
*   **📈 Distribution Analysis:**
    *   **Histogram:** See where a player falls in the distribution curve.
    *   **League Comparison:** Box plots comparing the physical intensity or technical quality of different leagues.
*   **🌌 Tactical Galaxy (PCA):**
    *   A 2D scatter plot representing the entire database.
    *   **Clusters:** See how "Target Men" naturally group away from "False 9s".
    *   **Hybrids:** Identify players in the "white space" between clusters (e.g., Fullbacks who act like Midfielders).
