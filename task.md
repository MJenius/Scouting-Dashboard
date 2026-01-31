# Task: Refine Scouting Dashboard Data & Engine

## 1. Data Integrity (`utils/data_engine.py`)
- [ ] **Normalize National League (NL) Data**: 
    - In `clean_feature_columns`, detect missing values (especially in lower leagues).
    - Fill missing values with the **Global Position Average** (or League Position Average if available/non-zero) instead of 0.0. This prevents "Limited Data" players from being penalized as "0 performance" players in similarity/scaling.
- [ ] **Goalkeeper Metric Contamination**: 
    - Ensure `calculate_position_percentiles` and cleaning steps explicitly mask outfield metrics for GKs (set to NaN or specific sentinels) so they don't influence global scaling or accidentally get percentiled.
- [ ] **Robust CSV Parsing**:
    - Update `load_data` to detect delimiters (`;` vs `,`) automatically or try/except to handle user uploads with different formats.

## 2. Similarity Engine (`utils/similarity.py`)
- [ ] **Driver Refinement ("Shared Excellence")**:
    - Modify `find_similar_players` and `calculate_feature_attribution`.
    - Change logic for "Primary Drivers" string. Instead of just "lowest distance", prioritize features where **both** players have high percentiles (e.g., > 70th) AND distance is low.
    - Logic: Score = `(Player1_Pct + Player2_Pct) - (Distance_Penalty)`.
- [ ] **Weakness Correlation**:
    - Adjust similarity calculation or weighting to reduce the penalty for mismatches in "weak" stats (stats where target player < 50th percentile).
- [ ] **Fuzzy Search Precision**:
    - Lower `FUZZY_MATCH_THRESHOLD` (e.g., to 75 or 80) or improve rapidfuzz scorer (e.g., `token_sort_ratio` vs `WRatio`) to handle accents/names better.

## 3. Clustering & ML (`utils/clustering.py`)
- [ ] **Dynamic k**:
    - Implement a method to determine optimal `k` (e.g., Silhouette Analysis) instead of hardcoded `k=8`.
    - Check `k` range (e.g., 6 to 12).
- [ ] **PCA Validation**:
    - Check explained variance. If low (< 70%), consider increasing features or noting it. (User suggests re-verifying).

## 4. UI/UX (`app.py`, `utils/narrative_generator.py`)
- [ ] **Narrative Repetitiveness**:
    - Add variation to `utils/narrative_generator.py` templates.
- [ ] **Tactical Galaxy Search**:
    - (Low priority for this turn, focus on engine first).

## 5. Head-to-Head Improvements
- [x] **Smart Summary**: Add text-based summary comparing players overall + specific attributes before the chart.

## Checklist from User
- [ ] Normalize NL
- [ ] Dynamic k
- [ ] Driver Refinement
