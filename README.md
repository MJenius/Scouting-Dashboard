# Football Scouting & Recruitment Dashboard

A professional-grade **Decision Support System (DSS)** designed for football scouts and recruitment analysts. This platform leverages **Machine Learning** and **Generative AI** to identify tactical archetypes, stylistic "statistical twins," and undervalued prospects across the top tiers of European football and the English pyramid.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

---

## Key Features

### Advanced Analytics & ML
*   **Weighted Similarity Engine:** Uses **Cosine Similarity** with "Triple Weighting" on elite traits (>75th percentile) to find stylistic matches based on strengths, not shared weaknesses.
*   **Tactical Style Map (PCA):** Visualizes the "Footballing Universe" in 2D. IDs "tactical hybrids" (e.g., Fullbacks acting as Midfielders) sitting between clusters.
*   **Archetype Discovery:** Implements **K-Means Clustering** to categorize players into 8-12 tactical roles (e.g., *Deep-Lying Playmaker*, *Target Man*).
*   **Explainable AI (SHAP & Drivers):**
    *   **Similarity Drivers:** Identifies the exact metrics driving the similarity between two players.
    *   **SHAP Explanations:** Deconstructs the "Performance Valuation" model to show which stats contribute most to a player's rating.
*   **Age-Curve Analysis:** Detects "High-Ceiling Prospects" by calculating Z-scores relative to specific age cohorts (e.g., "This 19-year-old is 2.1 standard deviations above average for his age").

### Operational Tools
*   **Generative Scouting Reports:** Integrated **Gemini 1.5 Pro API** generates prose-style scouting narratives, analyzing tactical profiles and statistical outliers.
*   **Hidden Gems Discovery:** Multi-factor filtering (Age, League Tier, Efficiency Metrics) to identify high-potential prospects outside major leagues.
*   **PDF Dossier Export:** One-click generation of professional scouting reports for offline recruitment meetings.
*   **Head-to-Head Comparison:** Radar charts and "Relative Quality" toggles for direct player verification.

---

## System Architecture

The project follows a modern **Client-Server Architecture**:

### 1. **Backend (FastAPI)**
*   **Hybrid Schema (SQL + JSON):** Uses SQLite with a JSON column for stats. This allows flexible storage of varying stats per league (e.g., Premier League has `xG`, National League only has `Gls/90`) without schema migrations.
*   **Stateful Services:** The `SimilarityEngine` and `ModelEvaluator` are loaded once at startup for high-performance inference.
*   **API Endpoints:** RESTful API for search, similarity, and attribution.

### 2. **Frontend (Streamlit)**
*   **Interactive UI:** Plots, interactive tables, and dynamic filtering.
*   **API Integration:** Communicates with the FastAPI backend via a resilient `APIClient` (with local fallback mode).

### 3. **Data Pipeline (ETL)**
*   **Idempotent Ingestion:** `etl/ingest_data.py` uses `UPSERT` logic to safely re-run data loads without duplicates.
*   **Structured Logging:** JSON-based logging tracks data quality and missing value percentages per ingestion run.

---

## Technical Rigor

1.  **Positional Peer-Group Scaling:** Features are standardized using `StandardScaler` fitted **separately** for Forwards, Midfielders, and Defenders.
2.  **League-Aware Imputation:** Handles "Data Deserts" (lower leagues) by using positional medians from adjacent tracked leagues rather than zero-filling.
3.  **Goalkeeper Isolation:** Outfield metrics are strictly masked from GK profiles to prevent statistical contamination.
4.  **Performance Valuation:** Uses a Random Forest Regressor (R-squared monitored) to estimate a "Performance Value" based on on-pitch stats, validated against market data.

---

## Project Structure

```bash
|-- app.py                      # Streamlit Frontend Entry Point
|-- backend/
|   |-- app/
|   |   |-- main.py             # FastAPI Backend Entry Point
|   |   |-- models.py           # SQLAlchemy Models (Hybrid Schema)
|   |   `-- services/           # Business Logic
|   |       `-- evaluator.py    # ML Model & SHAP Explainer
|-- etl/
|   |-- ingest_data.py          # CSV -> SQLite Ingestion Script
|   `-- logs/                   # Structured ETL Logs
|-- utils/
|   |-- data_engine.py          # Cleaning & Preprocessing Pipeline
|   |-- similarity.py           # Weighted Cosine Similarity Logic
|   |-- visualizations.py       # Plotly Components
|   |-- llm_integration.py      # Gemini API Client
|   `-- pdf_export.py           # FPDF Report Generator
|-- data/                       # CSV Data Source
`-- Dockerfile                  # Container Configuration
```

---

## Installation & Usage

### Prerequisites
*   Python 3.9+
*   Google Gemini API Key (for narrative generation)

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/scouting-dashboard.git
cd scouting-dashboard
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_api_key_here
```

### 3. Run the System
**Option A: Full System (Recommended)**
Run the startup script to launch both Backend and Frontend:
```bash
./start.sh
```

**Option B: Manual Start**
Terminal 1 (Backend):
```bash
uvicorn backend.app.main:app --reload
```
Terminal 2 (Frontend):
```bash
streamlit run app.py
```

---

## Limitations & Future Work

*   **Data Sparsity:** Lower league style-matching relies on output proxies due to lack of event-level data (dribbles/crosses).
*   **Cold Start:** Initial ingestion handles 3,000+ players; migration to PostgreSQL is recommended for multi-season horizontal scaling.

---

## Disclaimer
Data sourced from various providers (FBref, Understat) for educational and demonstration purposes. This tool is a **Decision Support System** and should be used to *augment*, not replace, traditional video and live scouting.
