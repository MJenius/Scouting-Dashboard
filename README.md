# ⚽ Football Scouting & Recruitment Dashboard

A professional-grade **Decision Support System (DSS)** designed for football scouts and recruitment analysts. This platform leverages Machine Learning to identify tactical archetypes, stylistic "statistical twins," and undervalued prospects across the top tiers of European football and the English pyramid.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)

## 🚀 Key Features

* **🔍 Advanced Similarity Engine:** Uses **Weighted Cosine Similarity** to find stylistic matches. Unlike standard models, this engine applies "Triple Weighting" to a benchmark player's elite traits (>75th percentile) to ensure matches are based on strengths, not shared weaknesses.
* **🌌 Tactical Style Map (PCA):** Visualizes the entire "Footballing Universe" in 2D using **Principal Component Analysis**. Scouts can identify "tactical hybrids"—players sitting on the border between clusters (e.g., an Inverted Fullback sitting between a Defender and Midfielder).
* **🤖 Archetype Discovery:** Implements **K-Means Clustering** with dynamic $k$ optimization to automatically categorize players into 8-12 tactical archetypes (e.g., *Deep-Lying Playmaker*, *Target Man*, *Defensive Anchor*).
* **📊 Explainable AI (Similarity Drivers):** Moves beyond "black box" scores by explicitly identifying the top 3 statistical metrics driving the similarity between any two players.
* **💎 Hidden Gems Discovery:** A multi-factor filtering system that cross-references performance percentiles with age and league tier to identify high-potential prospects outside the Premier League.
* **📝 Generative Scouting Reports:** Integrated with the **Gemini 1.5 Pro API** to generate unique, prose-style scouting narratives based on a player's tactical profile and statistical outliers.

---

## 🛠️ Technical Rigor & Data Pipeline

This project implements industry-standard data science practices to ensure statistical integrity:

1.  **Positional Peer-Group Scaling:** Features are standardized using `StandardScaler` fitted **separately** for Forwards, Midfielders, and Defenders. This ensures a Center-Back is ranked relative to other defenders, not compared unfairly to a Striker's goal volume.
2.  **League-Aware Imputation:** Handles the "Data Desert" of lower leagues (like the National League) by using positional medians from adjacent tracked leagues (League Two) rather than defaulting missing values to zero, which would skew similarity logic.
3.  **Expected Metrics Integration:** Merges raw output stats with advanced **Expected Goals (xG)**, **Expected Assists (xA)**, **xGChain**, and **xGBuildup** metrics to provide a more predictive view of player quality.
4.  **Goalkeeper Isolation:** Outfield metrics are strictly masked from Goalkeeper profiles to prevent statistical contamination.

---

## 📁 Project Structure

```bash
├── app.py                  # Main Multi-page Streamlit Application
├── config.yaml             # Global application and ML settings
├── utils/
│   ├── data_engine.py      # 7-step Data Pipeline (Cleaning, Scaling, Imputation)
│   ├── similarity.py       # Weighted Cosine Similarity & Match Explainer
│   ├── clustering.py       # K-Means & PCA Logic
│   ├── visualizations.py   # Plotly Interactive Components (Beeswarms, Radars)
│   ├── llm_integration.py  # Gemini API Generative Narratives
│   └── pdf_export.py       # One-page PDF Dossier Generation
└── data/                   # Master and Advanced Stats CSV Files
```

## 💻 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/scouting-dashboard.git
   cd scouting-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API Keys**
   Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

## ⚠️ Disclaimer
Data sourced from various providers (FBref, Understat). This tool is intended for recruitment analysis and decision support.
