# FinBotX — Financial Resilience Prediction & Market Sentiment AI

FinBotX predicts a company's financial resilience from its financial ratios
and combines the prediction with a recency-weighted market-sentiment analysis,
producing a professional, self-contained HTML report with a credit-style
rating (A+ … D).

![FinBotX](assets/logo.png)

## Architecture

| Layer | Component | Technology |
|---|---|---|
| Financial model | `src/bankruptcy_engine.py` | XGBoost + isotonic calibration (scikit-learn) |
| Sentiment model | `src/sentiment_engine.py` | TF-IDF bigrams + SMOTE + Logistic Regression |
| News ingestion | `src/news_scraper.py` | Public RSS feeds, 3-day freshness window |
| Explainability | SHAP TreeExplainer (cached) | `shap` |
| Report | `src/report_generator.py` | Self-contained HTML + base64 charts |
| Client | `app.py` | Streamlit |
| Configuration | `src/config.py` | Rating scale, news window, thresholds |

## Key design decisions

**Honest metrics, not vanity accuracy.** Only ~3.2% of companies in the
Taiwan Economic Journal dataset are at risk, so a dummy "always stable"
model scores ~96.8% accuracy. FinBotX therefore optimizes and reports
**recall / precision / ROC-AUC for the at-risk class** (metrics are written
to `models/model_metrics.json` at training time and read dynamically by the
UI — nothing is hard-coded).

Held-out test results:
- ROC-AUC: **0.9575**
- At-risk recall: **0.80** (the original Random Forest caught only 0.25)
- Decision threshold tuned on a leak-free validation split, maximizing
  F2 (recall-weighted), because missing a genuinely distressed company is
  costlier than a false alarm.

**Calibrated probabilities drive the rating.** The A+ … D scale maps the
*calibrated* stability probability (isotonic regression), so "92% stable"
actually means ~92%.

| Grade | Stability | Meaning |
|---|---|---|
| A+ | 95–100% | Very high stability — minimal distress risk |
| A | 85–95% | High financial stability |
| A− | 75–85% | Good stability — low risk |
| B+ | 60–75% | Reasonable stability — growth potential |
| B | 45–60% | Grey zone — further review recommended |
| B− | 30–45% | Signs of weakness — elevated risk |
| C | 15–30% | Significant financial risk |
| D | 0–15% | Financial distress — very high risk |

**Fresh, recency-weighted sentiment.** Headlines older than
`NEWS_WINDOW_DAYS` (default **3 days**, `src/config.py`) are discarded at
the parsing stage. Within the window, items are weighted by exponential
decay (half-life 1.5 days), and the aggregate sentiment score (−1 … +1,
also shown as a 0–100 index) is the weighted mean of per-item polarity
P(positive) − P(negative). Alternatively, upload a `.txt` file of opinions
(one per line) instead of using the news feeds.

## Setup

```bash
pip install -r requirements.txt
python main.py            # validates data, trains models if missing, launches UI
# or directly:
streamlit run app.py
```

First launch trains both models automatically (~1–2 minutes); afterwards the
saved artifacts in `models/` load instantly.

## Using the app

1. Enter the company name (and optionally its stock symbol) in the sidebar.
2. Upload an Excel file with the financial data. Two layouts are accepted:
   - **Wide**: row 1 = feature names, row 2 = values
   - **Tall**: column A = feature name, column B = value
   Header matching is flexible — `Debt Ratio`, `debt_ratio` and
   `debt-ratio` are all recognized. Sample files are in `templates/`.
3. Choose the sentiment source: live news feeds (last 3 days) or an
   uploaded opinions file.
4. Generate the report: rating badge, calibrated probabilities, SHAP risk
   drivers, sentiment section, and a downloadable HTML report.

## Project structure

```
finbotx/
├── app.py                     # Streamlit client
├── main.py                    # Entry point (validation + engine warm-up)
├── requirements.txt
├── assets/logo.png
├── data/
│   ├── bankruptcy_data.csv    # Taiwan Economic Journal (6,819 companies)
│   └── Sentences_AllAgree.txt # Financial PhraseBank v1.0
├── models/                    # trained artifacts + model_metrics.json
├── src/
│   ├── config.py              # rating scale, news window, thresholds
│   ├── column_mapping.py      # raw → clean feature names, groups, criticals
│   ├── bankruptcy_engine.py
│   ├── sentiment_engine.py
│   ├── news_scraper.py
│   └── report_generator.py
└── templates/                 # sample Excel files + example report
```

## Disclaimer

FinBotX is an academic machine-learning system and does not constitute
financial advice.
