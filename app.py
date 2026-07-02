"""
app.py
------
Streamlit web interface for FinBotX.

Features:
  - Manual data entry form grouped by financial category
  - Excel file upload (auto-maps column names)
  - Validates critical fields before running prediction
  - Runs full analysis pipeline and displays the HTML report inline
  - Download button for the generated report

Run:
    streamlit run app.py
"""

import os
import tempfile

import numpy as np
import pandas as pd
import streamlit as st

from src.bankruptcy_engine import BankruptcyEngine
from src.sentiment_engine  import SentimentEngine
from src.news_scraper      import NewsScraper
from src.report_generator  import ReportGenerator
from src.column_mapping    import FEATURE_GROUPS, CRITICAL_FEATURES, ALL_FEATURES

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

st.set_page_config(
    page_title="FinBotX — Financial AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] { font-family: "DM Sans", sans-serif; }

  .main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
  }
  .main-header h1 {
    font-family: "DM Serif Display", serif;
    font-size: 2.8rem;
    color: white;
    margin: 0;
    letter-spacing: -1px;
  }
  .main-header p {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 0.5rem;
  }
  .section-label {
    font-family: "DM Mono", monospace;
    font-size: 0.7rem;
    color: #6366f1;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
  }
  .critical-note {
    background: #1e293b;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    font-size: 0.85rem;
    color: #94a3b8;
    margin-bottom: 1.5rem;
  }
  .stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: "DM Sans", sans-serif;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.7rem 2rem;
    width: 100%;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Engine loader (cached — runs only once per session)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_engines():
    bankruptcy = BankruptcyEngine()
    bankruptcy.load_model()
    sentiment  = SentimentEngine()
    sentiment.load_model()
    return {
        "bankruptcy": bankruptcy,
        "sentiment":  sentiment,
        "scraper":    NewsScraper(),
        "report":     ReportGenerator(),
    }

# ------------------------------------------------------------------
# Helper: collect form values into a flat dict
# ------------------------------------------------------------------

def collect_form_values(form_inputs: dict) -> dict:
    """Converts raw form widget values to float (None if empty)."""
    result = {}
    for feature, value in form_inputs.items():
        if value == "" or value is None:
            result[feature] = None
        else:
            try:
                result[feature] = float(value)
            except (ValueError, TypeError):
                result[feature] = None
    return result

# ------------------------------------------------------------------
# Helper: parse uploaded Excel
# ------------------------------------------------------------------

def parse_excel(uploaded_file) -> dict:
    """
    Reads an Excel file where:
      Row 1 = feature names  (matching ALL_FEATURES clean English names)
      Row 2 = values

    OR a two-column layout:
      Column A = feature name
      Column B = value
    """
    df = pd.read_excel(uploaded_file)

    # Detect layout
    if df.shape[0] >= 1 and df.shape[1] >= 2:
        first_col_vals = df.iloc[:, 0].astype(str).str.lower().tolist()
        # Two-column layout: first column contains feature names
        if any(f in first_col_vals for f in ALL_FEATURES):
            df.columns = ["feature", "value"]
            mapping = dict(zip(df["feature"].astype(str).str.strip(),
                               df["value"]))
            return {k: v for k, v in mapping.items() if k in ALL_FEATURES}

    # Wide layout: first row = headers, second row = values
    data = {}
    for col in df.columns:
        col_clean = str(col).strip()
        if col_clean in ALL_FEATURES and not df[col].empty:
            data[col_clean] = df[col].iloc[0]
    return data

# ------------------------------------------------------------------
# Main app
# ------------------------------------------------------------------

def run_app(engines: dict = None):
    if engines is None:
        engines = load_engines()

    # ── Header ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
      <h1>📊 FinBotX</h1>
      <p>Financial Resilience &amp; Market Sentiment AI · Academic Research System</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Analysis Settings")

        company_name = st.text_input(
            "Company Name",
            placeholder="e.g. Apple, Tesla …",
            help="Used to search for relevant news headlines"
        )
        stock_symbol = st.text_input(
            "Stock Symbol (optional)",
            placeholder="e.g. AAPL, TSLA",
            help="Improves news search accuracy"
        )

        st.divider()
        input_mode = st.radio(
            "Data Input Method",
            ["✏️  Manual Entry", "📂  Upload Excel"],
            index=0
        )

        st.divider()
        st.markdown("""
        <div style="font-size:0.78rem;color:#64748b;line-height:1.7">
        <strong style="color:#94a3b8">Model Info</strong><br/>
        Financial: Random Forest<br/>
        Accuracy: 97.36%<br/><br/>
        Sentiment: Logistic Regression<br/>
        Accuracy: 88.08%<br/><br/>
        Dataset: 6,819 companies<br/>
        Features: 95 financial ratios
        </div>
        """, unsafe_allow_html=True)

    # ── Input area ──────────────────────────────────────────────────
    form_values = {}

    if "✏️" in input_mode:
        st.markdown('<div class="critical-note">⭐ <strong>Bold fields are critical</strong> — must be filled for a valid prediction. All other fields are optional.</div>', unsafe_allow_html=True)

        tab_names = list(FEATURE_GROUPS.keys())
        tabs = st.tabs(tab_names)

        for tab, (group_name, features) in zip(tabs, FEATURE_GROUPS.items()):
            with tab:
                cols = st.columns(3)
                for i, feature in enumerate(features):
                    label = feature.replace("_", " ").title()
                    is_critical = feature in CRITICAL_FEATURES
                    display_label = f"**{label}**" if is_critical else label
                    with cols[i % 3]:
                        val = st.text_input(
                            display_label,
                            key=f"field_{group_name}_{feature}",
                            placeholder="e.g. 0.35",
                            help="Critical field — required" if is_critical else "Optional"
                        )
                        form_values[feature] = val

    else:
        st.markdown("#### 📂 Upload Excel File")
        st.markdown("""
        <div class="critical-note">
        Your Excel file should have one of these layouts:<br/>
        &nbsp;&nbsp;<strong>Wide:</strong> Row 1 = feature names, Row 2 = values<br/>
        &nbsp;&nbsp;<strong>Tall:</strong> Column A = feature name, Column B = value
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Choose Excel file", type=["xlsx", "xls"])
        if uploaded:
            try:
                parsed = parse_excel(uploaded)
                if parsed:
                    st.success(f"✅ Loaded {len(parsed)} features from file.")
                    form_values = {k: str(v) for k, v in parsed.items()}

                    with st.expander("Preview loaded data"):
                        preview_df = pd.DataFrame(
                            [(k.replace("_"," ").title(), v)
                             for k, v in parsed.items()],
                            columns=["Feature", "Value"]
                        )
                        st.dataframe(preview_df, use_container_width=True)
                else:
                    st.warning("⚠️ No matching feature columns found. Check column names.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # ── Run analysis button ─────────────────────────────────────────
    st.divider()
    run_col, _ = st.columns([1, 2])
    with run_col:
        run_clicked = st.button("🚀 Generate Financial Report")

    if run_clicked:
        if not company_name.strip():
            st.error("Please enter a company name in the sidebar.")
            st.stop()

        # Collect and validate
        collected = collect_form_values(form_values)
        is_valid, missing = engines["bankruptcy"].validate_input(collected)

        if not is_valid:
            missing_labels = [m.replace("_", " ").title() for m in missing]
            st.error(
                "⚠️ Missing critical fields: " +
                ", ".join(missing_labels) +
                ". Please fill them in before running the analysis."
            )
            st.stop()

        # Build input DataFrame
        input_df = pd.DataFrame([{
            f: (collected.get(f) if collected.get(f) is not None else 0.0)
            for f in ALL_FEATURES
        }])

        with st.spinner("🔍 Running financial analysis..."):

            # 1. Financial prediction
            prediction = engines["bankruptcy"].predict(input_df)

            # 2. News scraping
            headlines = engines["scraper"].fetch_headlines(
                company_name.strip(), stock_symbol.strip()
            )

            # 3. Sentiment analysis
            sentiment_results = []
            if headlines:
                texts  = [h["headline"] for h in headlines]
                scores = engines["sentiment"].analyze_batch(texts)
                for h, s in zip(headlines, scores):
                    sentiment_results.append({
                        "headline":   h["headline"],
                        "source":     h["source"],
                        "published":  h["published"],
                        "label":      s["label"],
                        "confidence": s["confidence"],
                        "scores":     s["scores"],
                    })

            # 4. Generate report
            submitted_clean = {
                k: v for k, v in collected.items() if v is not None
            }
            with tempfile.NamedTemporaryFile(
                suffix=".html", delete=False, mode="w", encoding="utf-8"
            ) as tmp:
                report_path = tmp.name

            engines["report"].generate(
                company_name    = company_name.strip(),
                prediction_result = prediction,
                sentiment_results = sentiment_results,
                submitted_data  = submitted_clean,
                output_path     = report_path,
            )

        # ── Results summary ────────────────────────────────────────
        st.success("✅ Analysis complete!")

        risk_label = prediction["risk_label"]
        risk_color = {"Low": "#22c55e", "Medium": "#f59e0b", "High": "#ef4444"}[risk_label]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Risk Level",         risk_label)
        col2.metric("Risk Score",         f"{prediction['risk_score']:.1%}")
        col3.metric("Stability Score",    f"{prediction['stability_score']:.1%}")
        col4.metric("Headlines Found",    len(sentiment_results))

        # ── SHAP top factors ────────────────────────────────────────
        st.markdown("#### 🔍 Top Risk Drivers")
        factors_df = pd.DataFrame(prediction["top_risk_factors"])
        factors_df["feature"] = factors_df["feature"].str.replace("_", " ").str.title()
        factors_df["direction"] = factors_df["shap_impact"].apply(
            lambda x: "↑ Risk" if x > 0 else "↓ Stability"
        )
        st.dataframe(
            factors_df[["feature", "value", "shap_impact", "direction"]].rename(columns={
                "feature": "Feature", "value": "Value",
                "shap_impact": "SHAP Impact", "direction": "Direction"
            }),
            use_container_width=True,
            hide_index=True,
        )

        # ── Sentiment summary ───────────────────────────────────────
        if sentiment_results:
            st.markdown("#### 📰 Sentiment Summary")
            sent_df = pd.DataFrame(sentiment_results)[
                ["headline", "label", "confidence", "source"]
            ].rename(columns={
                "headline": "Headline", "label": "Sentiment",
                "confidence": "Confidence", "source": "Source"
            })
            st.dataframe(sent_df, use_container_width=True, hide_index=True)
        else:
            st.info("ℹ️ No news headlines found — sentiment section skipped in report.")

        # ── Download button ─────────────────────────────────────────
        st.divider()
        with open(report_path, "r", encoding="utf-8") as f:
            report_html = f.read()

        st.download_button(
            label     = "⬇️ Download Full HTML Report",
            data      = report_html,
            file_name = f"FinBotX_{company_name.replace(' ','_')}_Report.html",
            mime      = "text/html",
        )

        # Embed report preview
        st.markdown("#### 📄 Report Preview")
        st.components.v1.html(report_html, height=900, scrolling=True)

        os.unlink(report_path)


# ------------------------------------------------------------------
# Direct launch
# ------------------------------------------------------------------

if __name__ == "__main__":
    run_app()
