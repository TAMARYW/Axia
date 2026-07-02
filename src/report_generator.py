"""
report_generator.py
-------------------
Generates a full financial analysis report as a self-contained HTML file.

The report includes:
  - Company header with risk badge (Low / Medium / High)
  - Financial resilience score with visual gauge
  - SHAP bar chart: top features driving the prediction
  - Sentiment section (shown only if headlines were found)
  - Full data table of all submitted financial metrics
  - Methodology notes

All charts are embedded as base64 PNG — no external dependencies at runtime.
"""

import base64
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RISK_COLORS = {
    "Low":    "#22c55e",
    "Medium": "#f59e0b",
    "High":   "#ef4444",
}

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "neutral":  "#6366f1",
    "negative": "#ef4444",
}


class ReportGenerator:

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        company_name:      str,
        prediction_result: dict,
        sentiment_results: list,
        submitted_data:    dict,
        output_path:       str = "report.html",
    ) -> str:
        gauge_img = self._build_gauge(prediction_result["risk_score"])
        shap_img  = self._build_shap_chart(prediction_result["top_risk_factors"])
        sent_img  = self._build_sentiment_chart(sentiment_results) if sentiment_results else None

        html = self._render_html(
            company_name, prediction_result, sentiment_results,
            submitted_data, gauge_img, shap_img, sent_img
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return output_path

    # ------------------------------------------------------------------
    # Chart builders
    # ------------------------------------------------------------------

    def _build_gauge(self, risk_score: float) -> str:
        fig, ax = plt.subplots(figsize=(5, 2.8), subplot_kw={"aspect": "equal"})
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")

        segments = [
            (0,   60,  "#22c55e"),
            (60,  120, "#f59e0b"),
            (120, 180, "#ef4444"),
        ]
        for start, end, color in segments:
            theta = np.linspace(np.radians(180 - end), np.radians(180 - start), 100)
            ax.fill_between(
                np.cos(theta), np.sin(theta),
                0.6 * np.cos(theta), 0.6 * np.sin(theta),
                color=color, alpha=0.85
            )

        angle = np.radians(180 - risk_score * 180)
        ax.annotate(
            "", xy=(0.72 * np.cos(angle), 0.72 * np.sin(angle)),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5, mutation_scale=18)
        )
        ax.add_patch(plt.Circle((0, 0), 0.07, color="white", zorder=5))
        ax.text(0, -0.22, "{:.0%}".format(risk_score), color="white",
                fontsize=22, fontweight="bold", ha="center", va="center",
                fontfamily="monospace")
        ax.text(0, -0.42, "RISK SCORE", color="#94a3b8",
                fontsize=8, ha="center", va="center", fontfamily="monospace")
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.55, 1.1)
        ax.axis("off")
        plt.tight_layout(pad=0)
        return self._fig_to_base64(fig)

    def _build_shap_chart(self, top_factors: list) -> str:
        if not top_factors:
            return ""

        labels = [f["feature"].replace("_", " ").title() for f in top_factors]
        values = [f["shap_impact"] for f in top_factors]
        colors = ["#ef4444" if v > 0 else "#22c55e" for v in values]

        fig, ax = plt.subplots(figsize=(7, max(3.5, len(labels) * 0.42)))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")

        bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1],
                       height=0.6, edgecolor="none")
        ax.axvline(0, color="#475569", linewidth=0.8)
        ax.set_xlabel("SHAP Impact → Risk", color="#94a3b8", fontsize=9)
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)

        for bar, val in zip(bars, values[::-1]):
            ax.text(
                val + (0.001 if val >= 0 else -0.001),
                bar.get_y() + bar.get_height() / 2,
                "{:+.4f}".format(val), va="center",
                ha="left" if val >= 0 else "right",
                color="white", fontsize=7, fontfamily="monospace"
            )

        plt.tight_layout(pad=0.5)
        return self._fig_to_base64(fig)

    def _build_sentiment_chart(self, sentiment_results: list) -> str:
        from collections import Counter
        counts = Counter(r["label"] for r in sentiment_results)
        labels = list(counts.keys())
        sizes  = list(counts.values())
        colors = [SENTIMENT_COLORS.get(l, "#6366f1") for l in labels]

        fig, ax = plt.subplots(figsize=(4, 3))
        fig.patch.set_facecolor("#0f172a")
        ax.set_facecolor("#0f172a")

        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, colors=colors,
            autopct="%1.0f%%", startangle=90,
            wedgeprops={"width": 0.55, "edgecolor": "#0f172a", "linewidth": 2},
            pctdistance=0.75,
        )
        for at in autotexts:
            at.set_color("white")
            at.set_fontsize(9)

        legend_patches = [
            mpatches.Patch(color=c, label=l.capitalize())
            for l, c in zip(labels, colors)
        ]
        ax.legend(handles=legend_patches, loc="lower center",
                  bbox_to_anchor=(0.5, -0.12), ncol=3,
                  frameon=False, labelcolor="#cbd5e1", fontsize=8)
        plt.tight_layout(pad=0.3)
        return self._fig_to_base64(fig)

    # ------------------------------------------------------------------
    # HTML helpers
    # ------------------------------------------------------------------

    def _sentiment_rows(self, sentiment_results: list) -> str:
        rows = []
        for r in sentiment_results[:8]:
            color    = SENTIMENT_COLORS.get(r["label"], "#6366f1")
            headline = r.get("headline", "—")
            label    = r["label"]
            conf     = "{:.0%}".format(r["confidence"])
            row  = "<tr><td>" + headline + "</td>"
            row += '<td><span style="color:' + color + '">' + label + "</span></td>"
            row += "<td>" + conf + "</td></tr>"
            rows.append(row)
        return "".join(rows)

    def _data_rows(self, submitted_data: dict) -> str:
        rows = []
        for k, v in submitted_data.items():
            if v not in (None, "", "nan"):
                label = k.replace("_", " ").title()
                rows.append("<tr><td>" + label + "</td><td>" + str(v) + "</td></tr>")
        return "".join(rows)

    def _sentiment_section(self, sentiment_results, sent_img) -> str:
        if not (sentiment_results and sent_img):
            return """
            <section class="card muted">
              <h2>&#128�; Media Sentiment Analysis</h2>
              <p>No recent headlines were found for this company.
                 Sentiment analysis was skipped.</p>
            </section>"""

        from collections import Counter
        label_counts = Counter(r["label"] for r in sentiment_results)
        dominant     = max(label_counts, key=label_counts.get)
        avg_conf     = sum(r["confidence"] for r in sentiment_results) / len(sentiment_results)
        dom_color    = SENTIMENT_COLORS.get(dominant, "#6366f1")
        n            = len(sentiment_results)
        rows_html    = self._sentiment_rows(sentiment_results)

        return (
            '<section class="card">'
            "<h2>&#128240; Media Sentiment Analysis</h2>"
            '<p class="subtitle">Based on ' + str(n) + " recent headlines &middot; "
            "Average confidence " + "{:.0%}".format(avg_conf) + "</p>"
            '<div class="two-col">'
            '<div><img src="data:image/png;base64,' + sent_img + '" style="width:100%;max-width:320px"/></div>'
            '<div class="sentiment-summary">'
            '<div class="badge" style="background:' + dom_color + '22;color:' + dom_color + ';border-color:' + dom_color + '">'
            "Dominant: " + dominant.upper() + "</div>"
            '<table class="data-table" style="margin-top:1rem">'
            "<thead><tr><th>Headline</th><th>Sentiment</th><th>Confidence</th></tr></thead>"
            "<tbody>" + rows_html + "</tbody>"
            "</table></div></div></section>"
        )

    # ------------------------------------------------------------------
    # Main HTML renderer
    # ------------------------------------------------------------------

    def _render_html(
        self, company_name, pred, sentiment_results,
        submitted_data, gauge_img, shap_img, sent_img
    ) -> str:
        risk_label   = pred["risk_label"]
        risk_color   = RISK_COLORS[risk_label]
        risk_score   = pred["risk_score"]
        stable_score = pred["stability_score"]
        n_features   = len(submitted_data)
        now_str      = datetime.now().strftime("%B %d, %Y at %H:%M")
        year_str     = str(datetime.now().year)

        verdict = (
            "The model classifies this company as <strong>financially stable</strong> "
            "with a low probability of distress."
            if pred["prediction"] == 0 else
            "The model identifies <strong>elevated financial risk</strong>. "
            "Key indicators suggest potential vulnerability."
        )

        sentiment_html = self._sentiment_section(sentiment_results, sent_img)
        data_rows_html = self._data_rows(submitted_data)

        return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>FinBotX Report &mdash; """ + company_name + """</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    :root {
      --bg:      #0f172a; --surface: #1e293b; --border: #334155;
      --text:    #e2e8f0; --muted:   #94a3b8; --accent: #6366f1;
      --risk:    """ + risk_color + """;
    }
    body { background:var(--bg); color:var(--text);
           font-family:"DM Sans",sans-serif; font-size:15px;
           line-height:1.6; padding:2rem 1rem; }
    .container { max-width:900px; margin:0 auto; }
    .report-header { display:flex; justify-content:space-between;
      align-items:flex-start; border-bottom:1px solid var(--border);
      padding-bottom:1.5rem; margin-bottom:2rem; }
    .report-header h1 { font-family:"DM Serif Display",serif;
      font-size:2.2rem; color:white; letter-spacing:-0.5px; }
    .report-header .meta { color:var(--muted); font-size:0.85rem; margin-top:0.3rem; }
    .badge { display:inline-block; padding:0.3rem 0.9rem; border-radius:999px;
      border:1px solid var(--risk);
      background:color-mix(in srgb, var(--risk) 15%, transparent);
      color:var(--risk); font-family:"DM Mono",monospace;
      font-size:0.85rem; font-weight:500; letter-spacing:0.05em; }
    .card { background:var(--surface); border:1px solid var(--border);
      border-radius:12px; padding:1.8rem; margin-bottom:1.5rem; }
    .card.muted { opacity:0.6; }
    .card h2 { font-family:"DM Serif Display",serif;
      font-size:1.3rem; margin-bottom:0.4rem; color:white; }
    .subtitle { color:var(--muted); font-size:0.85rem; margin-bottom:1.2rem; }
    .scores-row { display:grid; grid-template-columns:1fr 1fr 1fr; gap:1rem; margin:1.2rem 0; }
    .score-box { background:#0f172a; border-radius:8px; padding:1rem;
      text-align:center; border:1px solid var(--border); }
    .score-box .value { font-family:"DM Mono",monospace; font-size:1.8rem;
      font-weight:500; color:white; }
    .score-box .label { color:var(--muted); font-size:0.75rem; margin-top:0.2rem; }
    .two-col { display:grid; grid-template-columns:auto 1fr; gap:2rem; align-items:start; }
    .verdict { border-left:3px solid var(--risk); padding:0.8rem 1rem;
      background:#0f172a; border-radius:0 8px 8px 0; margin:1rem 0; font-size:0.95rem; }
    .data-table { width:100%; border-collapse:collapse; font-size:0.85rem; }
    .data-table th { text-align:left; padding:0.5rem 0.7rem;
      border-bottom:1px solid var(--border); color:var(--muted);
      font-weight:500; font-family:"DM Mono",monospace; font-size:0.75rem; }
    .data-table td { padding:0.45rem 0.7rem; border-bottom:1px solid #1e293b; }
    .data-table tr:hover td { background:#1e293b; }
    footer { text-align:center; color:var(--muted); font-size:0.75rem;
      margin-top:3rem; padding-top:1.5rem; border-top:1px solid var(--border); }
    @media (max-width:600px) {
      .scores-row { grid-template-columns:1fr 1fr; }
      .two-col { grid-template-columns:1fr; }
    }
  </style>
</head>
<body>
<div class="container">

  <div class="report-header">
    <div>
      <div style="color:var(--muted);font-family:'DM Mono',monospace;font-size:0.75rem;margin-bottom:0.4rem">
        FINBOTX &middot; FINANCIAL RESILIENCE REPORT
      </div>
      <h1>""" + company_name + """</h1>
      <div class="meta">Generated """ + now_str + """</div>
    </div>
    <div style="text-align:right">
      <div class="badge">""" + risk_label.upper() + """ RISK</div>
      <div class="meta" style="margin-top:0.5rem">Random Forest &middot; 97.36% accuracy</div>
    </div>
  </div>

  <section class="card">
    <h2>&#128202; Financial Resilience Score</h2>
    <p class="subtitle">Machine learning prediction based on """ + str(n_features) + """ financial indicators</p>
    <div class="two-col">
      <img src="data:image/png;base64,""" + gauge_img + """" style="width:260px"/>
      <div>
        <div class="scores-row">
          <div class="score-box">
            <div class="value" style="color:""" + risk_color + """">""" + "{:.0%}".format(risk_score) + """</div>
            <div class="label">RISK PROBABILITY</div>
          </div>
          <div class="score-box">
            <div class="value" style="color:#22c55e">""" + "{:.0%}".format(stable_score) + """</div>
            <div class="label">STABILITY PROBABILITY</div>
          </div>
          <div class="score-box">
            <div class="value" style="color:var(--accent)">""" + risk_label + """</div>
            <div class="label">RISK LEVEL</div>
          </div>
        </div>
        <div class="verdict">""" + verdict + """</div>
      </div>
    </div>
  </section>

  <section class="card">
    <h2>&#128269; Key Risk Drivers (SHAP Analysis)</h2>
    <p class="subtitle">Red bars push toward risk &middot; Green bars push toward stability</p>
    <img src="data:image/png;base64,""" + shap_img + """" style="width:100%;max-width:700px"/>
  </section>

  """ + sentiment_html + """

  <section class="card">
    <h2>&#128203; Submitted Financial Data</h2>
    <p class="subtitle">All metrics provided for this analysis</p>
    <table class="data-table">
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>""" + data_rows_html + """</tbody>
    </table>
  </section>

  <section class="card muted">
    <h2>&#8505;&#65039; Methodology</h2>
    <p style="font-size:0.85rem;color:var(--muted);line-height:1.8">
      <strong style="color:var(--text)">Financial Model:</strong>
      Random Forest Classifier trained on the Taiwan Economic Journal bankruptcy
      dataset (6,819 companies, 95 financial ratios). Test accuracy: 97.36%.
      SHAP TreeExplainer used for feature attribution.<br/><br/>
      <strong style="color:var(--text)">Sentiment Model:</strong>
      Logistic Regression with TF-IDF bigrams trained on Financial PhraseBank v1.0
      (AllAgree subset, 2,264 sentences). SMOTE applied for class balancing.
      Test accuracy: 88.08%.<br/><br/>
      <strong style="color:var(--text)">Disclaimer:</strong>
      This report is generated by an academic machine learning system and does
      not constitute financial advice.
    </p>
  </section>

  <footer>FinBotX &middot; Academic Financial AI System &middot; """ + year_str + """</footer>
</div>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _fig_to_base64(self, fig) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    dummy_prediction = {
        "prediction":      1,
        "stability_score": 0.32,
        "risk_score":      0.68,
        "risk_label":      "High",
        "top_risk_factors": [
            {"feature": "debt_ratio",                     "value":  0.82, "shap_impact":  0.043},
            {"feature": "roa_before_tax_c",               "value": -0.03, "shap_impact":  0.031},
            {"feature": "cash_flow_to_total_assets",      "value":  0.01, "shap_impact":  0.027},
            {"feature": "current_ratio",                  "value":  0.95, "shap_impact": -0.019},
            {"feature": "retained_earnings_to_total_assets","value":-0.12, "shap_impact":  0.018},
            {"feature": "working_capital_to_total_assets","value": -0.08, "shap_impact":  0.015},
            {"feature": "net_value_to_assets",            "value":  0.18, "shap_impact": -0.012},
            {"feature": "interest_coverage_ratio",        "value":  1.20, "shap_impact":  0.011},
            {"feature": "total_asset_turnover",           "value":  0.45, "shap_impact": -0.009},
            {"feature": "eps_last_four_quarters",         "value": -0.50, "shap_impact":  0.008},
        ],
    }
    dummy_sentiment = [
        {"headline": "Company faces credit downgrade amid rising debt", "label": "negative", "confidence": 0.81},
        {"headline": "Quarterly revenue misses analyst expectations",   "label": "negative", "confidence": 0.74},
        {"headline": "Management announces restructuring plan",         "label": "neutral",  "confidence": 0.61},
    ]
    dummy_data = {
        "debt_ratio": 0.82, "roa_before_tax_c": -0.03,
        "current_ratio": 0.95, "cash_flow_to_total_assets": 0.01,
        "eps_last_four_quarters": -0.5,
    }

    gen  = ReportGenerator()
    path = gen.generate("Acme Corp", dummy_prediction, dummy_sentiment, dummy_data, "/tmp/test_report.html")
    print("Report saved to:", path)
