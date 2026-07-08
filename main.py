"""
main.py
-------
Entry point for the Axia financial analysis system.

Responsibilities:
  - Validate that required data files exist
  - Initialize (load or train) all engines once at startup
  - Hand ready engine instances to the Streamlit app layer

Run:
    python main.py           (validates + trains if needed, then launches UI)
    streamlit run app.py     (direct UI launch)
"""

import os
import sys

from src.bankruptcy_engine import BankruptcyEngine
from src.sentiment_engine  import SentimentEngine
from src.news_scraper      import NewsScraper
from src.report_generator  import ReportGenerator


def initialize_engines() -> dict:
    """Loads or trains all ML engines; returns ready-to-use instances."""
    print("=" * 52)
    print("  Axia - Financial Resilience AI")
    print("=" * 52)

    print("\n[1/2] Loading Financial Resilience Engine...")
    bankruptcy = BankruptcyEngine()
    bankruptcy.load_model()

    print("\n[2/2] Loading Sentiment Engine...")
    sentiment = SentimentEngine()
    sentiment.load_model()

    print("\nAll engines ready.\n")
    return {
        "bankruptcy": bankruptcy,
        "sentiment":  sentiment,
        "scraper":    NewsScraper(),
        "report":     ReportGenerator(),
    }


def main():
    required = [
        os.path.join("data", "bankruptcy_data.csv"),
        os.path.join("data", "Sentences_AllAgree.txt"),
    ]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("Missing required data files:")
        for f in missing:
            print(f"   - {f}")
        sys.exit(1)

    engines = initialize_engines()

    from app import run_app
    run_app(engines)


if __name__ == "__main__":
    main()
