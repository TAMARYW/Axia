"""
main.py
-------
Entry point for the FinBotX financial analysis system.

Responsibilities:
  - Initialize all engines once at startup (Dependency Injection)
  - Ensure models are trained / loaded before the app starts
  - Pass engine instances into the app layer

Run:
    python main.py
"""

import os
import sys

from src.bankruptcy_engine import BankruptcyEngine
from src.sentiment_engine   import SentimentEngine
from src.news_scraper       import NewsScraper
from src.report_generator   import ReportGenerator


def initialize_engines() -> dict:
    """
    Loads or trains all ML engines at startup.
    Returns a dict of ready-to-use engine instances.
    """
    print("=" * 52)
    print("  FinBotX — Financial AI System")
    print("=" * 52)

    engines = {}

    # Financial resilience model
    print("\n[1/2] Loading Financial Resilience Engine...")
    bankruptcy_engine = BankruptcyEngine()
    bankruptcy_engine.load_model()
    engines["bankruptcy"] = bankruptcy_engine

    # Sentiment NLP model
    print("\n[2/2] Loading Sentiment Engine...")
    sentiment_engine = SentimentEngine()
    sentiment_engine.load_model()
    engines["sentiment"] = sentiment_engine

    # Stateless utilities (no training needed)
    engines["scraper"]   = NewsScraper()
    engines["report"]    = ReportGenerator()

    print("\n✅ All engines ready.\n")
    return engines


def main():
    # Validate project structure before starting
    required_files = [
        os.path.join("data", "bankruptcy_data.csv"),
        os.path.join("data", "Sentences_AllAgree.txt"),
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print("❌ Missing required data files:")
        for f in missing:
            print(f"   - {f}")
        print("\nPlease place the data files in the data/ directory and try again.")
        sys.exit(1)

    engines = initialize_engines()

    # Launch the Streamlit UI (imported here to avoid circular imports)
    from app import run_app
    run_app(engines)


if __name__ == "__main__":
    main()
