"""
news_scraper.py
---------------
Scrapes financial news headlines for a given company name or stock symbol
from a curated list of reliable RSS feeds.

Strategy:
  - Query up to 3 RSS sources per company
  - Filter headlines that mention the company name or symbol
  - Return up to MAX_HEADLINES results
  - If nothing is found, return an empty list (caller handles gracefully)

No API keys required — uses public RSS feeds only.
"""

import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

# Maximum headlines to collect per company (keeps report concise)
MAX_HEADLINES = 10

# Curated list of financial RSS feeds (public, no authentication needed)
RSS_FEEDS = [
    {
        "source": "Reuters Business",
        "url":    "https://feeds.reuters.com/reuters/businessNews",
    },
    {
        "source": "CNBC Finance",
        "url":    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    },
    {
        "source": "MarketWatch",
        "url":    "https://feeds.marketwatch.com/marketwatch/topstories/",
    },
    {
        "source": "Yahoo Finance",
        "url":    "https://finance.yahoo.com/rss/",
    },
    {
        "source": "Investing.com",
        "url":    "https://www.investing.com/rss/news.rss",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

TIMEOUT_SECONDS = 6


class NewsScraper:

    def __init__(self, max_headlines: int = MAX_HEADLINES):
        self.max_headlines = max_headlines

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_headlines(self, company_name: str, symbol: str = "") -> list[dict]:
        """
        Fetches relevant financial headlines for a given company.

        Parameters
        ----------
        company_name : str
            Full company name (e.g. "Apple")
        symbol : str
            Stock ticker symbol (e.g. "AAPL") — optional but improves matching

        Returns
        -------
        list of dicts:
            [{"headline": str, "source": str, "published": str}, ...]
            Returns empty list if no relevant headlines found or all feeds fail.
        """
        keywords = self._build_keywords(company_name, symbol)
        collected = []

        for feed in RSS_FEEDS:
            if len(collected) >= self.max_headlines:
                break

            headlines = self._fetch_feed(feed["source"], feed["url"], keywords)
            collected.extend(headlines)

        # Deduplicate by headline text and cap at max
        seen = set()
        unique = []
        for item in collected:
            if item["headline"] not in seen:
                seen.add(item["headline"])
                unique.append(item)
            if len(unique) >= self.max_headlines:
                break

        return unique

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_keywords(self, company_name: str, symbol: str) -> list[str]:
        """Builds a list of search terms to match against headlines."""
        keywords = [company_name.lower().strip()]

        # Add individual words from multi-word company names (min 4 chars)
        for word in company_name.split():
            if len(word) >= 4:
                keywords.append(word.lower())

        if symbol:
            keywords.append(symbol.upper())
            keywords.append(symbol.lower())

        return list(set(keywords))

    def _fetch_feed(self, source: str, url: str, keywords: list[str]) -> list[dict]:
        """
        Fetches and parses a single RSS feed.
        Returns matching headlines as a list of dicts.
        Silently returns [] on any network or parse error.
        """
        try:
            request  = urllib.request.Request(url, headers=HEADERS)
            response = urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS)
            content  = response.read()
            root     = ET.fromstring(content)
        except Exception:
            return []

        results = []

        # RSS items are under channel > item
        for item in root.iter("item"):
            title       = item.findtext("title") or ""
            pub_date    = item.findtext("pubDate") or ""

            if not title:
                continue

            # Keep headline only if it mentions the company
            if any(kw in title.lower() for kw in keywords):
                results.append({
                    "headline":  title.strip(),
                    "source":    source,
                    "published": self._format_date(pub_date),
                })

        return results

    def _format_date(self, raw_date: str) -> str:
        """Converts RSS pubDate string to a clean readable format."""
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(raw_date.strip(), fmt)
                return dt.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                continue
        return raw_date.strip()


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    scraper = NewsScraper()

    test_companies = [
        ("Apple", "AAPL"),
        ("Tesla", "TSLA"),
        ("Microsoft", "MSFT"),
    ]

    for company, symbol in test_companies:
        print(f"\n🔍 Searching headlines for: {company} ({symbol})")
        headlines = scraper.fetch_headlines(company, symbol)

        if not headlines:
            print("  ⚠️  No headlines found — sentiment section will be skipped in report.")
        else:
            for h in headlines:
                print(f"  [{h['source']}] {h['headline']} ({h['published']})")
