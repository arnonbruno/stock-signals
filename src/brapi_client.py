#!/usr/bin/env python3
"""
BrAPI Price Fetcher with Caching
Fetches real-time prices for Brazilian stocks using brapi.dev API.
Includes file-based caching with configurable TTL to respect rate limits.
"""

import os
import json
import requests
from typing import Dict, List, Optional
from datetime import datetime, timezone
from pathlib import Path


CACHE_FILE = Path(__file__).parent.parent / "price_cache.json"
DEFAULT_TTL_SECONDS = 15 * 60  # 15 minutes


class BrAPIClient:
    """Client for brapi.dev API - real-time Brazilian stock data with caching."""

    BASE_URL = "https://brapi.dev/api/quote"

    def __init__(self, api_key: str = None, timeout: int = 30,
                 cache_ttl: int = DEFAULT_TTL_SECONDS):
        self.api_key = api_key or os.environ.get("BRAPI_API_KEY", "")
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._cache = self._load_cache()

    # -- Cache helpers --

    def _load_cache(self) -> Dict:
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {}

    def _save_cache(self):
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump(self._cache, f, indent=2)
        except OSError as e:
            print(f"⚠️ Could not write price cache: {e}")

    def _cache_get(self, ticker: str) -> Optional[float]:
        entry = self._cache.get(ticker)
        if entry is None:
            return None
        cached_ts = entry.get("ts", 0)
        now = datetime.now(timezone.utc).timestamp()
        if now - cached_ts > self.cache_ttl:
            return None
        return entry.get("price")

    def _cache_put(self, ticker: str, price: float):
        self._cache[ticker] = {
            "price": price,
            "ts": datetime.now(timezone.utc).timestamp(),
            "updated": datetime.now(timezone.utc).isoformat(),
        }

    # -- API methods --

    def _build_url(self, tickers: str) -> str:
        url = f"{self.BASE_URL}/{tickers}"
        if self.api_key:
            url += f"?token={self.api_key}"
        return url

    def get_price(self, ticker: str) -> Optional[float]:
        """
        Fetch current price for a single ticker (cache-aware).

        Args:
            ticker: Stock ticker without .SA suffix (e.g., 'PETR4')

        Returns:
            Current price or None if fetch fails
        """
        cached = self._cache_get(ticker)
        if cached is not None:
            return cached

        try:
            url = self._build_url(ticker)
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            if data.get("results") and len(data["results"]) > 0:
                price = float(data["results"][0]["regularMarketPrice"])
                self._cache_put(ticker, price)
                self._save_cache()
                return price
        except Exception as e:
            print(f"❌ BrAPI error for {ticker}: {e}")

        return None

    def get_prices(self, tickers: List[str]) -> Dict[str, float]:
        """
        Fetch prices for multiple tickers, using cache where possible.
        Fetches one ticker at a time (free-plan limit) with cache to minimize calls.

        Args:
            tickers: List of stock tickers (without .SA suffix)

        Returns:
            Dict mapping ticker -> price
        """
        prices: Dict[str, float] = {}
        need_fetch: List[str] = []

        for t in tickers:
            cached = self._cache_get(t)
            if cached is not None:
                prices[t] = cached
            else:
                need_fetch.append(t)

        if not need_fetch:
            return prices

        import time
        for i, ticker in enumerate(need_fetch):
            try:
                url = self._build_url(ticker)
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()

                data = response.json()
                if data.get("results") and len(data["results"]) > 0:
                    price = float(data["results"][0]["regularMarketPrice"])
                    prices[ticker] = price
                    self._cache_put(ticker, price)

            except Exception as e:
                print(f"❌ BrAPI error for {ticker}: {e}")

            # Small delay between requests to respect rate limits
            if i < len(need_fetch) - 1:
                time.sleep(0.3)

        self._save_cache()
        return prices

    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        """Get detailed ticker information."""
        try:
            url = self._build_url(ticker)
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            if data.get("results") and len(data["results"]) > 0:
                return data["results"][0]
        except Exception as e:
            print(f"❌ BrAPI info error for {ticker}: {e}")

        return None


def main():
    """Test the BrAPI client."""
    client = BrAPIClient()

    print("Fetching PETR4...")
    price = client.get_price("PETR4")
    if price:
        print(f"PETR4: R$ {price:.2f}")
    else:
        print("PETR4: failed to fetch")

    print("\nFetching batch...")
    tickers = ["PETR4", "VALE3", "GGBR4", "GOAU4", "ECOR3"]
    prices = client.get_prices(tickers)
    for ticker, p in prices.items():
        print(f"{ticker}: R$ {p:.2f}")


if __name__ == "__main__":
    main()
