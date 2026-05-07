#!/usr/bin/env python3
"""
BrAPI quote helper with file cache (``price_cache.json`` at repo root).

Used by ``production_simple.SimpleProductionRunner`` for spot prices while
historical candles come from yfinance. For the BrAPI-first historical client,
see ``src.data.brapi_client``.
"""

import json
import logging
import math
import os
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf


CACHE_FILE = Path(__file__).parent.parent / "price_cache.json"
DEFAULT_TTL_SECONDS = 60 * 60  # 60 minutes for free-tier resilience
BRAZIL_TZ = ZoneInfo("America/Sao_Paulo")
MARKET_OPEN = time(10, 0)
MARKET_CLOSE = time(17, 15)
FRESH_PRICE_SECONDS = 60 * 60  # 60 minutes to tolerate Yahoo delays during market hours

logger = logging.getLogger(__name__)


class BrAPIClient:
    """Client for brapi.dev API - real-time Brazilian stock data with caching."""

    BASE_URL = "https://brapi.dev/api/quote"
    _quota_exceeded = False

    def __init__(self, api_key: str = None, timeout: int = 30,
                 cache_ttl: int = DEFAULT_TTL_SECONDS):
        self.api_key = api_key or os.environ.get("BRAPI_API_KEY", "")
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._cache = self._load_cache()

    # -- Cache helpers --

    @staticmethod
    def _strip_sa_suffix(ticker: str) -> str:
        return ticker[:-3] if ticker.endswith(".SA") else ticker

    @classmethod
    def _to_yfinance_ticker(cls, ticker: str) -> str:
        base_ticker = cls._strip_sa_suffix(ticker.upper())
        return f"{base_ticker}.SA"

    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _now_brazil() -> datetime:
        return datetime.now(BRAZIL_TZ)

    def is_market_hours(self, current_time: Optional[datetime] = None) -> bool:
        current_time = current_time.astimezone(BRAZIL_TZ) if current_time else self._now_brazil()
        if current_time.weekday() >= 5:
            return False
        return MARKET_OPEN <= current_time.time() <= MARKET_CLOSE

    @staticmethod
    def _is_price_reasonable(price: Optional[float]) -> bool:
        if price is None:
            return False
        try:
            price = float(price)
        except (TypeError, ValueError):
            return False
        return math.isfinite(price) and price > 0.0

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        if value in (None, ""):
            return None
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            except (TypeError, ValueError, OSError):
                return None
        if isinstance(value, str):
            try:
                cleaned = value.replace("Z", "+00:00")
                parsed = datetime.fromisoformat(cleaned)
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        return None

    def _is_quote_fresh(self, quote_time: Optional[datetime], allow_stale: bool = False) -> bool:
        """
        Check if quote is fresh enough for current market conditions.

        Args:
            quote_time: Timestamp of the quote
            allow_stale: If True, accept stale quotes (used when BrAPI quota exceeded)

        Returns:
            True if quote is fresh enough, False otherwise
        """
        if allow_stale:
            return True  # Accept any quote when explicitly allowed
        if quote_time is None:
            return not self.is_market_hours()
        if not self.is_market_hours():
            return True
        age_seconds = (self._now_utc() - quote_time.astimezone(timezone.utc)).total_seconds()
        return age_seconds <= FRESH_PRICE_SECONDS

    @classmethod
    def _set_quota_exceeded(cls, value: bool) -> None:
        cls._quota_exceeded = value

    def _is_quota_response(self, response: requests.Response) -> bool:
        body = ""
        try:
            payload = response.json()
            body = json.dumps(payload).lower()
        except ValueError:
            body = response.text.lower()

        quota_markers = (
            "quota",
            "monthly limit",
            "rate limit",
            "request limit",
            "limit exceeded",
            "too many requests",
            "credits",
        )
        return response.status_code == 429 or any(marker in body for marker in quota_markers)

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

    def _cache_get_entry(self, ticker: str, allow_stale: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get cached entry for ticker.

        Args:
            ticker: Stock ticker (without .SA suffix)
            allow_stale: If True, return entry even if stale (used when BrAPI quota exceeded)

        Returns:
            Cached entry dict or None
        """
        ticker = self._strip_sa_suffix(ticker.upper())
        entry = self._cache.get(ticker)
        if entry is None:
            return None
        if not self._is_price_reasonable(entry.get("price")):
            return None
        cached_ts = float(entry.get("ts", 0) or 0)
        now = self._now_utc().timestamp()
        if now - cached_ts > self.cache_ttl:
            return None

        # Skip freshness check when explicitly allowing stale quotes
        if not allow_stale:
            quote_time = self._parse_timestamp(entry.get("quote_ts")) or self._parse_timestamp(cached_ts)
            if not self._is_quote_fresh(quote_time):
                return None

        return dict(entry)

    def _cache_get(self, ticker: str) -> Optional[float]:
        entry = self._cache_get_entry(ticker)
        if entry is None:
            return None
        return float(entry["price"])

    def _cache_put(self, ticker: str, price: float, source: str = "brapi",
                   quote_ts: Any = None, is_stale: bool = False):
        ticker = self._strip_sa_suffix(ticker.upper())
        now = self._now_utc()
        quote_time = self._parse_timestamp(quote_ts) or now
        self._cache[ticker] = {
            "price": price,
            "ts": now.timestamp(),
            "updated": now.isoformat(),
            "quote_ts": quote_time.isoformat(),
            "source": source,
            "is_stale": is_stale,
        }

    def get_price_snapshot(self, ticker: str, allow_stale: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get price snapshot for ticker.

        Args:
            ticker: Stock ticker (without .SA suffix)
            allow_stale: If True, return snapshot even if stale

        Returns:
            Snapshot dict with price, source, quote_ts, is_stale, or None
        """
        return self._cache_get_entry(ticker, allow_stale=allow_stale)

    # -- API methods --

    def _build_url(self, tickers: str) -> str:
        url = f"{self.BASE_URL}/{tickers}"
        if self.api_key:
            url += f"?token={self.api_key}"
        return url

    def is_quota_exceeded(self) -> bool:
        return type(self)._quota_exceeded

    @staticmethod
    def _extract_brapi_quote(result: Dict[str, Any]) -> Tuple[Optional[float], Optional[Any]]:
        price = result.get("regularMarketPrice")
        quote_time = (
            result.get("regularMarketTime")
            or result.get("regularMarketDate")
            or result.get("updatedAt")
        )
        return price, quote_time

    def _request_brapi(self, tickers: str) -> Optional[Dict[str, Any]]:
        url = self._build_url(tickers)
        response = requests.get(url, timeout=self.timeout)
        if not response.ok:
            if self._is_quota_response(response):
                self._set_quota_exceeded(True)
                logger.warning("BrAPI quota exhausted for %s; switching to Yahoo Finance fallback", tickers)
            response.raise_for_status()

        self._set_quota_exceeded(False)
        return response.json()

    @staticmethod
    def _extract_latest_close(close_data: Any, yf_ticker: str) -> Tuple[Optional[float], Optional[datetime]]:
        if close_data is None:
            return None, None

        if isinstance(close_data, pd.Series):
            series = close_data.dropna()
        elif isinstance(close_data, pd.DataFrame):
            if yf_ticker not in close_data.columns:
                return None, None
            series = close_data[yf_ticker].dropna()
        else:
            return None, None

        if series.empty:
            return None, None

        price = series.iloc[-1]
        if pd.isna(price):
            return None, None

        last_index = series.index[-1]
        if isinstance(last_index, pd.Timestamp):
            quote_time = last_index.to_pydatetime()
            if quote_time.tzinfo is None:
                quote_time = quote_time.replace(tzinfo=BRAZIL_TZ).astimezone(timezone.utc)
        else:
            quote_time = None

        return float(price), quote_time

    def _fallback_info_price(self, ticker: str, allow_stale: bool = False) -> Optional[Tuple[float, Optional[datetime], bool]]:
        """
        Get fallback price from yfinance ticker info.

        Args:
            ticker: Stock ticker (without .SA suffix)
            allow_stale: If True, accept stale quotes and mark them as such

        Returns:
            Tuple of (price, quote_time, is_stale) or None if failed
        """
        yf_ticker = self._to_yfinance_ticker(ticker)
        stock = yf.Ticker(yf_ticker)

        price = None
        quote_time = None
        try:
            info = stock.info or {}
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            quote_time = self._parse_timestamp(info.get("regularMarketTime"))
        except Exception as exc:
            logger.warning("Yahoo info fallback failed for %s: %s", ticker, exc)
            return None

        if not self._is_price_reasonable(price):
            return None

        is_stale = not self._is_quote_fresh(quote_time, allow_stale=False)
        if is_stale and not allow_stale:
            logger.warning("Yahoo info price for %s is stale; ignoring fallback quote", ticker)
            return None

        return float(price), quote_time, is_stale

    def get_fallback_prices(self, tickers: List[str], allow_stale: bool = False) -> Dict[str, float]:
        """
        Fetch prices using yfinance fallback.

        Args:
            tickers: List of stock tickers (without .SA suffix)
            allow_stale: If True, accept stale quotes (previous day's close)

        Returns:
            Dict mapping ticker -> price
        """
        prices: Dict[str, float] = {}
        requested = [self._strip_sa_suffix(ticker.upper()) for ticker in tickers]
        pending = []

        for ticker in requested:
            cached = self._cache_get_entry(ticker)
            if cached is not None:
                prices[ticker] = float(cached["price"])
            else:
                pending.append(ticker)

        if not pending:
            return prices

        yf_tickers = [self._to_yfinance_ticker(ticker) for ticker in pending]

        try:
            data = yf.download(
                tickers=yf_tickers,
                period="1d",
                interval="1m",
                progress=False,
                auto_adjust=False,
                prepost=False,
                threads=True,
            )

            close_data = data.get("Close") if isinstance(data, pd.DataFrame) else None
            if close_data is not None:
                for ticker, yf_ticker in zip(pending, yf_tickers):
                    price, quote_time = self._extract_latest_close(close_data, yf_ticker)
                    if not self._is_price_reasonable(price):
                        continue
                    is_stale = not self._is_quote_fresh(quote_time, allow_stale=False)
                    if is_stale and not allow_stale:
                        logger.warning("Yahoo intraday quote for %s is older than 15 minutes; ignoring it", ticker)
                        continue
                    prices[ticker] = float(price)
                    self._cache_put(ticker, float(price), source="yfinance", quote_ts=quote_time, is_stale=is_stale)
        except Exception as exc:
            logger.warning("Yahoo batch fallback failed for %s: %s", ",".join(pending), exc)

        missing = [ticker for ticker in pending if ticker not in prices]
        for ticker in missing:
            fallback = self._fallback_info_price(ticker, allow_stale=allow_stale)
            if fallback is None:
                continue
            price, quote_time, is_stale = fallback
            prices[ticker] = price
            self._cache_put(ticker, price, source="yfinance_info", quote_ts=quote_time, is_stale=is_stale)

        if pending:
            self._save_cache()

        return prices

    def get_fallback_price(self, ticker: str, allow_stale: bool = False) -> Optional[float]:
        return self.get_fallback_prices([ticker], allow_stale=allow_stale).get(self._strip_sa_suffix(ticker.upper()))

    def get_price(self, ticker: str) -> Optional[float]:
        """
        Fetch current price for a single ticker (cache-aware).

        Args:
            ticker: Stock ticker without .SA suffix (e.g., 'PETR4')

        Returns:
            Current price or None if fetch fails
        """
        ticker = self._strip_sa_suffix(ticker.upper())
        cached = self._cache_get(ticker)
        if cached is not None:
            return cached

        if not self.is_quota_exceeded():
            try:
                data = self._request_brapi(ticker)
                if data and data.get("results"):
                    price, quote_time = self._extract_brapi_quote(data["results"][0])
                    if self._is_price_reasonable(price) and self._is_quote_fresh(self._parse_timestamp(quote_time)):
                        self._cache_put(ticker, float(price), source="brapi", quote_ts=quote_time, is_stale=False)
                        self._save_cache()
                        return float(price)
                    logger.warning("BrAPI returned invalid or stale quote for %s; using fallback", ticker)
            except Exception as exc:
                logger.warning("BrAPI error for %s: %s", ticker, exc)

        # When quota exceeded, accept stale quotes
        allow_stale = self.is_quota_exceeded()
        fallback_price = self.get_fallback_price(ticker, allow_stale=allow_stale)
        if fallback_price is not None:
            if allow_stale:
                logger.warning("Using stale Yahoo Finance fallback quote for %s (previous close)", ticker)
        return fallback_price

    def get_prices(self, tickers: List[str], batch_size: int = 1) -> Dict[str, float]:
        """
        Fetch prices for multiple tickers, using cache where possible.
        Uses batch API to fetch multiple tickers per request (more efficient).
        
        Args:
            tickers: List of stock tickers (without .SA suffix)
            batch_size: Number of tickers per batch request (default 1 for free-tier limit)

        Returns:
            Dict mapping ticker -> price
        """
        prices: Dict[str, float] = {}
        need_fetch: List[str] = []

        for t in tickers:
            clean_ticker = self._strip_sa_suffix(t.upper())
            cached = self._cache_get(clean_ticker)
            if cached is not None:
                prices[clean_ticker] = cached
            else:
                need_fetch.append(clean_ticker)

        if not need_fetch:
            return prices

        import time

        if not self.is_quota_exceeded():
            for i in range(0, len(need_fetch), batch_size):
                batch = need_fetch[i:i + batch_size]
                batch_str = ",".join(batch)

                try:
                    data = self._request_brapi(batch_str)
                    if data and data.get("results"):
                        for result in data["results"]:
                            ticker = result.get("symbol")
                            price, quote_time = self._extract_brapi_quote(result)
                            if not ticker or not self._is_price_reasonable(price):
                                continue
                            if not self._is_quote_fresh(self._parse_timestamp(quote_time)):
                                logger.warning("BrAPI quote for %s is older than 15 minutes; ignoring it", ticker)
                                continue
                            prices[ticker] = float(price)
                            self._cache_put(ticker, float(price), source="brapi", quote_ts=quote_time)
                except Exception as exc:
                    logger.warning("BrAPI batch failed for %s: %s", batch_str, exc)
                    if self.is_quota_exceeded():
                        break

                if i + batch_size < len(need_fetch):
                    time.sleep(2)

        missing = [ticker for ticker in need_fetch if ticker not in prices]
        if missing:
            # When BrAPI quota is exceeded, accept stale quotes (previous close)
            allow_stale = self.is_quota_exceeded()
            if allow_stale:
                logger.warning("BrAPI quota exceeded; accepting stale quotes (previous close) for %d tickers", len(missing))
            prices.update(self.get_fallback_prices(missing, allow_stale=allow_stale))

        self._save_cache()
        return prices

    def get_spot_price(self, ticker: str) -> Optional[float]:
        return self.get_price(ticker)

    def get_batch_prices(self, tickers: List[str], batch_size: int = 1) -> Dict[str, float]:
        return self.get_prices(tickers, batch_size=batch_size)

    def get_ticker_info(self, ticker: str) -> Optional[Dict]:
        """Get detailed ticker information."""
        try:
            data = self._request_brapi(self._strip_sa_suffix(ticker.upper()))
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
