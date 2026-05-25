"""
BrAPI client for batch quotes and historical series (brapi.dev).

Primary consumers: ``production_brapi.BrAPIProductionRunner`` and
``run_production.ProductionRunner``. Uses on-disk caches (see ``CACHE_DIR`` in
this module, typically under ``src/data/cache``) — distinct from ``src.brapi_client``
(``price_cache.json``) used by ``production_simple``.
"""

import requests
import time
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
from datetime import datetime, timedelta
import pandas as pd


class BrAPIClient:
    """Client for brapi.dev API - Brazilian stock market data."""
    
    BASE_URL = "https://brapi.dev/api"
    CACHE_DIR = Path(__file__).parent.parent / "data" / "cache"
    
    def __init__(self, api_key: Optional[str] = None, cache_ttl: int = 300):
        """
        Initialize BrAPI client.
        
        Args:
            api_key: Optional API key for premium features
            cache_ttl: Cache time-to-live in seconds (default 5 min)
        """
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.CACHE_DIR / f"{cache_key}.json"
    
    def _load_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached data if still valid."""
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            cached_at = data.get('cached_at', 0)
            if time.time() - cached_at < self.cache_ttl:
                return data.get('data')
        except:
            pass
        
        return None
    
    def _save_cache(self, cache_key: str, data: Any) -> None:
        """Save data to cache."""
        cache_path = self._get_cache_path(cache_key)
        with open(cache_path, 'w') as f:
            json.dump({
                'cached_at': time.time(),
                'data': data
            }, f)
    
    def get_quotes(self, tickers: List[str], use_cache: bool = True) -> Dict[str, Dict]:
        """
        Get current quotes for multiple tickers.
        
        Args:
            tickers: List of ticker symbols (e.g., ['PETR4', 'VALE3'])
            use_cache: Whether to use cached data
            
        Returns:
            Dict mapping ticker to quote data
        """
        if not tickers:
            return {}
        
        # Clean tickers (remove .SA suffix if present)
        tickers = [t.replace('.SA', '') for t in tickers]
        
        # Check cache for all tickers
        results = {}
        uncached_tickers = []
        
        if use_cache:
            for ticker in tickers:
                cached = self._load_cache(f"quote_{ticker}")
                if cached:
                    results[ticker] = cached
                else:
                    uncached_tickers.append(ticker)
        else:
            uncached_tickers = tickers
        
        if not uncached_tickers:
            return results
        
        # BrAPI free tier allows only 1 ticker per request
        # Fetch individually with rate-limit-aware delays
        for i, ticker in enumerate(uncached_tickers):
            # Delay between requests (except first)
            if i > 0:
                time.sleep(1.0)
            
            url = f"{self.BASE_URL}/quote/{ticker}"
            
            # Retry logic for server/rate-limit errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.get(url, headers=self._get_headers(), timeout=15)
                    
                    if response.status_code == 429:
                        # Rate limited — exponential backoff
                        if attempt < max_retries - 1:
                            wait = 5 * (2 ** attempt)
                            time.sleep(wait)
                            continue
                        # Give up on this ticker after retries
                    
                    if response.status_code == 502:
                        # Server error, retry with delay
                        if attempt < max_retries - 1:
                            time.sleep(1 + attempt)
                            continue
                    
                    if response.status_code == 400:
                        # Plan limit (e.g. free tier batch not allowed) — no retry
                        try:
                            body = response.json()
                            if body.get("code") == "QUOTES_PER_REQUEST_EXCEEDED":
                                break
                        except ValueError:
                            pass
                    
                    response.raise_for_status()
                    data = response.json()
                    
                    for result in data.get('results', []):
                        sym = result.get('symbol')
                        if sym:
                            results[sym] = result
                            if use_cache:
                                self._save_cache(f"quote_{sym}", result)
                    break  # Success
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        pass  # Silently skip failed batches
                    else:
                        time.sleep(1 + attempt)
        
        return results
    
    def get_quote(self, ticker: str, use_cache: bool = True) -> Optional[Dict]:
        """Get current quote for a single ticker."""
        quotes = self.get_quotes([ticker], use_cache)
        return quotes.get(ticker.replace('.SA', ''))
    
    def get_historical(
        self, 
        ticker: str, 
        range_: str = "1y",
        interval: str = "1d",
        use_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get historical price data for a ticker.
        
        Args:
            ticker: Stock ticker (e.g., 'PETR4')
            range_: Time range ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            
        Returns:
            DataFrame with OHLCV data
        """
        ticker = ticker.replace('.SA', '')
        cache_key = f"hist_{ticker}_{range_}_{interval}"
        
        # Check cache
        if use_cache:
            cached = self._load_cache(cache_key)
            if cached:
                df = pd.DataFrame(cached)
                if not df.empty and 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    return df
        
        # Fetch historical data
        url = f"{self.BASE_URL}/quote/{ticker}"
        params = {
            'range': range_,
            'interval': interval
        }
        
        try:
            response = requests.get(url, headers=self._get_headers(), params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract historical data
            results = data.get('results', [])
            if results and len(results) > 0:
                hist_data = results[0] if isinstance(results, list) else results
                
                # Historical data is in historicalDataPrice array
                ohlcv = hist_data.get('historicalDataPrice', [])
                
                if ohlcv:
                    df = pd.DataFrame(ohlcv)
                    
                    # Ensure we have the expected columns
                    if not df.empty and 'date' in df.columns:
                        # Convert Unix timestamp to datetime
                        df['date'] = pd.to_datetime(df['date'], unit='s')
                        
                        # Rename columns to match expected format
                        column_map = {
                            'open': 'Open',
                            'high': 'High',
                            'low': 'Low',
                            'close': 'Close',
                            'volume': 'Volume',
                            'adjustedClose': 'Adj Close'
                        }
                        df.rename(columns=column_map, inplace=True)
                        
                        if use_cache:
                            # Convert timestamps to strings for JSON serialization
                            df_to_cache = df.copy()
                            df_to_cache['date'] = df_to_cache['date'].astype(str)
                            self._save_cache(cache_key, df_to_cache.to_dict('records'))
                        
                        df.set_index('date', inplace=True)
                        return df
            
        except Exception as e:
            print(f"Error fetching historical data for {ticker}: {e}")
        
        return None
    
    def get_available_tickers(self) -> List[str]:
        """Get list of available tickers."""
        url = f"{self.BASE_URL}/available"
        
        try:
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get('stocks', [])
        except Exception as e:
            print(f"Error fetching available tickers: {e}")
            return []


class BrapiPriceProvider:
    """
    Price provider for stock-signals system using BrAPI.
    Compatible with the existing PriceProvider interface.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.client = BrAPIClient(api_key=api_key)
    
    def get_price(self, ticker: str) -> Optional[float]:
        """Get current price for a ticker."""
        quote = self.client.get_quote(ticker)
        if quote:
            return quote.get('regularMarketPrice')
        return None
    
    def get_prices(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        """Get current prices for multiple tickers."""
        quotes = self.client.get_quotes(tickers)
        return {
            ticker: quote.get('regularMarketPrice') 
            for ticker, quote in quotes.items()
        }
    
    def get_ohlcv(self, ticker: str, days: int = 365) -> Optional[pd.DataFrame]:
        """Get OHLCV data for technical analysis."""
        range_map = {
            30: "1mo",
            90: "3mo",
            180: "6mo",
            365: "1y",
            730: "2y",
        }
        range_ = range_map.get(days, "1y")
        return self.client.get_historical(ticker, range_=range_)
    
    def get_quote_data(self, ticker: str) -> Optional[Dict]:
        """Get full quote data including 52-week range, P/E, etc."""
        return self.client.get_quote(ticker)
    
    def get_batch_quote_data(self, tickers: List[str]) -> Dict[str, Dict]:
        """Get full quote data for multiple tickers."""
        return self.client.get_quotes(tickers)


# Test function
if __name__ == "__main__":
    client = BrAPIClient()
    
    # Test single quote
    print("Testing single quote...")
    quote = client.get_quote("PETR4")
    if quote:
        print(f"  PETR4: R$ {quote['regularMarketPrice']:.2f} ({quote['regularMarketChangePercent']:+.2f}%)")
        print(f"  52-week range: {quote['fiftyTwoWeekRange']}")
        print(f"  P/E: {quote.get('priceEarnings', 'N/A'):.2f}" if quote.get('priceEarnings') else "  P/E: N/A")
    
    # Test batch quotes
    print("\nTesting batch quotes...")
    tickers = ["PETR4", "VALE3", "ITUB4", "BBDC4", "ABEV3"]
    quotes = client.get_quotes(tickers)
    for ticker, q in quotes.items():
        print(f"  {ticker}: R$ {q['regularMarketPrice']:.2f} ({q['regularMarketChangePercent']:+.2f}%)")
    
    # Test historical data
    print("\nTesting historical data...")
    df = client.get_historical("PETR4", range_="1mo")
    if df is not None and not df.empty:
        print(f"  Got {len(df)} days of data")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Last 3 rows:\n{df.tail(3)}")
    
    print("\n✅ BrAPI client working!")
