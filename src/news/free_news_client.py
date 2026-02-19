#!/usr/bin/env python3
"""
Free news client for historical and real-time stock market news.

Sources:
1. Google News RSS - Real-time news (last 30 days)
2. newsdata.io API - Brazilian financial news (with API key)
3. FinBERT - Local sentiment analysis (specialized for finance)

Features:
- Free historical news data for backtesting
- Real-time news for live trading
- Local sentiment analysis (no API costs)
- Caching to avoid redundant requests
"""

import logging
import time
import requests
import feedparser
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from functools import lru_cache
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
import os
from urllib.parse import quote_plus
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from stock-signals directory
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)

from .news_cache import NewsCache


logger = logging.getLogger(__name__)


class FinBERTSentimentAnalyzer:
    """
    Local sentiment analyzer using FinBERT (specialized for financial text).
    
    FinBERT is a pre-trained NLP model for financial sentiment analysis.
    Download happens once, then runs locally (no API calls).
    """
    
    def __init__(self):
        """Initialize FinBERT model (downloads on first run)."""
        logger.info("Loading FinBERT model (may take a moment on first run)...")
        
        model_name = "ProsusAI/finbert"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()  # Set to evaluation mode
            
            logger.info("FinBERT loaded successfully")
        
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            logger.warning("Falling back to simple rule-based sentiment")
            self.tokenizer = None
            self.model = None
    
    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of financial text using ensemble approach.
        
        Ensemble combines:
        1. FinBERT (primary, specialized for finance)
        2. Simple lexicon fallback (when FinBERT fails)
        
        Args:
            text: Financial news text
        
        Returns:
            Sentiment score (-1.0 to +1.0)
        """
        if not text or not text.strip():
            return 0.0
        
        # Fallback to rule-based if model failed to load
        if self.model is None:
            return self._simple_sentiment(text)
        
        try:
            # Tokenize
            inputs = self.tokenizer(text, return_tensors="pt", 
                                   truncation=True, max_length=512, 
                                   padding=True)
            
            # Run model
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: [positive, negative, neutral]
            probs = predictions[0].tolist()
            positive, negative, neutral = probs
            
            # Convert to -1 to +1 scale
            finbert_sentiment = positive - negative
            
            # ENSEMBLE: Blend FinBERT with simple lexicon
            # Weight FinBERT higher (0.7) vs lexicon (0.3)
            lexicon_sentiment = self._simple_sentiment(text)
            
            # Weighted average
            ensemble_sentiment = 0.7 * finbert_sentiment + 0.3 * lexicon_sentiment
            
            return ensemble_sentiment
        
        except Exception as e:
            logger.warning(f"FinBERT analysis failed: {e}, using fallback")
            return self._simple_sentiment(text)
    
    def _simple_sentiment(self, text: str) -> float:
        """
        Enhanced lexicon-based sentiment (fallback when FinBERT fails).
        
        Uses expanded financial lexicon based on Loughran-McDonald approach.
        Categories: Positive, Negative, Uncertainty, Litigious, Strong Modal, Weak Modal
        """
        text_lower = text.lower()
        
        # POSITIVE WORDS (Portuguese financial context)
        positive_words = {
            # Performance
            'alta': 1.5, 'subiu': 1.3, 'crescimento': 1.4, 'lucro': 1.6, 
            'ganho': 1.4, 'valorização': 1.5, 'recorde': 1.7, 'alta histórico': 1.8,
            # Optimism
            'otimista': 1.2, 'positivo': 1.1, 'sucesso': 1.3, 'expansão': 1.4,
            'crescer': 1.3, 'superou': 1.5, 'melhor': 1.2, 'superar': 1.4,
            # Financial strength
            'dividendo': 1.3, 'resultado': 1.0, 'receita': 1.1, 'faturamento': 1.1,
            'rentabilidade': 1.4, 'margem': 1.0, 'eficiência': 1.2,
            # Market sentiment
            'compra': 0.8, 'demanda': 1.0, 'interesse': 0.7, 'oportunidade': 1.1
        }
        
        # NEGATIVE WORDS (Portuguese financial context)
        negative_words = {
            # Performance decline
            'queda': -1.5, 'caiu': -1.3, 'prejuízo': -1.7, 'perda': -1.6,
            'desvalorização': -1.5, 'baixa': -1.2, 'recuo': -1.1, 'negativo': -1.2,
            # Pessimism
            'pessimista': -1.3, 'risco': -1.0, 'crise': -1.6, 'recessão': -1.7,
            'instabilidade': -1.3, 'incerteza': -1.2, 'preocupação': -1.1,
            # Financial weakness
            'divida': -1.2, 'endividamento': -1.4, 'inadimplência': -1.6,
            'redução': -0.8, 'corte': -1.0, 'fechamento': -0.9,
            # Market sentiment
            'venda': -0.7, 'pressão': -0.9, 'volatilidade': -0.8
        }
        
        # UNCERTAINTY WORDS (moderate impact)
        uncertainty_words = {
            'pode': -0.3, 'possível': -0.2, 'provável': -0.2, 'talvez': -0.3,
            'incerto': -0.5, 'imprevisível': -0.6, 'variável': -0.3
        }
        
        # Calculate weighted sentiment scores
        pos_score = sum(weight for word, weight in positive_words.items() if word in text_lower)
        neg_score = sum(weight for word, weight in negative_words.items() if word in text_lower)
        uncertainty_score = sum(weight for word, weight in uncertainty_words.items() if word in text_lower)
        
        # Total sentiment
        total_score = pos_score + neg_score + uncertainty_score
        
        # Normalize to -1 to +1 range
        if total_score == 0:
            return 0.0
        
        # Soft normalization (preserves relative strength)
        normalized = total_score / (abs(total_score) + 3)  # +3 prevents extreme values
        
        return max(-1.0, min(1.0, normalized))


class FreeNewsClient:
    """Client for free historical and real-time financial news."""
    
    # Company name mapping for Brazilian stocks
    TICKER_MAP = {
        'PETR4.SA': 'Petrobras',
        'PETR3.SA': 'Petrobras',
        'VALE3.SA': 'Vale',
        'ITUB4.SA': 'Itaú',
        'BBDC4.SA': 'Bradesco',
        'BBAS3.SA': 'Banco do Brasil',
        'ABEV3.SA': 'Ambev',
        'B3SA3.SA': 'B3',
        'SUZB3.SA': 'Suzano',
        'RENT3.SA': 'Localiza',
        'WEGE3.SA': 'WEG',
        'MGLU3.SA': 'Magazine Luiza',
        'PCAR3.SA': 'Pão de Açúcar',
        'LREN3.SA': 'Lojas Renner',
        'RAIZ4.SA': 'Raízen',
        'GGBR4.SA': 'Gerdau',
        'ASAI3.SA': 'Assaí',
        'JBSS3.SA': 'JBS',
        'RDOR3.SA': 'Rede D\'Or',
    }
    
    def __init__(self, cache_ttl_hours: int = 6, cache_file: str = None):
        """Initialize free news client with caching.
        
        Args:
            cache_ttl_hours: Cache time-to-live in hours
            cache_file: Optional custom cache file path (for testing)
        """
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = FinBERTSentimentAnalyzer()
        
        # Initialize news cache (reduces API calls, leaves 50 credits headroom)
        cache_kwargs = {'ttl_hours': cache_ttl_hours}
        if cache_file:
            cache_kwargs['cache_file'] = cache_file
        self.cache = NewsCache(**cache_kwargs)
        
        self.last_request_time = 0
        self.min_request_interval = 2.0  # Be polite: 2 seconds between requests
        
        logger.info(f"FreeNewsClient initialized (cache TTL: {cache_ttl_hours}h)")
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _ticker_to_query(self, ticker: str) -> str:
        """Convert ticker to company name."""
        return self.TICKER_MAP.get(ticker, ticker.replace('.SA', ''))
    
    def _fetch_google_news(self, query: str, days: int = 7) -> List[Dict]:
        """
        Fetch news from Google News RSS.
        
        Args:
            query: Search query (company name)
            days: Days to look back (1-30)
        
        Returns:
            List of news articles
        """
        self._rate_limit()
        
        # URL encode the query to handle spaces and special characters
        encoded_query = quote_plus(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}+when:{days}d&hl=pt-BR&gl=BR&ceid=BR:pt-419"
        
        logger.debug(f"Fetching Google News for '{query}' (last {days} days)")
        
        try:
            feed = feedparser.parse(url)
            
            articles = []
            for entry in feed.entries:
                articles.append({
                    'title': entry.get('title', ''),
                    'link': entry.get('link', ''),
                    'published': entry.get('published', ''),
                    'summary': entry.get('summary', ''),
                    'source': 'google_news'
                })
            
            logger.info(f"Found {len(articles)} articles from Google News for '{query}'")
            return articles
        
        except Exception as e:
            logger.error(f"Failed to fetch Google News: {e}")
            return []
    
    def _fetch_newsdata_io(self, ticker: str, date: str) -> List[Dict]:
        """
        Fetch news from newsdata.io API with smart filtering.
        
        Strategy (in order):
        1. Market endpoint (financial news focused, may have limited coverage)
        2. Regular endpoint with qInTitle (ticker in title - more relevant)
        3. Regular endpoint with simple query + sports exclusion
        
        Args:
            ticker: Stock ticker (e.g., VALE3.SA or VALE3)
            date: Date string (YYYY-MM-DD)
        
        Returns:
            List of news articles
        """
        import os
        import time
        from urllib.parse import quote
        
        self._rate_limit()
        
        # Clean ticker for search
        ticker_search = ticker.replace('.SA', '')
        
        # SECURITY: Only use environment variable, no hardcoded fallback
        api_key = os.environ.get('NEWSDATA_API_KEY')
        if not api_key:
            logger.error("NEWSDATA_API_KEY environment variable not set - news fetching disabled")
            return []
        
        # Check API budget before making request
        from src.news.api_budget_tracker import get_budget_tracker
        budget_tracker = get_budget_tracker()
        
        if not budget_tracker.can_make_request():
            logger.warning(f"API budget exhausted - cannot fetch news for {ticker_search}")
            return []
        
        # Strategy 1: Try Market endpoint (financial news focused)
        market_url = "https://newsdata.io/api/1/market"
        market_params = {
            'q': ticker_search,
            'country': 'br',
            'language': 'pt',
            'apikey': api_key
        }
        
        logger.debug(f"Trying newsdata.io Market endpoint for '{ticker_search}'")
        articles = self._make_newsdata_request(market_url, market_params, ticker_search, budget_tracker)
        
        if articles:
            logger.info(f"Found {len(articles)} articles from newsdata.io Market for '{ticker_search}'")
            return articles
        
        # Strategy 2: Try qInTitle (ticker in title - more relevant)
        news_url = "https://newsdata.io/api/1/news"
        title_params = {
            'qInTitle': ticker_search,
            'country': 'br',
            'language': 'pt',
            'apikey': api_key
        }
        
        logger.debug(f"Trying newsdata.io qInTitle for '{ticker_search}'")
        articles = self._make_newsdata_request(news_url, title_params, ticker_search, budget_tracker)
        
        if articles:
            logger.info(f"Found {len(articles)} articles from newsdata.io (qInTitle) for '{ticker_search}'")
            return articles
        
        # Strategy 3: Simple query with sports exclusion (within 100 char limit)
        # Note: newsdata.io has 100 char query limit
        simple_query = f'{ticker_search} NOT futebol NOT esporte'
        simple_params = {
            'q': simple_query,
            'country': 'br',
            'language': 'pt',
            'apikey': api_key
        }
        
        logger.debug(f"Trying newsdata.io simple query for '{ticker_search}'")
        articles = self._make_newsdata_request(news_url, simple_params, ticker_search, budget_tracker)
        
        if articles:
            logger.info(f"Found {len(articles)} articles from newsdata.io (filtered) for '{ticker_search}'")
            return articles
        
        logger.info(f"No articles found in newsdata.io for '{ticker_search}'")
        return []
    
    def _make_newsdata_request(self, url: str, params: dict, ticker: str, budget_tracker) -> List[Dict]:
        """
        Make a request to newsdata.io with retry logic.
        
        Args:
            url: API endpoint URL
            params: Request parameters
            ticker: Ticker for logging
            budget_tracker: API budget tracker
        
        Returns:
            List of formatted articles
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=15)
                
                # Handle specific status codes
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    logger.warning(f"Rate limited (HTTP 429) for {ticker}. Retry after {retry_after}s")
                    if attempt < max_retries - 1:
                        time.sleep(retry_after)
                        continue
                    return []
                
                elif response.status_code == 401:
                    logger.error("API key invalid (HTTP 401) - check NEWSDATA_API_KEY")
                    return []
                
                elif response.status_code == 403:
                    logger.error("API key quota exceeded or access forbidden (HTTP 403)")
                    return []
                
                elif response.status_code == 422:
                    # Query error (e.g., too long, invalid format)
                    logger.warning(f"Query error for {ticker}: {response.text[:100]}")
                    return []
                
                elif response.status_code >= 500:
                    logger.warning(f"Server error (HTTP {response.status_code}) for {ticker}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return []
                
                response.raise_for_status()
                
                # Record successful API call
                budget_tracker.record_call(credits_used=1, ticker=ticker)
                
                data = response.json()
                raw_articles = data.get('results', [])
                
                # Filter out sports/entertainment based on title keywords
                exclude_keywords = ['futebol', 'jogo', 'partida', 'copa', 'campeonato', 'selção']
                formatted_articles = []
                
                for article in raw_articles:
                    title = article.get('title', '').lower()
                    
                    # Skip sports articles
                    if any(kw in title for kw in exclude_keywords):
                        continue
                    
                    formatted_articles.append({
                        'title': article.get('title', ''),
                        'link': article.get('link', ''),
                        'published': article.get('pubDate', ''),
                        'summary': article.get('description', ''),
                        'source': 'newsdata.io',
                        'source_publication': article.get('source_name', 'Unknown')
                    })
                
                return formatted_articles
            
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries}) for {ticker}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            
            except Exception as e:
                logger.error(f"Error fetching newsdata.io for {ticker}: {e}")
                return []
        
        return []
    
    def _fetch_investing_com(self, ticker: str, date: str) -> List[Dict]:
        """
        Scrape Investing.com Brasil for historical news.
        
        Args:
            ticker: Stock ticker (e.g., PETR4.SA)
            date: Date string (YYYY-MM-DD)
        
        Returns:
            List of news articles
        """
        self._rate_limit()
        
        query = self._ticker_to_query(ticker)
        
        # Investing.com Brasil news search URL
        search_url = f"https://br.investing.com/search/?q={query}&tab=news"
        
        logger.debug(f"Scraping Investing.com for '{query}' on {date}")
        
        try:
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            articles = []
            
            # Find news articles (adapt selector based on actual HTML)
            news_items = soup.find_all('article', class_='js-article-item')
            
            for item in news_items[:10]:  # Limit to 10 articles
                try:
                    title_elem = item.find('a', class_='title')
                    title = title_elem.get_text(strip=True) if title_elem else ''
                    link = title_elem.get('href', '') if title_elem else ''
                    
                    date_elem = item.find('time')
                    pub_date = date_elem.get('datetime', '') if date_elem else ''
                    
                    summary_elem = item.find('p')
                    summary = summary_elem.get_text(strip=True) if summary_elem else ''
                    
                    if title:
                        articles.append({
                            'title': title,
                            'link': link,
                            'published': pub_date,
                            'summary': summary,
                            'source': 'investing_com'
                        })
                
                except Exception as e:
                    logger.debug(f"Failed to parse article: {e}")
                    continue
            
            logger.info(f"Scraped {len(articles)} articles from Investing.com for '{query}'")
            return articles
        
        except Exception as e:
            logger.error(f"Failed to scrape Investing.com: {e}")
            return []
    
    def _analyze_articles(self, articles: List[Dict]) -> float:
        """
        Analyze sentiment of articles using FinBERT.
        
        Args:
            articles: List of article dicts
        
        Returns:
            Average sentiment score (-1.0 to +1.0)
        """
        if not articles:
            return 0.0
        
        sentiments = []
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('summary', '')}"
            
            if text.strip():
                sentiment = self.sentiment_analyzer.analyze(text)
                sentiments.append(sentiment)
        
        if not sentiments:
            return 0.0
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        logger.debug(f"Analyzed {len(sentiments)} articles: avg_sentiment={avg_sentiment:.2f}")
        
        return avg_sentiment
    
    @lru_cache(maxsize=256)
    def get_sentiment(self, ticker: str, date: str, use_cache: bool = True) -> float:
        """
        Get sentiment score for a ticker on a given date.
        
        Smart caching strategy:
        1. Check cache first (if fresh, return immediately - no API call)
        2. If cache miss, fetch from newsdata.io API
        3. Fallback to Investing.com scraping if needed
        4. Store in cache for future use (6-hour TTL)
        
        Args:
            ticker: Stock ticker (e.g., PETR4.SA)
            date: Date string (YYYY-MM-DD)
            use_cache: Use cache if fresh (default True)
        
        Returns:
            Sentiment score (-1.0 to +1.0)
        """
        try:
            # CACHE: Check if we have fresh sentiment in cache
            if use_cache:
                cached_sentiment = self.cache.get(ticker)
                if cached_sentiment is not None:
                    logger.info(f"✅ Cache HIT: {ticker} (sentiment: {cached_sentiment:+.2f}, age: {self.cache.get_cache_age(ticker)})")
                    return cached_sentiment
            
            # MISS: Fetch fresh news
            logger.debug(f"Cache MISS or disabled - fetching fresh news for {ticker}")
            
            # Try newsdata.io API first (paid API, Brazilian financial news)
            articles = self._fetch_newsdata_io(ticker, date)
            
            # Fallback 1: If newsdata.io fails, try Google News RSS (free, no API key needed)
            if not articles:
                logger.debug(f"newsdata.io returned no results, trying Google News for {ticker}")
                query = self._ticker_to_query(ticker)
                articles = self._fetch_google_news(query, days=7)
            
            # DISABLED: Investing.com scraping - returns 403 Forbidden, causes timeouts
            # Fallback 2: If newsdata.io fails, try Investing.com scraping
            # if not articles:
            #     logger.debug(f"newsdata.io returned no results, falling back to Investing.com for {ticker}")
            #     articles = self._fetch_investing_com(ticker, date)
            
            if not articles:
                logger.info(f"No news found for {ticker} on {date} (Google News + newsdata.io)")
                return 0.0
            
            sentiment = self._analyze_articles(articles)
            
            # Determine source: check if articles came from newsdata.io or Investing.com fallback
            source = articles[0].get('source', 'unknown') if articles else 'unknown'
            
            # CACHE: Store sentiment + article summaries for future use
            self.cache.set(ticker, sentiment, source=source, articles=articles)
            
            logger.info(f"{ticker} on {date}: {len(articles)} articles from {source}, sentiment={sentiment:+.2f}")
            
            return sentiment
        
        except Exception as e:
            logger.error(f"Failed to get sentiment for {ticker}: {e}")
            return 0.0
    
    def get_sentiment_batch(self, tickers: List[str], date: str) -> Dict[str, float]:
        """
        Get sentiment scores for multiple tickers.
        
        Args:
            tickers: List of stock tickers
            date: Date string (YYYY-MM-DD)
        
        Returns:
            Dict mapping ticker to sentiment score
        """
        results = {}
        
        for ticker in tickers:
            results[ticker] = self.get_sentiment(ticker, date)
            time.sleep(0.5)  # Small delay between tickers
        
        return results
    
    def refresh_sentiment_batch(self, tickers: List[str], date: str = None) -> Dict[str, float]:
        """
        Refresh sentiment for specific tickers (e.g., top movers).
        
        Used for smart news fetching: only fetch fresh news for stocks with significant moves.
        This minimizes API calls while keeping high-conviction signals updated.
        
        Args:
            tickers: List of tickers to fetch fresh news for
            date: Date string (defaults to today)
        
        Returns:
            Dict of ticker -> sentiment scores
        
        Example:
            # Fetch news only for top 10 movers
            results = news_client.refresh_sentiment_batch(['VALE3.SA', 'PETR4.SA', 'ITUB4.SA'])
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        results = {}
        
        logger.info(f"🚀 Refreshing sentiment for {len(tickers)} top movers (API calls will be used)")
        
        for ticker in tickers:
            sentiment = self.get_sentiment(ticker, date, use_cache=False)
            results[ticker] = sentiment
        
        logger.info(f"✅ Refreshed {len(tickers)} tickers. API credits used: ~{len(tickers)}")
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.stats()
    
    def clear_cache(self):
        """Clear the sentiment cache."""
        self.get_sentiment.cache_clear()
        logger.info("Sentiment cache cleared")


# Singleton instance
_client: Optional[FreeNewsClient] = None


def get_client() -> FreeNewsClient:
    """Get or create the global FreeNewsClient instance."""
    global _client
    if _client is None:
        _client = FreeNewsClient()
    return _client
