#!/usr/bin/env python3
"""
News cache management - reduces API calls by caching sentiment scores.

Strategy:
- Cache news sentiment for up to 6 hours
- Only fetch fresh news for top movers (high-confidence signals)
- Reuse cached sentiment for consolidating stocks
- Budget: ~150 credits/day (50 headroom from 200 limit)
"""

import json
import logging
import fcntl
import time as time_module
from datetime import datetime, timedelta, time
from pathlib import Path
from typing import Dict, Optional, List, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class NewsCache:
    """Manage news sentiment cache with TTL and file locking for multi-process safety."""
    
    def __init__(self, cache_file: str = "news_sentiment_cache.json", ttl_hours: int = 6):
        self.cache_file = Path(cache_file)
        self.lock_file = Path(str(cache_file).replace('.json', '.lock'))
        self.ttl = timedelta(hours=ttl_hours)
        self.cache = self._load_cache()
    
    @contextmanager
    def _file_lock(self, timeout: float = 5.0):
        """File lock for multi-process safety.
        
        Args:
            timeout: Max seconds to wait for lock
        
        Raises:
            TimeoutError: If lock cannot be acquired within timeout
        """
        lock_fd = None
        try:
            lock_fd = open(self.lock_file, 'w')
            start_time = time_module.time()
            
            while True:
                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break  # Lock acquired
                except (IOError, OSError):
                    if time_module.time() - start_time > timeout:
                        raise TimeoutError(f"Could not acquire lock within {timeout}s")
                    time_module.sleep(0.05)  # Wait 50ms before retry
            
            yield lock_fd
        finally:
            if lock_fd:
                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                    lock_fd.close()
                except:
                    pass
    
    def _load_cache(self) -> Dict:
        """Load cache from disk (no locking - caller should handle locking)."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk (no locking - caller should handle locking)."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def _get_trading_ttl(self) -> int:
        """Get TTL based on market hours.
        
        B3 Market Hours: 10:00 - 17:00 GMT-3
        Returns shorter TTL during trading hours (4h), longer overnight (12h).
        """
        current_time = datetime.now().time()
        
        if time(10, 0) <= current_time < time(17, 0):
            return 4  # Trading hours - faster decay
        else:
            return 12  # Overnight - slower decay
    
    def is_fresh(self, ticker: str) -> bool:
        """Check if cached sentiment is still fresh (within TTL)."""
        if ticker not in self.cache:
            return False
        
        try:
            cached_time = datetime.fromisoformat(self.cache[ticker]['timestamp'])
            current_ttl_hours = self._get_trading_ttl()
            current_ttl = timedelta(hours=current_ttl_hours)
            age = datetime.now() - cached_time
            fresh = age < current_ttl
            
            if not fresh:
                logger.debug(f"{ticker} cache expired (age: {age})")
            
            return fresh
        except Exception as e:
            logger.warning(f"Error checking cache for {ticker}: {e}")
            return False
    
    def is_pending(self, ticker: str, max_age_seconds: int = 30) -> bool:
        """Check if ticker is currently being fetched by another process.
        
        A ticker is "pending" if it has a 'pending' flag set within the last max_age_seconds.
        This prevents multiple workers from fetching the same ticker simultaneously.
        """
        try:
            with self._file_lock(timeout=1.0):
                self.cache = self._load_cache()
                
                if ticker not in self.cache:
                    return False
                
                entry = self.cache[ticker]
                if not entry.get('pending'):
                    return False
                
                # Check if pending flag is recent (within max_age_seconds)
                cached_time = datetime.fromisoformat(entry['timestamp'])
                age = (datetime.now() - cached_time).total_seconds()
                
                if age < max_age_seconds:
                    logger.debug(f"{ticker} is pending (age: {age:.1f}s)")
                    return True
                
                return False
        except TimeoutError:
            # If we can't check, assume not pending
            return False
    
    def mark_pending(self, ticker: str):
        """Mark ticker as pending (being fetched by this process)."""
        try:
            with self._file_lock(timeout=1.0):
                self.cache = self._load_cache()
                
                if ticker in self.cache:
                    self.cache[ticker]['pending'] = True
                else:
                    self.cache[ticker] = {
                        'pending': True,
                        'timestamp': datetime.now().isoformat(),
                        'sentiment': 0.0,
                        'source': 'pending',
                        'articles': []
                    }
                
                # Save immediately
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache, f, indent=2)
        except TimeoutError:
            logger.warning(f"Could not mark {ticker} as pending")
    
    def clear_pending(self, ticker: str):
        """Clear pending flag for ticker."""
        try:
            with self._file_lock(timeout=1.0):
                self.cache = self._load_cache()
                
                if ticker in self.cache and 'pending' in self.cache[ticker]:
                    del self.cache[ticker]['pending']
                    with open(self.cache_file, 'w') as f:
                        json.dump(self.cache, f, indent=2)
        except TimeoutError:
            logger.warning(f"Could not clear pending for {ticker}")
    
    def get(self, ticker: str) -> Optional[float]:
        """Get cached sentiment if fresh, otherwise None.
        
        Reloads cache from disk to see updates from other processes.
        """
        try:
            with self._file_lock(timeout=1.0):
                self.cache = self._load_cache()
                
                if ticker in self.cache and self.is_fresh(ticker):
                    sentiment = self.cache[ticker].get('sentiment', 0.0)
                    logger.debug(f"Cache HIT: {ticker} (sentiment: {sentiment:+.2f})")
                    return sentiment
        except TimeoutError:
            logger.warning(f"Could not acquire lock to check cache for {ticker}")
        
        return None
    
    def get_full(self, ticker: str) -> Optional[Dict]:
        """Get cached sentiment + article details if fresh, otherwise None.
        
        Reloads cache from disk to see updates from other processes.
        """
        try:
            with self._file_lock(timeout=1.0):
                self.cache = self._load_cache()
                
                if ticker in self.cache and self.is_fresh(ticker):
                    logger.debug(f"Cache HIT: {ticker}")
                    return self.cache[ticker]
        except TimeoutError:
            logger.warning(f"Could not acquire lock to check cache for {ticker}")
        
        return None
    
    def set(self, ticker: str, sentiment: float, source: str = "newsdata_io", articles: List = None):
        """Cache sentiment score + article snippets with timestamp."""
        # Store top 3 article snippets for display in alerts
        article_summaries = []
        if articles:
            for article in articles[:3]:  # Keep only top 3
                article_summaries.append({
                    'title': article.get('title', ''),
                    'summary': article.get('summary', '')[:150],  # Truncate to 150 chars
                    'source_publication': article.get('source_publication', 'Unknown'),
                    'link': article.get('link', '')
                })
        
        try:
            with self._file_lock(timeout=2.0):
                self.cache = self._load_cache()
                
                self.cache[ticker] = {
                    'sentiment': sentiment,
                    'timestamp': datetime.now().isoformat(),
                    'source': source,
                    'articles': article_summaries
                }
                
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache, f, indent=2)
                
                logger.info(f"Cached {ticker}: {sentiment:+.2f} from {source} ({len(article_summaries)} articles)")
        except TimeoutError:
            logger.error(f"Could not acquire lock to save cache for {ticker}")
    
    def get_cache_age(self, ticker: str) -> Optional[timedelta]:
        """Get age of cached item."""
        if ticker not in self.cache:
            return None
        
        try:
            cached_time = datetime.fromisoformat(self.cache[ticker]['timestamp'])
            return datetime.now() - cached_time
        except:
            return None
    
    def prune_expired(self):
        """Remove expired entries from cache."""
        expired = [k for k in self.cache if not self.is_fresh(k)]
        for ticker in expired:
            del self.cache[ticker]
        
        if expired:
            self._save_cache()
            logger.info(f"Pruned {len(expired)} expired cache entries")
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        self.prune_expired()
        
        fresh = sum(1 for k in self.cache if self.is_fresh(k))
        expired = len(self.cache) - fresh
        
        return {
            'total_entries': len(self.cache),
            'fresh': fresh,
            'expired': expired,
            'cache_file': str(self.cache_file)
        }
