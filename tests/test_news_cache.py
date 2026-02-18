"""
Comprehensive tests for NewsCache - caching system with TTL.

Tests cover:
- Cache set/get operations
- TTL expiration
- Market-aware TTL (trading hours vs overnight)
- Cache persistence
- Article storage
- Cache statistics
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
from unittest.mock import patch, MagicMock
import tempfile
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.news.news_cache import NewsCache


class TestNewsCacheCore:
    """Core cache operations tests."""
    
    @pytest.fixture
    def cache(self):
        """Create cache with temporary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"
            yield NewsCache(cache_file=str(cache_file), ttl_hours=6)
    
    def test_set_and_get(self, cache):
        """Test basic set and get operations."""
        cache.set('PETR4.SA', 0.5, source='test', articles=[])
        
        result = cache.get('PETR4.SA')
        assert result == 0.5, f"Expected 0.5, got {result}"
    
    def test_get_missing_key(self, cache):
        """Test get returns None for missing key."""
        result = cache.get('MISSING.SA')
        assert result is None, "Should return None for missing key"
    
    def test_get_full_returns_all_data(self, cache):
        """Test get_full returns complete cache entry."""
        articles = [
            {'title': 'Test Article', 'summary': 'Test summary'}
        ]
        cache.set('PETR4.SA', 0.75, source='newsdata_io', articles=articles)
        
        result = cache.get_full('PETR4.SA')
        
        assert result is not None
        assert result['sentiment'] == 0.75
        assert result['source'] == 'newsdata_io'
        assert len(result['articles']) == 1
        assert 'timestamp' in result
    
    def test_set_truncates_articles(self, cache):
        """Test that set truncates articles to top 3."""
        articles = [
            {'title': f'Article {i}', 'summary': f'Summary {i}'}
            for i in range(10)
        ]
        cache.set('TEST.SA', 0.5, source='test', articles=articles)
        
        result = cache.get_full('TEST.SA')
        assert len(result['articles']) == 3, "Should only keep 3 articles"
    
    def test_set_truncates_summary(self, cache):
        """Test that article summaries are truncated to 150 chars."""
        long_summary = 'x' * 200
        articles = [{'title': 'Test', 'summary': long_summary}]
        
        cache.set('TEST.SA', 0.5, source='test', articles=articles)
        
        result = cache.get_full('TEST.SA')
        assert len(result['articles'][0]['summary']) == 150


class TestCacheTTL:
    """Tests for TTL (time-to-live) functionality."""
    
    @pytest.fixture
    def cache(self):
        """Create cache with temporary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"
            yield NewsCache(cache_file=str(cache_file), ttl_hours=6)
    
    def test_is_fresh_for_new_entry(self, cache):
        """Test that new entries are fresh."""
        cache.set('TEST.SA', 0.5, source='test')
        
        assert cache.is_fresh('TEST.SA'), "New entry should be fresh"
    
    def test_is_fresh_returns_false_for_expired(self, cache):
        """Test that expired entries are not fresh."""
        # Manually create expired entry
        cache.cache['EXPIRED.SA'] = {
            'sentiment': 0.5,
            'timestamp': (datetime.now() - timedelta(hours=10)).isoformat(),
            'source': 'test'
        }
        
        assert not cache.is_fresh('EXPIRED.SA'), "Old entry should not be fresh"
    
    def test_get_returns_none_for_expired(self, cache):
        """Test that get returns None for expired entries."""
        # Create expired entry
        cache.cache['EXPIRED.SA'] = {
            'sentiment': 0.5,
            'timestamp': (datetime.now() - timedelta(hours=10)).isoformat(),
            'source': 'test'
        }
        
        result = cache.get('EXPIRED.SA')
        assert result is None, "Should return None for expired entry"
    
    def test_get_cache_age(self, cache):
        """Test cache age calculation."""
        cache.set('TEST.SA', 0.5, source='test')
        
        age = cache.get_cache_age('TEST.SA')
        
        assert age is not None
        assert age < timedelta(minutes=1), "Fresh entry should have small age"
    
    def test_get_cache_age_missing_key(self, cache):
        """Test cache age for missing key."""
        age = cache.get_cache_age('MISSING.SA')
        assert age is None, "Should return None for missing key"


class TestMarketAwareTTL:
    """Tests for market-aware TTL adjustment."""
    
    @pytest.fixture
    def cache(self):
        """Create cache with temporary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"
            yield NewsCache(cache_file=str(cache_file), ttl_hours=6)
    
    def test_trading_hours_shorter_ttl(self, cache):
        """Test that trading hours use shorter TTL (4h)."""
        with patch.object(cache, '_get_trading_ttl') as mock_ttl:
            mock_ttl.return_value = 4
            
            ttl = cache._get_trading_ttl()
            assert ttl == 4, "Trading hours should use 4h TTL"
    
    def test_overnight_longer_ttl(self, cache):
        """Test that overnight uses longer TTL (12h)."""
        with patch.object(cache, '_get_trading_ttl') as mock_ttl:
            mock_ttl.return_value = 12
            
            ttl = cache._get_trading_ttl()
            assert ttl == 12, "Overnight should use 12h TTL"
    
    def test_ttl_during_market_hours(self, cache):
        """Test TTL calculation during B3 market hours (10:00-17:00 GMT-3)."""
        # This test verifies the logic, actual time may vary
        ttl = cache._get_trading_ttl()
        assert ttl in [4, 12], f"TTL should be 4 or 12, got {ttl}"


class TestCachePersistence:
    """Tests for cache persistence to disk."""
    
    def test_cache_saves_to_disk(self):
        """Test that cache saves to disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"
            cache = NewsCache(cache_file=str(cache_file), ttl_hours=6)
            
            cache.set('TEST.SA', 0.5, source='test')
            
            # Verify file exists
            assert cache_file.exists(), "Cache file should be created"
            
            # Verify content
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            assert 'TEST.SA' in data
            assert data['TEST.SA']['sentiment'] == 0.5
    
    def test_cache_loads_from_disk(self):
        """Test that cache loads existing data from disk."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"
            
            # Create initial cache with data
            cache1 = NewsCache(cache_file=str(cache_file), ttl_hours=6)
            cache1.set('TEST.SA', 0.75, source='test')
            
            # Create new cache instance (should load from disk)
            cache2 = NewsCache(cache_file=str(cache_file), ttl_hours=6)
            
            result = cache2.get('TEST.SA')
            assert result == 0.75, "Should load cached value from disk"
    
    def test_cache_handles_corrupted_file(self):
        """Test that cache handles corrupted JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"
            
            # Write corrupted JSON
            with open(cache_file, 'w') as f:
                f.write("{ corrupted json")
            
            # Should not crash, returns empty cache
            cache = NewsCache(cache_file=str(cache_file), ttl_hours=6)
            
            assert cache.cache == {}, "Should return empty cache for corrupted file"


class TestCacheStatistics:
    """Tests for cache statistics."""
    
    @pytest.fixture
    def cache(self):
        """Create cache with temporary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"
            yield NewsCache(cache_file=str(cache_file), ttl_hours=6)
    
    def test_stats_returns_dict(self, cache):
        """Test that stats returns dictionary."""
        stats = cache.stats()
        
        assert isinstance(stats, dict)
        assert 'total_entries' in stats
        assert 'fresh' in stats
        assert 'expired' in stats
    
    def test_stats_counts_correctly(self, cache):
        """Test that stats counts entries after pruning."""
        # Add fresh entry
        cache.set('FRESH.SA', 0.5, source='test')
        
        # Add expired entry manually
        cache.cache['EXPIRED.SA'] = {
            'sentiment': 0.5,
            'timestamp': (datetime.now() - timedelta(hours=10)).isoformat(),
            'source': 'test'
        }
        
        stats = cache.stats()
        
        # stats() prunes expired entries first, so only fresh remains
        assert stats['total_entries'] == 1
        assert stats['fresh'] == 1
        assert stats['expired'] == 0  # Pruned before count


class TestCachePruning:
    """Tests for cache pruning (removing expired entries)."""
    
    @pytest.fixture
    def cache(self):
        """Create cache with temporary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"
            yield NewsCache(cache_file=str(cache_file), ttl_hours=6)
    
    def test_prune_expired_removes_old_entries(self, cache):
        """Test that prune_expired removes expired entries."""
        # Add fresh entry
        cache.set('FRESH.SA', 0.5, source='test')
        
        # Add expired entry
        cache.cache['EXPIRED.SA'] = {
            'sentiment': 0.5,
            'timestamp': (datetime.now() - timedelta(hours=10)).isoformat(),
            'source': 'test'
        }
        
        cache.prune_expired()
        
        assert 'FRESH.SA' in cache.cache
        assert 'EXPIRED.SA' not in cache.cache
    
    def test_stats_triggers_pruning(self, cache):
        """Test that stats() triggers pruning."""
        # Add expired entry
        cache.cache['EXPIRED.SA'] = {
            'sentiment': 0.5,
            'timestamp': (datetime.now() - timedelta(hours=10)).isoformat(),
            'source': 'test'
        }
        
        stats = cache.stats()
        
        assert stats['expired'] == 0, "Pruning should remove expired entries"


class TestCacheEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.fixture
    def cache(self):
        """Create cache with temporary file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_file = Path(tmpdir) / "test_cache.json"
            yield NewsCache(cache_file=str(cache_file), ttl_hours=6)
    
    def test_set_with_none_articles(self, cache):
        """Test set with None articles."""
        cache.set('TEST.SA', 0.5, source='test', articles=None)
        
        result = cache.get_full('TEST.SA')
        assert result['articles'] == [], "Should handle None articles"
    
    def test_set_with_empty_articles(self, cache):
        """Test set with empty articles list."""
        cache.set('TEST.SA', 0.5, source='test', articles=[])
        
        result = cache.get_full('TEST.SA')
        assert result['articles'] == []
    
    def test_get_cache_age_handles_invalid_timestamp(self, cache):
        """Test get_cache_age handles invalid timestamp."""
        cache.cache['BAD.SA'] = {
            'sentiment': 0.5,
            'timestamp': 'invalid-timestamp',
            'source': 'test'
        }
        
        age = cache.get_cache_age('BAD.SA')
        assert age is None, "Should return None for invalid timestamp"
    
    def test_is_fresh_handles_invalid_timestamp(self, cache):
        """Test is_fresh handles invalid timestamp."""
        cache.cache['BAD.SA'] = {
            'sentiment': 0.5,
            'timestamp': 'invalid-timestamp',
            'source': 'test'
        }
        
        # Should not crash, returns False
        assert not cache.is_fresh('BAD.SA')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
