"""
API budget tracker for newsdata.io rate limiting.

Prevents exceeding daily credit limits by:
1. Tracking cumulative API calls per day
2. Stopping requests before hitting limit
3. Logging usage for monitoring
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict


logger = logging.getLogger(__name__)


class APIBudgetTracker:
    """
    Track newsdata.io API credit usage and enforce limits.
    
    Daily budget: 200 credits
    Conservative limit: 150 credits (50 credit safety margin)
    """
    
    DAILY_LIMIT = 200
    CONSERVATIVE_LIMIT = 150  # Leave 50 credit headroom
    CREDITS_PER_CALL = 1  # newsdata.io charges 1 credit per API call
    
    def __init__(self, budget_file: str = "api_budget.json"):
        self.budget_file = Path(budget_file)
        self.today = datetime.now().strftime("%Y-%m-%d")
        self.budget_data = self._load_budget()
        self._cleanup_old_days()
    
    def _load_budget(self) -> Dict:
        """Load budget data from disk."""
        if self.budget_file.exists():
            try:
                with open(self.budget_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load budget file: {e}")
                return {}
        return {}
    
    def _save_budget(self):
        """Save budget data to disk."""
        try:
            with open(self.budget_file, 'w') as f:
                json.dump(self.budget_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save budget file: {e}")
    
    def _cleanup_old_days(self):
        """Remove budget data older than 7 days."""
        cutoff_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        days_to_remove = [day for day in self.budget_data if day < cutoff_date]
        for day in days_to_remove:
            del self.budget_data[day]
        if days_to_remove:
            self._save_budget()
    
    def can_make_request(self, credits_needed: int = 1) -> bool:
        """
        Check if we can make a request without exceeding budget.
        
        Args:
            credits_needed: Credits required for this request (default 1)
        
        Returns:
            True if request can be made, False if budget exceeded
        """
        if self.today not in self.budget_data:
            self.budget_data[self.today] = {'used': 0, 'calls': 0}
        
        current_usage = self.budget_data[self.today]['used']
        remaining = self.CONSERVATIVE_LIMIT - current_usage
        
        if credits_needed > remaining:
            logger.warning(f"⚠️ API budget exhausted for today: {current_usage}/{self.CONSERVATIVE_LIMIT} used")
            return False
        
        return True
    
    def record_call(self, credits_used: int = 1, ticker: str = None):
        """
        Record an API call and credit usage.
        
        Args:
            credits_used: Number of credits used (default 1)
            ticker: Ticker that was looked up (for logging)
        """
        if self.today not in self.budget_data:
            self.budget_data[self.today] = {'used': 0, 'calls': 0}
        
        self.budget_data[self.today]['used'] += credits_used
        self.budget_data[self.today]['calls'] += 1
        
        current_usage = self.budget_data[self.today]['used']
        percent_used = (current_usage / self.CONSERVATIVE_LIMIT) * 100
        
        logger.info(f"API call recorded for {ticker or 'unknown'}: {current_usage}/{self.CONSERVATIVE_LIMIT} credits used ({percent_used:.1f}%)")
        
        self._save_budget()
    
    def get_usage(self) -> Dict:
        """Get today's API usage stats."""
        if self.today not in self.budget_data:
            return {
                'date': self.today,
                'used': 0,
                'calls': 0,
                'remaining': self.CONSERVATIVE_LIMIT,
                'percent_used': 0.0
            }
        
        data = self.budget_data[self.today]
        used = data['used']
        remaining = self.CONSERVATIVE_LIMIT - used
        percent_used = (used / self.CONSERVATIVE_LIMIT) * 100
        
        return {
            'date': self.today,
            'used': used,
            'calls': data['calls'],
            'remaining': remaining,
            'percent_used': percent_used
        }
    
    def reset_daily_budget(self):
        """Reset budget for new day (or force reset current day)."""
        self.today = datetime.now().strftime("%Y-%m-%d")
        # Reset to 0 - this handles both new day and force reset
        self.budget_data[self.today] = {'used': 0, 'calls': 0}
        self._save_budget()


# Global tracker instance
_budget_tracker: APIBudgetTracker = None


def get_budget_tracker() -> APIBudgetTracker:
    """Get or create global budget tracker."""
    global _budget_tracker
    if _budget_tracker is None:
        _budget_tracker = APIBudgetTracker()
    return _budget_tracker
