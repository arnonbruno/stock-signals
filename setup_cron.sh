#!/bin/bash

# Add daily market scan to crontab
# Runs at 10:30 AM BRT (market open) and 5:30 PM BRT (market close)

# Create cron entry
CRON_JOB="30 10,17 * * 1-5 cd /home/ulluboz/repos/stock-signals && /home/ulluboz/miniconda3/bin/python daily_market_scan.py >> daily_scan.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "daily_market_scan.py"; then
    echo "✅ Cron job already exists"
else
    # Add cron job
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "✅ Added daily market scan to cron"
    echo "   Runs at: 10:30 AM and 5:30 PM BRT (Mon-Fri)"
fi

# Show current crontab
echo ""
echo "Current crontab:"
crontab -l 2>/dev/null || echo "No crontab configured"
