"""
Fundamentus scraper for Brazilian stock fundamentals.

Scrapes fundamental data from fundamentus.com.br for value investing analysis.
Designed to run once daily and cache results.
"""

import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import time
from pathlib import Path
from datetime import datetime
import re


@dataclass
class FundamentalData:
    """Fundamental data for a stock."""
    ticker: str
    timestamp: str
    
    # Valuation ratios
    pe_ratio: Optional[float] = None  # P/L
    pb_ratio: Optional[float] = None  # P/VP
    ps_ratio: Optional[float] = None  # PSR (Price/Sales)
    p_ebit: Optional[float] = None  # P/EBIT
    ev_ebitda: Optional[float] = None  # EV/EBITDA
    ev_ebit: Optional[float] = None  # EV/EBIT
    
    # Profitability
    roe: Optional[float] = None  # ROE (%)
    roic: Optional[float] = None  # ROIC (%)
    ebit_margin: Optional[float] = None  # Marg. EBIT (%)
    net_margin: Optional[float] = None  # Marg. Líquida (%)
    gross_margin: Optional[float] = None  # Marg. Bruta (%)
    
    # Financial health
    debt_equity: Optional[float] = None  # Div Br/Patrim
    current_ratio: Optional[float] = None  # Liquidez Corr
    
    # Dividends & Growth
    div_yield: Optional[float] = None  # Div. Yield (%)
    revenue_growth_5y: Optional[float] = None  # Cres. Rec (5a) (%)
    
    # Per-share values
    eps: Optional[float] = None  # LPA
    bvps: Optional[float] = None  # VPA
    
    # Balance sheet (in millions)
    total_assets: Optional[float] = None  # Ativo
    total_equity: Optional[float] = None  # Patrim. Líq
    net_debt: Optional[float] = None  # Dívida Líquida
    ebit: Optional[float] = None  # EBIT (12m)
    net_income: Optional[float] = None  # Lucro Líquido (12m)
    revenue: Optional[float] = None  # Receita Líquida (12m)
    
    # Market data
    market_cap: Optional[float] = None  # Valor de mercado
    price: Optional[float] = None  # Cotação
    
    # Metadata
    sector: Optional[str] = None
    subsector: Optional[str] = None
    company_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'ticker': self.ticker,
            'timestamp': self.timestamp,
            'pe_ratio': self.pe_ratio,
            'pb_ratio': self.pb_ratio,
            'ps_ratio': self.ps_ratio,
            'p_ebit': self.p_ebit,
            'ev_ebitda': self.ev_ebitda,
            'ev_ebit': self.ev_ebit,
            'roe': self.roe,
            'roic': self.roic,
            'ebit_margin': self.ebit_margin,
            'net_margin': self.net_margin,
            'gross_margin': self.gross_margin,
            'debt_equity': self.debt_equity,
            'current_ratio': self.current_ratio,
            'div_yield': self.div_yield,
            'revenue_growth_5y': self.revenue_growth_5y,
            'eps': self.eps,
            'bvps': self.bvps,
            'total_assets': self.total_assets,
            'total_equity': self.total_equity,
            'net_debt': self.net_debt,
            'ebit': self.ebit,
            'net_income': self.net_income,
            'revenue': self.revenue,
            'market_cap': self.market_cap,
            'price': self.price,
            'sector': self.sector,
            'subsector': self.subsector,
            'company_name': self.company_name,
        }


class FundamentusScraper:
    """Scrapes fundamental data from fundamentus.com.br."""
    
    BASE_URL = "https://www.fundamentus.com.br/detalhes.php"
    HEADERS = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        'Accept-Language': 'pt-BR,pt;q=0.9,en;q=0.8',
    }
    
    # Mapping of Portuguese labels to English field names
    # Note: Fundamentus prefixes labels with '?' in the display
    LABEL_MAP = {
        'P/L': 'pe_ratio',
        'P/VP': 'pb_ratio',
        'PSR': 'ps_ratio',
        'P/EBIT': 'p_ebit',
        'EV / EBITDA': 'ev_ebitda',
        'EV / EBIT': 'ev_ebit',
        'ROE': 'roe',
        'ROIC': 'roic',
        'Marg. EBIT': 'ebit_margin',
        'Marg. Líquida': 'net_margin',
        'Marg. Bruta': 'gross_margin',
        'Div Br/ Patrim': 'debt_equity',
        'Div Brut/ Patrim': 'debt_equity',
        'Liquidez Corr': 'current_ratio',
        'Div. Yield': 'div_yield',
        'Cres. Rec (5a)': 'revenue_growth_5y',
        'LPA': 'eps',
        'VPA': 'bvps',
    }
    
    # Labels that might have '?' prefix
    PREFIX_VARIANTS = ['?', '']
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize scraper with optional cache directory."""
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent / 'data' / 'fundamentals'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _parse_number(self, value: str) -> Optional[float]:
        """Parse Brazilian number format (1.234.567,89) to float."""
        if not value or value.strip() in ('-', '', 'N/A'):
            return None
        
        try:
            # Remove % sign if present
            value = value.strip().replace('%', '').strip()
            
            # Handle Brazilian format:
            # - 1.234.567,89 (dots = thousands, comma = decimal)
            # - 1.234,56 (comma = decimal)
            # - 489.385.000.000 (dots = thousands, no decimal)
            
            if ',' in value and '.' in value:
                # Both: 1.234.567,89 -> remove dots, convert comma to dot
                value = value.replace('.', '').replace(',', '.')
            elif ',' in value:
                # Only comma: 1234,56 -> convert to dot
                value = value.replace(',', '.')
            elif '.' in value:
                # Only dots: could be thousands separator
                # Check if it looks like thousands (multiple dots or > 4 digits before dot)
                parts = value.split('.')
                if len(parts) > 2 or (len(parts) == 2 and len(parts[0]) > 3):
                    # It's a thousands separator, not decimal
                    value = value.replace('.', '')
                # else: it might be a US-style decimal like 1.5, leave as-is
            
            return float(value)
        except (ValueError, AttributeError):
            return None
    
    def _parse_large_number(self, value: str) -> Optional[float]:
        """Parse large numbers from fundamentus (in thousands) to millions."""
        num = self._parse_number(value)
        if num is not None:
            # Fundamentus shows values in thousands (1.212.040.000.000 = 1.212.040 thousand)
            # Convert to millions for readability
            return num / 1_000_000  # Convert thousands to millions
        return None
    
    def scrape_ticker(self, ticker: str) -> Optional[FundamentalData]:
        """Scrape fundamental data for a single ticker.
        
        Args:
            ticker: Stock ticker without .SA suffix (e.g., 'PETR4')
        
        Returns:
            FundamentalData object or None if failed
        """
        # Remove .SA suffix if present
        clean_ticker = ticker.replace('.SA', '')
        
        # SOTA: Basic structural validation variables
        indicators_parsed = 0
        
        try:
            url = f"{self.BASE_URL}?papel={clean_ticker}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Fix encoding issues
            response.encoding = 'iso-8859-1'
            soup = BeautifulSoup(response.text, 'html.parser')
            
            data = FundamentalData(
                ticker=clean_ticker,
                timestamp=datetime.now().isoformat()
            )
            
            # Parse company name and sector from header
            self._parse_header(soup, data)
            
            # Parse indicator tables
            indicators_parsed = self._parse_indicators(soup, data)
            
            # Parse balance sheet
            self._parse_balance_sheet(soup, data)
            
            # Structural validation: if we couldn't parse at least 5 key indicators, 
            # it means the HTML structure has likely changed or the page is empty/blocked.
            if indicators_parsed < 5:
                print(f"⚠️ Warning: Only {indicators_parsed} indicators parsed for {clean_ticker}. HTML structure might have changed or data is missing.")
                # We could implement a fallback to BrAPI here in the future
                # For now, return None to prevent corrupting the cache with zeroes
                if not data.pe_ratio and not data.pb_ratio and not data.roe:
                    return None
            
            return data
            
        except requests.RequestException as e:
            print(f"Error fetching {clean_ticker}: {e}")
            return None
        except Exception as e:
            print(f"Error parsing {clean_ticker}: {e}")
            return None
    
    def _parse_header(self, soup: BeautifulSoup, data: FundamentalData):
        """Parse header information (company name, sector, price)."""
        # Find tables with indicator data
        tables = soup.find_all('table', class_='w728')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                
                # Handle both 4-cell structure (2 label-value pairs) and 2-cell structure
                # Process cells in pairs: (label, value)
                for i in range(0, len(cells) - 1, 2):
                    raw_label = cells[i].get_text(strip=True)
                    label = self._normalize_label(raw_label)
                    value = cells[i + 1].get_text(strip=True)
                    
                    if label == 'Empresa':
                        data.company_name = value
                    elif label == 'Setor':
                        data.sector = value
                    elif label == 'Subsetor':
                        data.subsector = value
                    elif label == 'Cotação':
                        data.price = self._parse_number(value)
                    elif label == 'Valor de mercado':
                        data.market_cap = self._parse_large_number(value)
    
    def _normalize_label(self, label: str) -> str:
        """Normalize label by removing '?' prefix and whitespace."""
        label = label.strip()
        if label.startswith('?'):
            label = label[1:].strip()
        return label
    
    def _parse_indicators(self, soup: BeautifulSoup, data: FundamentalData) -> int:
        """Parse the indicators table. Returns number of successful parses."""
        # Find tables with indicator data
        tables = soup.find_all('table', class_='w728')
        
        parsed_count = 0
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) < 2:
                    continue
                
                # Check alternating pattern (label in even cells, value in odd cells)
                # Example: ['Dia', '0,42%', '?P/L', '6,31', '?LPA', '6,01']
                for i in range(0, len(cells) - 1, 2):
                    raw_label = cells[i].get_text(strip=True)
                    label = self._normalize_label(raw_label)
                    
                    if label in self.LABEL_MAP:
                        field_name = self.LABEL_MAP[label]
                        value = cells[i + 1].get_text(strip=True)
                        parsed = self._parse_number(value)
                        if parsed is not None:
                            setattr(data, field_name, parsed)
                            parsed_count += 1
                            
        return parsed_count
    
    def _parse_balance_sheet(self, soup: BeautifulSoup, data: FundamentalData):
        """Parse balance sheet data."""
        tables = soup.find_all('table', class_='w728')
        
        # Balance sheet labels (exact matches from fundamentus)
        balance_items = {
            'Ativo': 'total_assets',
            'Dív. Líquida': 'net_debt',
            'Patrim. Líq': 'total_equity',
        }
        
        # Income statement labels (for 12m column)
        # Note: Income statement has both 12m and 3m values, we only want 12m
        income_items = {
            'Receita Líquida': 'revenue',
            'EBIT': 'ebit',
            'Lucro Líquido': 'net_income',
        }
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                
                # Process cells in pairs (label, value)
                # For income statement, only take the first pair (12-month values)
                for i in range(0, len(cells) - 1, 2):
                    raw_label = cells[i].get_text(strip=True)
                    label = self._normalize_label(raw_label)
                    
                    # Check balance sheet items
                    if label in balance_items:
                        # Only set if not already set (avoid overwriting)
                        field = balance_items[label]
                        if getattr(data, field) is None:
                            value = self._parse_large_number(cells[i + 1].get_text(strip=True))
                            if value is not None:
                                setattr(data, field, value)
                    
                    # Check income statement items
                    if label in income_items:
                        # Only set if not already set (take first = 12-month)
                        field = income_items[label]
                        if getattr(data, field) is None:
                            value = self._parse_large_number(cells[i + 1].get_text(strip=True))
                            if value is not None:
                                setattr(data, field, value)
    
    def scrape_all(self, tickers: list, delay: float = 0.5) -> Dict[str, FundamentalData]:
        """Scrape multiple tickers with rate limiting.
        
        Args:
            tickers: List of ticker symbols (with or without .SA suffix)
            delay: Delay between requests in seconds (default 0.5s)
        
        Returns:
            Dictionary mapping tickers to FundamentalData objects
        """
        results = {}
        total = len(tickers)
        
        for i, ticker in enumerate(tickers, 1):
            clean_ticker = ticker.replace('.SA', '')
            print(f"Scraping {clean_ticker} ({i}/{total})...")
            
            data = self.scrape_ticker(ticker)
            if data:
                results[clean_ticker] = data
            
            if i < total:  # Don't delay after last request
                time.sleep(delay)
        
        return results
    
    def save_cache(self, data: Dict[str, FundamentalData], filename: str = 'fundamentals_cache.json'):
        """Save scraped data to cache file."""
        cache_path = self.cache_dir / filename
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'total_stocks': len(data),
            'data': {ticker: fd.to_dict() for ticker, fd in data.items()}
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(data)} stocks to {cache_path}")
    
    def load_cache(self, filename: str = 'fundamentals_cache.json') -> Optional[Dict[str, FundamentalData]]:
        """Load cached fundamental data."""
        cache_path = self.cache_dir / filename
        
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            results = {}
            for ticker, data_dict in cache_data.get('data', {}).items():
                results[ticker] = FundamentalData(**data_dict)
            
            print(f"Loaded {len(results)} stocks from cache ({cache_data.get('timestamp', 'unknown')})")
            return results
        except Exception as e:
            print(f"Error loading cache: {e}")
            return None
    
    def is_cache_stale(self, filename: str = 'fundamentals_cache.json', max_age_hours: int = 24) -> bool:
        """Check if cache is stale (older than max_age_hours)."""
        cache_path = self.cache_dir / filename
        
        if not cache_path.exists():
            return True
        
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            timestamp = datetime.fromisoformat(cache_data['timestamp'])
            age = datetime.now() - timestamp
            
            return age.total_seconds() > max_age_hours * 3600
        except Exception:
            return True


if __name__ == '__main__':
    # Test with a few tickers
    scraper = FundamentusScraper()
    
    test_tickers = ['PETR4', 'VALE3', 'ITUB4']
    
    for ticker in test_tickers:
        data = scraper.scrape_ticker(ticker)
        if data:
            print(f"\n{data.ticker}:")
            print(f"  P/E: {data.pe_ratio}")
            print(f"  P/B: {data.pb_ratio}")
            print(f"  ROE: {data.roe}%")
            print(f"  ROIC: {data.roic}%")
            print(f"  Debt/Equity: {data.debt_equity}")
            print(f"  Div Yield: {data.div_yield}%")
