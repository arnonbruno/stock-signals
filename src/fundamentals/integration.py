"""
Fundamental analysis integration for stock signals.

Combines fundamental scores with technical analysis to provide
a comprehensive investment decision framework.
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from pathlib import Path
import json
from datetime import datetime


@dataclass
class IntegratedScore:
    """Combined technical + fundamental score."""
    ticker: str
    
    # Technical scores (from existing system)
    technical_score: float  # 0-100
    trend: str  # UPTREND, DOWNTREND, SIDEWAYS
    confidence: float  # 0-1
    
    # Fundamental scores
    fundamental_score: float  # 0-100
    fundamental_grade: str  # A, B, C, D, F
    value_score: float  # 0-100
    quality_score: float  # 0-100
    growth_score: float  # 0-100
    
    # Combined
    composite_score: float  # Weighted combination
    recommendation: str  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    
    # Flags
    is_value_pick: bool  # Good value + reasonable technicals
    is_quality_pick: bool  # High quality + reasonable technicals
    is_momentum_pick: bool  # Strong technicals + acceptable fundamentals
    is_avoid: bool  # Poor fundamentals + poor technicals
    
    # Key metrics for display
    pe_ratio: Optional[float]
    pb_ratio: Optional[float]
    roe: Optional[float]
    div_yield: Optional[float]
    
    # Notes
    strengths: List[str]
    weaknesses: List[str]
    action_notes: List[str]


class FundamentalIntegrator:
    """
    Integrates fundamental analysis with technical signals.
    
    Weights (configurable):
    - Technical: 50% (trend, momentum, volume)
    - Fundamental: 50% (value, quality, growth)
    
    Special adjustments:
    - Strong fundamentals can boost weak technical signals (contrarian)
    - Strong technicals can overcome weak fundamentals (momentum)
    - Poor fundamentals cap upside regardless of technicals
    """
    
    # Weights for composite score
    TECHNICAL_WEIGHT = 0.50
    FUNDAMENTAL_WEIGHT = 0.50
    
    # Thresholds
    STRONG_FUNDAMENTAL_THRESHOLD = 70  # Fundamental score above this is "strong"
    WEAK_FUNDAMENTAL_THRESHOLD = 40  # Below this is "weak"
    STRONG_TECHNICAL_THRESHOLD = 60
    WEAK_TECHNICAL_THRESHOLD = 40
    
    def __init__(self, fundamentals_path: Optional[Path] = None):
        """Initialize with path to fundamental scores cache."""
        if fundamentals_path:
            self.fundamentals_path = fundamentals_path
        else:
            # Find the project root (where data/ folder is)
            # Walk up from this file until we find data/fundamentals/
            current = Path(__file__).resolve().parent
            for _ in range(5):  # Max 5 levels up
                candidate = current / 'data' / 'fundamentals' / 'fundamental_scores.json'
                if candidate.exists():
                    self.fundamentals_path = candidate
                    break
                current = current.parent
            else:
                # Fallback to relative path
                self.fundamentals_path = Path(__file__).parent.parent.parent / 'data' / 'fundamentals' / 'fundamental_scores.json'
        
        self._fundamentals_cache = None
    
    def load_fundamentals(self) -> Dict:
        """Load fundamental scores from cache."""
        if self._fundamentals_cache is not None:
            return self._fundamentals_cache
        
        if not self.fundamentals_path.exists():
            print(f"Warning: Fundamentals cache not found at {self.fundamentals_path}")
            return {}
        
        with open(self.fundamentals_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Index by ticker
        self._fundamentals_cache = {
            s['ticker']: s for s in data.get('scores', [])
        }
        
        return self._fundamentals_cache
    
    def get_fundamental_score(self, ticker: str) -> Optional[Dict]:
        """Get fundamental score for a ticker."""
        fundamentals = self.load_fundamentals()
        # Try with and without .SA suffix
        clean_ticker = ticker.replace('.SA', '')
        return fundamentals.get(clean_ticker)
    
    def integrate(self, ticker: str, technical_score: float, 
                  trend: str, confidence: float,
                  strengths: Optional[List[str]] = None,
                  weaknesses: Optional[List[str]] = None) -> IntegratedScore:
        """
        Combine technical and fundamental analysis for a stock.
        
        Args:
            ticker: Stock ticker
            technical_score: Technical analysis score (0-100)
            trend: Current trend (UPTREND, DOWNTREND, SIDEWAYS)
            confidence: Technical confidence (0-1)
            strengths: Technical strengths
            weaknesses: Technical weaknesses
        
        Returns:
            IntegratedScore with combined analysis
        """
        # Get fundamental data
        fund_data = self.get_fundamental_score(ticker)
        
        # Default fundamental scores if not available
        if fund_data:
            fund_score = fund_data.get('composite_score', 50)
            fund_grade = fund_data.get('grade', 'C')
            value_score = fund_data.get('value_score', 50)
            quality_score = fund_data.get('quality_score', 50)
            growth_score = fund_data.get('growth_score', 50)
            fund_strengths = fund_data.get('strengths', [])
            fund_weaknesses = fund_data.get('weaknesses', [])
        else:
            # No fundamental data available
            fund_score = 50  # Neutral
            fund_grade = 'N/A'
            value_score = 50
            quality_score = 50
            growth_score = 50
            fund_strengths = []
            fund_weaknesses = ['No fundamental data available']
        
        # Calculate composite score with adjustments
        composite = self._calculate_composite(
            technical_score, fund_score, trend, 
            value_score, quality_score
        )
        
        # Determine recommendation
        # Load raw fundamental data for metrics and P/E validation
        pe_ratio = None
        pb_ratio = None
        roe = None
        div_yield = None
        
        raw_path = self.fundamentals_path.parent / 'fundamentals_cache.json'
        if raw_path.exists():
            with open(raw_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            ticker_data = raw_data.get('data', {}).get(ticker.replace('.SA', ''), {})
            pe_ratio = ticker_data.get('pe_ratio')
            pb_ratio = ticker_data.get('pb_ratio')
            roe = ticker_data.get('roe')
            div_yield = ticker_data.get('div_yield')
        
        rec_data = {'pe_ratio': pe_ratio or (fund_data.get('pe_ratio') if fund_data else None),
                     'value_score': value_score}
        recommendation = self._get_recommendation(composite, trend, fund_score, rec_data)
        
        # Determine flags
        is_value_pick = self._is_value_pick(technical_score, value_score, quality_score)
        is_quality_pick = self._is_quality_pick(technical_score, quality_score)
        is_momentum_pick = self._is_momentum_pick(technical_score, fund_score)
        is_avoid = self._is_avoid(technical_score, fund_score)
        
        # Generate action notes
        action_notes = self._generate_action_notes(
            trend, fund_score, value_score, quality_score,
            is_value_pick, is_quality_pick, is_momentum_pick
        )
        
        # Combine strengths/weaknesses
        all_strengths = (strengths or []) + fund_strengths
        all_weaknesses = (weaknesses or []) + fund_weaknesses
        
        return IntegratedScore(
            ticker=ticker,
            technical_score=technical_score,
            trend=trend,
            confidence=confidence,
            fundamental_score=fund_score,
            fundamental_grade=fund_grade,
            value_score=value_score,
            quality_score=quality_score,
            growth_score=growth_score,
            composite_score=composite,
            recommendation=recommendation,
            is_value_pick=is_value_pick,
            is_quality_pick=is_quality_pick,
            is_momentum_pick=is_momentum_pick,
            is_avoid=is_avoid,
            pe_ratio=pe_ratio,
            pb_ratio=pb_ratio,
            roe=roe,
            div_yield=div_yield,
            strengths=all_strengths,
            weaknesses=all_weaknesses,
            action_notes=action_notes
        )
    
    def _calculate_composite(self, tech_score: float, fund_score: float,
                            trend: str, value_score: float, 
                            quality_score: float) -> float:
        """Calculate weighted composite score with adjustments."""
        # Base composite
        composite = (tech_score * self.TECHNICAL_WEIGHT + 
                    fund_score * self.FUNDAMENTAL_WEIGHT)
        
        # Adjustment 1: High quality can boost score in uptrends
        if trend == 'UPTREND' and quality_score >= 70:
            composite += 5  # Quality momentum boost
        
        # Adjustment 2: Poor fundamentals cap upside
        if fund_score < 35:
            composite = min(composite, 60)  # Cap at 60
        
        # Adjustment 3: Very poor fundamentals in downtrend = severe penalty
        if fund_score < 30 and trend == 'DOWNTREND':
            composite -= 10  # Avoid trap
        
        # Adjustment 4: Deep value in sideways market = contrarian opportunity
        if value_score >= 80 and trend == 'SIDEWAYS':
            composite += 5  # Contrarian value play
        
        return max(0, min(100, composite))
    
    def _get_recommendation(self, composite: float, trend: str, 
                           fund_score: float, data: Optional[Dict] = None) -> str:
        """Generate recommendation based on composite score."""
        # Base recommendation on composite
        if composite >= 80:  # Raised from 75 to require stronger fundamentals
            base_rec = 'STRONG_BUY'
        elif composite >= 70:
            base_rec = 'BUY'
        elif composite >= 40:
            base_rec = 'HOLD'
        elif composite >= 25:
            base_rec = 'SELL'
        else:
            base_rec = 'STRONG_SELL'
        
        # Adjust for fundamental concerns
        if fund_score < 30 and base_rec in ['STRONG_BUY', 'BUY']:
            if base_rec == 'STRONG_BUY':
                return 'BUY'
            else:
                return 'HOLD'
        
        # P/E validation - don't recommend STRONG_BUY for expensive stocks
        pe_ratio = None
        value_score = None
        if data is not None:
            pe_ratio = data.get('pe_ratio')
            value_score = data.get('value_score')
        
        if pe_ratio is not None and pe_ratio > 20:
            if base_rec == 'STRONG_BUY':
                return 'BUY'
            elif base_rec == 'BUY':
                if value_score is not None and value_score >= 80:
                    base_rec = 'BUY'
                else:
                    base_rec = 'HOLD'
            if base_rec == 'BUY' and pe_ratio > 25:
                return 'HOLD'
        
        return base_rec
    
    def _is_value_pick(self, tech_score: float, value_score: float, 
                       quality_score: float) -> bool:
        """Check if this is a value investment opportunity."""
        return (value_score >= 70 and 
                quality_score >= 50 and 
                tech_score >= 35)
    
    def _is_quality_pick(self, tech_score: float, quality_score: float) -> bool:
        """Check if this is a quality compounder."""
        return quality_score >= 75 and tech_score >= 45
    
    def _is_momentum_pick(self, tech_score: float, fund_score: float) -> bool:
        """Check if this is a momentum play."""
        return tech_score >= 70 and fund_score >= 45
    
    def _is_avoid(self, tech_score: float, fund_score: float) -> bool:
        """Check if this stock should be avoided."""
        return tech_score < 35 and fund_score < 40
    
    def _generate_action_notes(self, trend: str, fund_score: float,
                               value_score: float, quality_score: float,
                               is_value: bool, is_quality: bool,
                               is_momentum: bool) -> List[str]:
        """Generate actionable notes for the investor."""
        notes = []
        
        if is_value:
            notes.append("VALUE PLAY: Undervalued with acceptable quality")
        
        if is_quality:
            notes.append("QUALITY PLAY: High ROE/ROIC compounder")
        
        if is_momentum:
            notes.append("MOMENTUM PLAY: Strong technicals, acceptable fundamentals")
        
        if fund_score < 35:
            notes.append("⚠️ FUNDAMENTAL CONCERN: Poor financial health")
        
        if value_score >= 80 and trend != 'UPTREND':
            notes.append("💡 CONTRARIAN: Deep value, wait for reversal signals")
        
        if quality_score >= 70 and trend == 'UPTREND':
            notes.append("✅ QUALITY MOMENTUM: High quality in uptrend")
        
        return notes


def format_integrated_signal(score: IntegratedScore) -> str:
    """Format integrated score for display/alerts."""
    lines = []
    
    # Header
    rec_emoji = {
        'STRONG_BUY': '🚀',
        'BUY': '🟢',
        'HOLD': '🟡',
        'SELL': '🔴',
        'STRONG_SELL': '💀'
    }
    
    emoji = rec_emoji.get(score.recommendation, '❓')
    lines.append(f"{emoji} {score.ticker} - {score.recommendation}")
    lines.append(f"Composite: {score.composite_score:.1f} | Tech: {score.technical_score:.1f} | Fund: {score.fundamental_score:.1f} ({score.fundamental_grade})")
    
    # Key metrics
    metrics = []
    if score.pe_ratio:
        metrics.append(f"P/E: {score.pe_ratio:.1f}")
    if score.pb_ratio:
        metrics.append(f"P/B: {score.pb_ratio:.2f}")
    if score.roe:
        metrics.append(f"ROE: {score.roe:.1f}%")
    if score.div_yield:
        metrics.append(f"Div: {score.div_yield:.1f}%")
    
    if metrics:
        lines.append(" | ".join(metrics))
    
    # Component scores
    lines.append(f"Value: {score.value_score:.0f} | Quality: {score.quality_score:.0f} | Growth: {score.growth_score:.0f}")
    
    # Flags
    flags = []
    if score.is_value_pick:
        flags.append("💰 Value")
    if score.is_quality_pick:
        flags.append("⭐ Quality")
    if score.is_momentum_pick:
        flags.append("📈 Momentum")
    if score.is_avoid:
        flags.append("⚠️ AVOID")
    
    if flags:
        lines.append(" | ".join(flags))
    
    # Action notes
    if score.action_notes:
        lines.append("")
        for note in score.action_notes:
            lines.append(f"  {note}")
    
    return "\n".join(lines)