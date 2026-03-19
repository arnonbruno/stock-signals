"""
Value Investing Scorer based on Graham, Lynch, and Greenblatt principles.

Implements fundamental scoring methods:
- Graham's Defensive Investor criteria
- Lynch's Growth at Reasonable Price (GARP)
- Greenblatt's Magic Formula (ROC + Earnings Yield)
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import math


@dataclass
class GrahamScore:
    """Graham's Defensive Investor scoring."""
    total_score: float
    max_score: float
    passed_criteria: List[str]
    failed_criteria: List[str]
    
    @property
    def percentage(self) -> float:
        return (self.total_score / self.max_score * 100) if self.max_score > 0 else 0


@dataclass
class LynchScore:
    """Lynch's GARP scoring."""
    peg_ratio: Optional[float]
    growth_rate: Optional[float]
    pe_ratio: Optional[float]
    score: float  # 0-100
    rating: str  # Excellent, Good, Fair, Poor


@dataclass
class GreenblattScore:
    """Greenblatt's Magic Formula ranking."""
    earnings_yield: Optional[float]
    return_on_capital: Optional[float]
    combined_rank: float  # Lower is better (ranking)
    score: float  # 0-100


@dataclass
class FundamentalScore:
    """Combined fundamental score."""
    ticker: str
    graham: GrahamScore
    lynch: Optional[LynchScore]
    greenblatt: Optional[GreenblattScore]
    
    # Composite scores
    value_score: float  # 0-100 (lower valuations = higher score)
    quality_score: float  # 0-100 (profitability, financial health)
    growth_score: float  # 0-100 (revenue growth)
    
    # Overall
    composite_score: float  # 0-100 weighted average
    grade: str  # A, B, C, D, F
    
    # Signals
    is_value: bool  # Meets value criteria
    is_quality: bool  # High profitability
    is_growth: bool  # Growing revenues
    is_undervalued: bool  # Below intrinsic value estimate
    is_holding_or_financial: bool  # Holding/financial institution
    
    # Notes
    strengths: List[str]
    weaknesses: List[str]


class FundamentalScorer:
    """
    Scores stocks based on fundamental analysis principles.
    
    Weighted approach:
    - Graham (30%): Safety, defensive criteria
    - Lynch (35%): GARP, PEG ratio
    - Greenblatt (35%): Magic Formula ranking
    """
    
    # Graham's defensive investor thresholds
    GRAHAM_THRESHOLDS = {
        'pe_max': 15,  # P/E <= 15
        'pb_max': 1.5,  # P/B <= 1.5
        'current_ratio_min': 2.0,  # Current ratio >= 2
        'debt_equity_max': 1.0,  # D/E <= 1.0
        'dividend_yield_min': 2.0,  # Div yield >= 2% (optional)
        'earnings_stability': True,  # Positive earnings
        'earnings_growth_min': 0.0,  # Some earnings growth
    }
    
    # Lynch PEG thresholds
    LYNCH_PEG_THRESHOLDS = {
        'excellent': 0.5,  # PEG < 0.5: Excellent
        'good': 1.0,  # PEG < 1.0: Good
        'fair': 1.5,  # PEG < 1.5: Fair
        'poor': 2.0,  # PEG < 2.0: Poor
        # PEG >= 2.0: Very Poor
    }
    
    def __init__(self):
        self._greenblatt_rankings = {}
    
    def score_graham(self, data) -> GrahamScore:
        """
        Score based on Graham's Defensive Investor criteria.
        
        Criteria (each worth 1 point):
        1. P/E ratio <= 15
        2. P/B ratio <= 1.5
        3. P/E * P/B <= 22.5 (Graham's formula)
        4. Current ratio >= 2
        5. Debt/Equity <= 1
        6. Positive earnings
        7. Dividend yield >= 2% (bonus, not required)
        """
        passed = []
        failed = []
        score = 0
        max_score = 7
        
        # P/E ratio
        if data.pe_ratio is not None:
            if data.pe_ratio <= self.GRAHAM_THRESHOLDS['pe_max']:
                passed.append(f"P/E ({data.pe_ratio:.1f}) <= 15")
                score += 1
            else:
                failed.append(f"P/E ({data.pe_ratio:.1f}) > 15")
        else:
            failed.append("P/E: N/A")
        
        # P/B ratio
        if data.pb_ratio is not None:
            if data.pb_ratio <= self.GRAHAM_THRESHOLDS['pb_max']:
                passed.append(f"P/B ({data.pb_ratio:.2f}) <= 1.5")
                score += 1
            else:
                failed.append(f"P/B ({data.pb_ratio:.2f}) > 1.5")
        else:
            failed.append("P/B: N/A")
        
        # Graham number: P/E * P/B <= 22.5
        if data.pe_ratio is not None and data.pb_ratio is not None:
            graham_multiple = data.pe_ratio * data.pb_ratio
            if graham_multiple <= 22.5:
                passed.append(f"P/E × P/B ({graham_multiple:.1f}) <= 22.5")
                score += 1
            else:
                failed.append(f"P/E × P/B ({graham_multiple:.1f}) > 22.5")
        
        # Current ratio
        if data.current_ratio is not None:
            if data.current_ratio >= self.GRAHAM_THRESHOLDS['current_ratio_min']:
                passed.append(f"Current ratio ({data.current_ratio:.2f}) >= 2")
                score += 1
            else:
                failed.append(f"Current ratio ({data.current_ratio:.2f}) < 2")
        else:
            failed.append("Current ratio: N/A")
        
        # Debt/Equity
        if data.debt_equity is not None:
            if data.debt_equity <= self.GRAHAM_THRESHOLDS['debt_equity_max']:
                passed.append(f"D/E ({data.debt_equity:.2f}) <= 1.0")
                score += 1
            else:
                failed.append(f"D/E ({data.debt_equity:.2f}) > 1.0")
        else:
            failed.append("D/E: N/A")
        
        # Positive earnings
        if data.eps is not None and data.eps > 0:
            passed.append(f"Positive EPS ({data.eps:.2f})")
            score += 1
        else:
            failed.append("Non-positive EPS")
        
        # Dividend yield (bonus)
        if data.div_yield is not None:
            if data.div_yield >= self.GRAHAM_THRESHOLDS['dividend_yield_min']:
                passed.append(f"Div yield ({data.div_yield:.1f}%) >= 2%")
                score += 1
            else:
                failed.append(f"Div yield ({data.div_yield:.1f}%) < 2%")
        else:
            failed.append("Div yield: N/A")
        
        return GrahamScore(
            total_score=score,
            max_score=max_score,
            passed_criteria=passed,
            failed_criteria=failed
        )
    
    def is_holding_or_financial(self, data) -> bool:
        """
        Detect if stock is a holding company or financial institution.
        
        These companies have different metrics:
        - EBIT is not meaningful (holdings get income from dividends)
        - EV/EBIT is inflated
        - ROIC is not applicable
        - Growth expectations are lower (stable dividend payers)
        
        Detection criteria:
        1. EV/EBIT > 100 (indicating EBIT is very low relative to value)
        2. P/EBIT > 100
        3. Net margin > 50% (dividend income has high margin)
        4. Sector is "Intermediários Financeiros"
        """
        # Check EV/EBIT or P/EBIT threshold
        if data.ev_ebit is not None and data.ev_ebit > 100:
            return True
        if data.p_ebit is not None and data.p_ebit > 100:
            return True
        
        # Check for high net margin (dividend income)
        if data.net_margin is not None and data.net_margin > 50:
            return True
        
        # Check sector
        if data.sector and 'financeiro' in data.sector.lower():
            return True
        
        return False
    
    def score_holding_or_financial(self, data, graham: GrahamScore) -> Tuple[float, Optional[LynchScore], Optional[GreenblattScore]]:
        """
        Score holdings and financial institutions using alternative metrics.
        
        For holdings/financials:
        - Graham: 50% (value + dividend focus)
        - Quality: 50% (ROE, dividend yield, stability)
        - Skip Greenblatt (EBIT-based metrics don't apply)
        - Lower growth expectations
        
        Returns:
            Tuple of (composite_score, lynch_score, greenblatt_score)
        """
        # Graham score (50% weight)
        graham_score = graham.percentage
        
        # Quality score based on ROE and Dividend (50% weight)
        quality_score = 50  # Start neutral
        
        # ROE is key for financials
        if data.roe is not None:
            if data.roe > 20:
                quality_score += 20
            elif data.roe > 15:
                quality_score += 15
            elif data.roe > 10:
                quality_score += 10
            elif data.roe > 5:
                quality_score += 5
            else:
                quality_score -= 10
        
        # Dividend yield is important for holdings
        if data.div_yield is not None:
            if data.div_yield > 6:
                quality_score += 15
            elif data.div_yield > 4:
                quality_score += 10
            elif data.div_yield > 2:
                quality_score += 5
        
        # Low debt is good
        if data.debt_equity is not None:
            if data.debt_equity < 0.5:
                quality_score += 10
            elif data.debt_equity < 1.0:
                quality_score += 5
        
        quality_score = max(0, min(100, quality_score))
        
        # Composite: 50% Graham + 50% Quality
        composite = (graham_score * 0.50) + (quality_score * 0.50)
        
        # Skip Lynch and Greenblatt for holdings
        return composite, None, None
    
    def score_lynch(self, data) -> Optional[LynchScore]:
        """
        Score based on Lynch's GARP (Growth at Reasonable Price) approach.
        
        PEG = P/E / Growth Rate
        - PEG < 0.5: Excellent (100 points)
        - PEG < 1.0: Good (80 points)
        - PEG < 1.5: Fair (60 points)
        - PEG < 2.0: Poor (40 points)
        - PEG >= 2.0: Very Poor (20 points)
        """
        if data.pe_ratio is None or data.pe_ratio <= 0:
            return None
        
        growth_rate = data.revenue_growth_5y
        if growth_rate is None or growth_rate <= 0:
            # If no growth data, can't calculate PEG
            return None
        
        peg_ratio = data.pe_ratio / growth_rate
        
        # Score based on PEG
        if peg_ratio < 0.5:
            score = 100
            rating = "Excellent"
        elif peg_ratio < 1.0:
            score = 80
            rating = "Good"
        elif peg_ratio < 1.5:
            score = 60
            rating = "Fair"
        elif peg_ratio < 2.0:
            score = 40
            rating = "Poor"
        else:
            score = 20
            rating = "Very Poor"
        
        return LynchScore(
            peg_ratio=peg_ratio,
            growth_rate=growth_rate,
            pe_ratio=data.pe_ratio,
            score=score,
            rating=rating
        )
    
    def score_greenblatt(self, data) -> Optional[GreenblattScore]:
        """
        Score based on Greenblatt's Magic Formula.
        
        Components:
        1. Earnings Yield = EBIT / EV (or 1/PE if EV not available)
        2. Return on Capital = EBIT / (Net Working Capital + Net Fixed Assets)
           Approximated by ROIC if available
        
        Higher earnings yield + higher ROC = better score
        """
        # Calculate earnings yield
        earnings_yield = None
        
        if data.ev_ebit is not None and data.ev_ebit > 0:
            earnings_yield = (1 / data.ev_ebit) * 100
        elif data.p_ebit is not None and data.p_ebit > 0:
            earnings_yield = (1 / data.p_ebit) * 100
        elif data.pe_ratio is not None and data.pe_ratio > 0:
            earnings_yield = (1 / data.pe_ratio) * 100
        
        # Get ROC (ROIC is our proxy)
        roc = data.roic
        
        if earnings_yield is None or roc is None:
            return None
        
        # Score each component
        # Earnings yield: 0-50 points
        # Higher yield = better (inverse relationship with valuation)
        ey_score = min(50, earnings_yield * 10)  # 5% yield = 50 points
        
        # ROC: 0-50 points
        # Higher ROC = better
        roc_score = min(50, roc)  # 50% ROC = 50 points
        
        total_score = ey_score + roc_score
        
        return GreenblattScore(
            earnings_yield=earnings_yield,
            return_on_capital=roc,
            combined_rank=0,  # Will be set during batch ranking
            score=total_score
        )
    
    def calculate_value_score(self, data, graham: GrahamScore) -> float:
        """Calculate pure value score (0-100) based on valuation metrics."""
        score = graham.percentage  # Start with Graham percentage
        
        # P/E penalty - stocks with high P/E shouldn't get perfect value scores
        if data.pe_ratio is not None and data.pe_ratio > 15:
            penalty = 15 if data.pe_ratio <= 20 else 30
            score -= penalty
        
        # Bonus for very low valuations
        if data.pe_ratio is not None:
            if data.pe_ratio < 8:
                score += 10  # Very cheap
            elif data.pe_ratio < 10:
                score += 5
        
        if data.pb_ratio is not None:
            if data.pb_ratio < 0.8:
                score += 10  # Trading below book value
            elif data.pb_ratio < 1.0:
                score += 5
        
        # EV/EBITDA bonus
        if data.ev_ebitda is not None:
            if data.ev_ebitda < 5:
                score += 10
            elif data.ev_ebitda < 8:
                score += 5
        
        return min(100, max(0, score))
    
    def calculate_quality_score(self, data) -> float:
        """Calculate quality score (0-100) based on profitability and financial health."""
        score = 50  # Start at neutral
        
        # ROE
        if data.roe is not None:
            if data.roe > 20:
                score += 15
            elif data.roe > 15:
                score += 10
            elif data.roe > 10:
                score += 5
            elif data.roe < 5:
                score -= 10
        
        # ROIC
        if data.roic is not None:
            if data.roic > 15:
                score += 15
            elif data.roic > 10:
                score += 10
            elif data.roic > 5:
                score += 5
            elif data.roic < 0:
                score -= 15
        
        # Margins
        if data.net_margin is not None:
            if data.net_margin > 15:
                score += 10
            elif data.net_margin > 10:
                score += 5
            elif data.net_margin < 5:
                score -= 5
        
        # Debt
        if data.debt_equity is not None:
            if data.debt_equity < 0.3:
                score += 10
            elif data.debt_equity < 0.5:
                score += 5
            elif data.debt_equity > 1.0:
                score -= 10
            elif data.debt_equity > 2.0:
                score -= 20
        
        # Current ratio
        if data.current_ratio is not None:
            if data.current_ratio > 2:
                score += 5
            elif data.current_ratio < 1:
                score -= 10
        
        return max(0, min(100, score))
    
    def calculate_growth_score(self, data) -> float:
        """Calculate growth score (0-100) based on revenue growth."""
        score = 50  # Neutral
        
        if data.revenue_growth_5y is not None:
            growth = data.revenue_growth_5y
            
            if growth > 20:
                score += 30
            elif growth > 15:
                score += 20
            elif growth > 10:
                score += 15
            elif growth > 5:
                score += 10
            elif growth > 0:
                score += 5
            elif growth < -5:
                score -= 15
            elif growth < 0:
                score -= 5
        
        return max(0, min(100, score))
    
    def get_grade(self, score: float) -> str:
        """Convert numerical score to letter grade."""
        if score >= 80:
            return 'A'
        elif score >= 65:
            return 'B'
        elif score >= 50:
            return 'C'
        elif score >= 35:
            return 'D'
        else:
            return 'F'
    
    def score(self, data) -> FundamentalScore:
        """
        Calculate comprehensive fundamental score.
        
        Weights:
        - Graham (value): 30%
        - Lynch (GARP): 35%
        - Greenblatt (Magic Formula): 35%
        
        For holdings/financials:
        - Graham: 50%
        - Quality (ROE + Div): 50%
        - Skip Lynch/Greenblatt (EBIT-based metrics don't apply)
        """
        # Individual scores
        graham = self.score_graham(data)
        
        # Check if this is a holding or financial institution
        is_holding = self.is_holding_or_financial(data)
        
        if is_holding:
            composite, lynch, greenblatt = self.score_holding_or_financial(data, graham)
        else:
            lynch = self.score_lynch(data)
            greenblatt = self.score_greenblatt(data)
            
            # Composite calculation
            weights = {'graham': 0.30, 'lynch': 0.35, 'greenblatt': 0.35}
            
            composite = 0
            total_weight = 0
            
            # Graham (always available)
            composite += graham.percentage * weights['graham']
            total_weight += weights['graham']
            
            # Lynch (optional)
            if lynch:
                composite += lynch.score * weights['lynch']
                total_weight += weights['lynch']
            else:
                # Redistribute Lynch's weight to Graham (not Greenblatt)
                composite += graham.percentage * weights['lynch']
                total_weight += weights['lynch']
            
            # Greenblatt (optional)
            if greenblatt:
                composite += greenblatt.score * weights['greenblatt']
                total_weight += weights['greenblatt']
            else:
                # Redistribute to Graham
                composite += graham.percentage * weights['greenblatt']
                total_weight += weights['greenblatt']
            
            composite = composite / total_weight if total_weight > 0 else 0
        
        # Calculate component scores
        value_score = self.calculate_value_score(data, graham)
        quality_score = self.calculate_quality_score(data)
        growth_score = self.calculate_growth_score(data)
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        # Check key metrics for strengths/weaknesses
        if data.pe_ratio and data.pe_ratio < 10:
            strengths.append(f"Low P/E ({data.pe_ratio:.1f})")
        elif data.pe_ratio and data.pe_ratio > 25:
            weaknesses.append(f"High P/E ({data.pe_ratio:.1f})")
        
        if data.pb_ratio and data.pb_ratio < 1.0:
            strengths.append(f"Trading below book value (P/B: {data.pb_ratio:.2f})")
        
        if data.roe and data.roe > 20:
            strengths.append(f"High ROE ({data.roe:.1f}%)")
        elif data.roe and data.roe < 8:
            weaknesses.append(f"Low ROE ({data.roe:.1f}%)")
        
        if data.roic and data.roic > 15:
            strengths.append(f"Excellent ROIC ({data.roic:.1f}%)")
        
        if data.debt_equity and data.debt_equity < 0.3:
            strengths.append(f"Low debt (D/E: {data.debt_equity:.2f})")
        elif data.debt_equity and data.debt_equity > 1.5:
            weaknesses.append(f"High debt (D/E: {data.debt_equity:.2f})")
        
        if data.div_yield and data.div_yield > 5:
            strengths.append(f"High dividend yield ({data.div_yield:.1f}%)")
        
        if lynch and lynch.peg_ratio and lynch.peg_ratio < 1.0:
            strengths.append(f"Attractive PEG ({lynch.peg_ratio:.2f})")
        
        # Determine flags
        is_value = value_score >= 60
        is_quality = quality_score >= 65
        is_growth = growth_score >= 60
        is_undervalued = graham.percentage >= 60
        
        return FundamentalScore(
            ticker=data.ticker,
            graham=graham,
            lynch=lynch,
            greenblatt=greenblatt,
            value_score=value_score,
            quality_score=quality_score,
            growth_score=growth_score,
            composite_score=composite,
            grade=self.get_grade(composite),
            is_value=is_value,
            is_quality=is_quality,
            is_growth=is_growth,
            is_undervalued=is_undervalued,
            is_holding_or_financial=is_holding,
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    def rank_greenblatt(self, scores: List[FundamentalScore]) -> None:
        """
        Apply Greenblatt ranking to a list of scores.
        
        Ranks stocks by:
        1. Earnings yield (higher = better, rank 1 = best)
        2. Return on capital (higher = better, rank 1 = best)
        
        Combined rank = rank(EY) + rank(ROC)
        Lower combined rank = better
        """
        # Filter stocks with Greenblatt scores
        valid_scores = [s for s in scores if s.greenblatt is not None]
        
        if not valid_scores:
            return
        
        # Rank by earnings yield (higher = better)
        sorted_by_ey = sorted(
            valid_scores,
            key=lambda s: s.greenblatt.earnings_yield or 0,
            reverse=True
        )
        ey_ranks = {s.ticker: i + 1 for i, s in enumerate(sorted_by_ey)}
        
        # Rank by ROC (higher = better)
        sorted_by_roc = sorted(
            valid_scores,
            key=lambda s: s.greenblatt.return_on_capital or 0,
            reverse=True
        )
        roc_ranks = {s.ticker: i + 1 for i, s in enumerate(sorted_by_roc)}
        
        # Combined rank
        for s in valid_scores:
            combined = ey_ranks.get(s.ticker, 0) + roc_ranks.get(s.ticker, 0)
            s.greenblatt.combined_rank = combined


def score_fundamentals(fundamentals: Dict, scorer: Optional[FundamentalScorer] = None) -> List[FundamentalScore]:
    """
    Score a dictionary of fundamental data.
    
    Args:
        fundamentals: Dict mapping ticker to FundamentalData
        scorer: Optional FundamentalScorer instance
    
    Returns:
        List of FundamentalScore objects, sorted by composite score descending
    """
    if scorer is None:
        scorer = FundamentalScorer()
    
    scores = [scorer.score(data) for data in fundamentals.values()]
    
    # Apply Greenblatt ranking
    scorer.rank_greenblatt(scores)
    
    # Sort by composite score
    scores.sort(key=lambda s: s.composite_score, reverse=True)
    
    return scores
