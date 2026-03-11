#!/usr/bin/env python3
"""
Expand ticker list to full IBOV + SMLL coverage.

IBOV: ~85-90 stocks (varies quarterly)
SMLL: ~80-100 stocks (varies quarterly)

Strategy:
1. Start with current validated tickers (65)
2. Add known IBOV constituents not in list
3. Add known SMLL constituents not in list
4. Validate liquidity for new additions
"""

import yfinance as yf
import json
from pathlib import Path
from datetime import datetime

# Known IBOV constituents (Feb 2025) - beyond current 50
ADDITIONAL_IBOV = [
    # Banks/Financials
    "BBDC3.SA", "BPAC11.SA", "SANB3.SA", "BPAC3.SA", "BPAC5.SA",
    # Energy/Utilities  
    "CMIG3.SA", "CPLE6.SA", "EGIE3.SA", "TAEE11.SA", "AESB3.SA",
    # Materials/Mining
    "CSNA3.SA", "USIM3.SA", "GOAU4.SA", "BRAP3.SA", "BRAP4.SA",
    # Consumer
    "CRFB3.SA", "NTCO3.SA", "VIIA3.SA", "AMAR3.SA", "LJQQ3.SA",
    # Industrial
    "EMBR3.SA", "CCRO3.SA", "ECOR3.SA", "ALSO3.SA", "ODPV3.SA",
    # Healthcare
    "FLRY3.SA", "RDOR3.SA", "GNDI3.SA", "QUAL3.SA",
    # Tech
    "TOTS3.SA", "LWSA3.SA", "SUZB3.SA", "KLBN3.SA", "KLBN4.SA",
    # Real Estate
    "MRVE3.SA", "CYRE3.SA", "JHSF3.SA", "EZTC3.SA", "DIRR3.SA",
    # Infrastructure
    "RAIL3.SA", "CCPR3.SA", "ALUP11.SA", "TIET11.SA",
    # Retail
    "MGLU3.SA", "ARZZ3.SA", "GUAR3.SA", "CEAB3.SA",
]

# Known SMLL constituents - beyond current 15
ADDITIONAL_SMLL = [
    # Small cap real estate
    "HBRE3.SA", "MTSA3.SA", "ALPK3.SA", "DESK3.SA", "OSXB3.SA",
    # Small cap industrials
    "POMO4.SA", "RANI3.SA", "FRIO3.SA", "PLPL3.SA", "PTNT3.SA",
    # Small cap consumer
    "CGRA3.SA", "CGRA4.SA", "WHRL3.SA", "LEVE3.SA", "MNPR3.SA",
    # Small cap financials
    "BAZA3.SA", "BRSR3.SA", "BRSR6.SA", "FESA3.SA", "FESA4.SA",
    # Small cap materials
    "FIBR3.SA", "TUPY3.SA", "FHER3.SA", "KEPL3.SA", "LUPA3.SA",
    # Small cap energy
    "CBAV3.SA", "EEEL3.SA", "ENMT3.SA", "NEOE3.SA", "OPCT3.SA",
    # Small cap healthcare
    "AALR3.SA", "PNVL3.SA", "ABEV3.SA", "PTBL3.SA", "REDE3.SA",
    # Small cap tech
    "LINX3.SA", "TRIS3.SA", "VULC3.SA", "FESA3.SA", "INTB3.SA",
    # Small cap retail
    "ANIM3.SA", "DPPI3.SA", "HOEP3.SA", "LREN3.SA", "MEAL3.SA",
    # Small cap infrastructure
    "PORT3.SA", "SEER3.SA", "TCNO3.SA", "TCNO4.SA", "TECN3.SA",
    # Additional liquid small caps
    "AFLT3.SA", "ALPA3.SA", "BALM3.SA", "BMOB3.SA", "BOBR3.SA",
    "BRKM3.SA", "BRKM5.SA", "BSEV3.SA", "CAML3.SA", "CESP3.SA",
    "CESP5.SA", "CESP6.SA", "COCE3.SA", "COCE5.SA", "COCE6.SA",
    "CPRE3.SA", "CRPG3.SA", "CRPG5.SA", "CTNM3.SA", "CTSA3.SA",
    "DTCY3.SA", "ELMD3.SA", "ENAT3.SA", "ETER3.SA", "FRTA3.SA",
    "GEPA3.SA", "GEPA4.SA", "GPAR3.SA", "GRND3.SA", "HAGA3.SA",
    "HAGA4.SA", "HAPV3.SA", "HBTS3.SA", "INEP3.SA", "INEP4.SA",
    "JALL3.SA", "JBSS3.SA", "KEPL3.SA", "LCAM3.SA", "LIGT3.SA",
    "LOGG3.SA", "LPAN3.SA", "LUPA3.SA", "MATD3.SA", "MDNE3.SA",
    "MELK3.SA", "MILS3.SA", "MMXM3.SA", "MYPK3.SA", "NEOE3.SA",
    "NGRD3.SA", "ODER3.SA", "OFER3.SA", "OPCT3.SA", "ORVR3.SA",
    "PARD3.SA", "PATS3.SA", "PDGR3.SA", "PGMN3.SA", "PKSA3.SA",
    "PLAS3.SA", "PMAM3.SA", "PNVL3.SA", "PNVL4.SA", "POWE3.SA",
    "PRIO3.SA", "PTCA3.SA", "PTCA4.SA", "PTNT3.SA", "QGEC3.SA",
    "QGEC4.SA", "RADL3.SA", "RCSL3.SA", "RCSL4.SA", "REDE3.SA",
    "RENT3.SA", "RNEW11.SA", "RNEW3.SA", "RNEW4.SA", "ROMI3.SA",
    "RPAD3.SA", "RPAD5.SA", "RSID3.SA", "SAPR11.SA", "SAPR3.SA",
    "SAPR4.SA", "SEDU3.SA", "SEIV3.SA", "SEIV4.SA", "SGPS3.SA",
    "SHOW3.SA", "SIMH3.SA", "SLCE3.SA", "SMFT3.SA", "SMTO3.SA",
    "SPRI3.SA", "STBP3.SA",
    "STBP11.SA", "SULA11.SA", "TASA3.SA", "TASA4.SA", "TCIN3.SA",
    "TECN3.SA", "TGMA3.SA", "TIMP3.SA", "TLEF3.SA", "TLEF4.SA",
    "TMCP3.SA", "TMCP4.SA", "TNLP3.SA", "TNLP4.SA", "TNEP3.SA",
    "TNEP4.SA", "TOYR3.SA", "TPIS3.SA", "TRAD3.SA", "TRIS3.SA",
    "TRUS3.SA", "UCAS3.SA", "UGPA3.SA", "UNIP3.SA",
    "UNIP5.SA", "UNIP6.SA", "USIM3.SA", "USIM5.SA", "USIM6.SA",
    "VIVA3.SA", "VIVR3.SA", "VULC3.SA", "WEGE3.SA", "WIZC3.SA",
    "WIZS3.SA", "YDUQ3.SA", "ZAMP3.SA",
]


def validate_ticker(ticker: str) -> bool:
    """Check if ticker is valid and has data on Yahoo Finance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        # Check if we got valid data
        return info is not None and 'symbol' in info
    except Exception:
        return False


def check_liquidity(ticker: str, min_volume: float = 5_000_000) -> bool:
    """Check if ticker has sufficient liquidity (R$5M daily volume)."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        
        if hist.empty:
            return False
        
        avg_volume = hist['Volume'].mean()
        avg_price = hist['Close'].mean()
        daily_turnover = avg_volume * avg_price
        
        return daily_turnover >= min_volume
    except Exception:
        return False


def expand_tickers():
    """Expand the ticker list with additional IBOV and SMLL stocks."""
    data_path = Path(__file__).parent.parent / 'data' / 'validated_tickers.json'
    
    # Load current tickers
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    current_tickers = set(data['all_tickers'])
    print(f"Current tickers: {len(current_tickers)}")
    
    # Combine all additional tickers
    all_additional = set(ADDITIONAL_IBOV + ADDITIONAL_SMLL)
    new_tickers = all_additional - current_tickers
    
    print(f"Additional tickers to validate: {len(new_tickers)}")
    
    # Validate new tickers
    valid_new = []
    for i, ticker in enumerate(sorted(new_tickers), 1):
        print(f"Validating {ticker} ({i}/{len(new_tickers)})...", end=" ")
        
        if validate_ticker(ticker):
            if check_liquidity(ticker):
                valid_new.append(ticker)
                print("✓ Valid + Liquid")
            else:
                print("✓ Valid, Low liquidity")
                # Still add, but flag
        else:
            print("✗ Invalid")
    
    print(f"\nValid new tickers: {len(valid_new)}")
    
    # Update data
    all_tickers = sorted(list(current_tickers | set(valid_new)))
    
    # Categorize new tickers
    ibov_tickers = set(data['indices']['IBOV_TOP50'])
    smll_tickers = set(data['indices']['SMLL_LIQUID'])
    
    for ticker in valid_new:
        if ticker in ADDITIONAL_IBOV:
            ibov_tickers.add(ticker)
        else:
            smll_tickers.add(ticker)
    
    # Save updated data
    updated_data = {
        'description': 'Brazilian stock tickers with liquidity validation',
        'generated_at': datetime.now().isoformat(),
        'total_tickers': len(all_tickers),
        'indices': {
            'IBOV': sorted(list(ibov_tickers)),
            'SMLL_LIQUID': sorted(list(smll_tickers)),
        },
        'all_tickers': all_tickers
    }
    
    # Backup old file
    backup_path = data_path.with_suffix('.json.backup')
    with open(backup_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Save new file
    with open(data_path, 'w') as f:
        json.dump(updated_data, f, indent=2)
    
    print(f"\nUpdated {data_path}")
    print(f"Total tickers: {len(all_tickers)}")
    print(f"IBOV: {len(ibov_tickers)}")
    print(f"SMLL: {len(smll_tickers)}")
    
    return all_tickers


if __name__ == '__main__':
    expand_tickers()