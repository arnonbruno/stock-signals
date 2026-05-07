#!/usr/bin/env python3
"""
SMART_PASSIVE Backtest — Full 50-ticker universe
Model picks top-10 stocks by fused_score, buy & hold 6 months.
"""

import sys, json, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from production_simple import SimpleProductionRunner

# ── Full 50-ticker universe ──
def load_tickers():
    try:
        with open('data/top_50_tickers.json') as f:
            data = json.load(f)
        tickers = data.get('top_50', [])
        if not tickers: raise ValueError
        return [f"{t}.SA" if not t.endswith('.SA') else t for t in tickers[:50]]
    except:
        return ['PETR4.SA','VALE3.SA','ITUB4.SA','BBDC4.SA','PETR3.SA','BBAS3.SA',
            'ABEV3.SA','WEGE3.SA','B3SA3.SA','RENT3.SA','SUZB3.SA','PRIO3.SA',
            'BBDC3.SA','EQTL3.SA','RADL3.SA','HAPV3.SA','RDOR3.SA','RAIL3.SA',
            'GGBR4.SA','SBSP3.SA','CSNA3.SA','LREN3.SA','CMIG4.SA','UGPA3.SA',
            'ITSA4.SA','HYPE3.SA','VBBR3.SA','RAIZ4.SA','SLCE3.SA','CPFE3.SA',
            'TAEE11.SA','EGIE3.SA','TOTS3.SA','BBSE3.SA','MRVE3.SA','CSAN3.SA',
            'CYRE3.SA','KLBN11.SA','MULT3.SA','BRKM5.SA','CMIN3.SA','GOAU4.SA',
            'USIM5.SA','DXCO3.SA','ALPA4.SA','ASAI3.SA','BEEF3.SA','BPAC11.SA',
            'BRAP4.SA','CAML3.SA']

ALL_TICKERS = load_tickers()
INITIAL_CAPITAL = 10000

# 3-year bull period
END = datetime.now()
START = END - timedelta(days=3*365)


def download(tickers, start, end):
    print(f"📥 Baixando {len(tickers)} ações de {start.date()} a {end.date()}...")
    data = {}
    ok = 0
    for t in tickers:
        try:
            df = yf.download(t, start=start.strftime('%Y-%m-%d'),
                           end=end.strftime('%Y-%m-%d'), progress=False)
            if not df.empty and len(df) >= 50:
                data[t] = df
                ok += 1
        except: pass
        if ok % 10 == 0 and ok > 0:
            print(f"  ...{ok}/{len(tickers)} downloads concluidos")
    print(f"✅ {len(data)}/{len(tickers)} downloads OK\n")
    return data


def smart_passive_bt(tickers, price_data, top_n=10, reb_months=6):
    """Model picks top-N, buys equally weighted, holds, rebalances periodically."""
    print(f"🧠 SMART_PASSIVE: top-{top_n} de {len(tickers)} tickers, hold {reb_months}m")
    print(f"   Capital inicial: R$ {INITIAL_CAPITAL:,.2f}")
    
    runner = SimpleProductionRunner(use_news=False, n_workers=1)
    
    all_dates = sorted(price_data[tickers[0]].index)
    history = []
    held = {}
    cash = INITIAL_CAPITAL
    last_rebalance = None
    rebalance_count = 0
    
    for i, date in enumerate(all_dates):
        should_rebalance = (last_rebalance is None or
                           (date - last_rebalance).days >= reb_months * 30)
        
        if should_rebalance:
            rebalance_count += 1
            print(f"\n  🔄 Rebalanceamento #{rebalance_count} em {date.date()}...")
            
            scored = []
            for t in tickers:
                if t not in price_data: continue
                df = price_data[t].loc[:date].copy()
                if len(df) < 50: continue
                
                res = runner.analyze_ticker(t, data=df, as_of_date=date)
                if res is None: continue
                
                fus = res.get('fused_score', 0.0)
                px = df['Close'].iloc[-1]
                if hasattr(px, 'item'): px = px.item()
                scored.append({'t': t, 'fus': fus, 'px': px,
                              'sig': res.get('signal','HOLD'),
                              'trend': res.get('trend','?')})
            
            # Top N by fused_score (exclude SELL signals)
            scored = sorted([s for s in scored if s['sig'] not in ['SELL','STRONG_SELL']],
                          key=lambda x: x['fus'], reverse=True)[:top_n]
            
            # Liquidate
            for t, s in list(held.items()):
                if t in price_data:
                    d = price_data[t].loc[:date]
                    if len(d) > 0:
                        px = d['Close'].iloc[-1]
                        if hasattr(px, 'item'): px = px.item()
                        cash += s * px
            held = {}
            
            # Buy top N equally
            alloc = cash / max(len(scored), 1)
            for c in scored:
                shares = alloc / c['px']
                held[c['t']] = shares
                cash -= alloc
            
            print(f"     Compras: {len(scored)} ativos, R$ {alloc:.2f} cada")
            for s in scored[:5]:
                print(f"       {s['t'].replace('.SA',''):8} fus={s['fus']:+.2f}  {s['sig']}  {s['trend']}")
            if len(scored) > 5:
                print(f"       ...mais {len(scored)-5}")
            
            last_rebalance = date
        
        # Mark to market
        v = cash
        for t, s in held.items():
            if t in price_data:
                d = price_data[t].loc[:date]
                if len(d) > 0:
                    px = d['Close'].iloc[-1]
                    if hasattr(px, 'item'): px = px.item()
                    v += s * px
        history.append({'date': date, 'value': v})
    
    return history


def passive_bt(tickers, price_data):
    """Buy & hold equally weighted all available tickers."""
    all_dates = sorted(price_data[tickers[0]].index)
    first = all_dates[0]
    positions = {}
    
    for t in tickers:
        if t in price_data:
            df = price_data[t].loc[:first]
            if len(df) > 0:
                px = df['Close'].iloc[-1]
                if hasattr(px, 'item'): px = px.item()
                positions[t] = {'shares': INITIAL_CAPITAL / len(tickers) / px}
    
    history = []
    for date in all_dates:
        v = 0
        for t, p in positions.items():
            if t in price_data:
                df = price_data[t].loc[:date]
                if len(df) > 0:
                    px = df['Close'].iloc[-1]
                    if hasattr(px, 'item'): px = px.item()
                    v += p['shares'] * px
        history.append({'date': date, 'value': v})
    return history


def metrics(history, initial):
    if not history: return {}
    vals = [h['value'] for h in history]
    rets = [(vals[i]-vals[i-1])/vals[i-1] for i in range(1, len(vals))]
    total = (vals[-1]/initial - 1) * 100
    vol = np.std(rets) * np.sqrt(252) * 100 if len(rets) > 1 else 0
    sharpe = (total - 5) / vol if vol > 0 else 0
    peak = vals[0]
    max_dd = max((peak - v)/peak for v in vals) * 100
    return {'total_return': total, 'volatility': vol, 'sharpe_ratio': sharpe, 'max_drawdown': max_dd}


def main():
    print("\n" + "="*70)
    print("🧠 SMART PASSIVE BACKTEST — Full 50-Ticker Universe")
    print("="*70)
    print(f"Período: {START.date()} → {END.date()} (3 anos)")
    print(f"Universo: {len(ALL_TICKERS)} tickers | Top-10 por fused_score | Hold 6 meses\n")
    
    # Download
    price_data = download(ALL_TICKERS, START, END)
    if len(price_data) < 10:
        print("❌ Dados insuficientes")
        return
    
    # Run PASSIVE (buy & hold all available)
    print("📈 Rodando PASSIVO (buy & hold todos)...")
    pas_hist = passive_bt(list(price_data.keys()), price_data)
    pas_m = metrics(pas_hist, INITIAL_CAPITAL)
    
    # Run SMART PASSIVE
    sp_hist = smart_passive_bt(list(price_data.keys()), price_data, top_n=10, reb_months=6)
    sp_m = metrics(sp_hist, INITIAL_CAPITAL)
    
    # Results
    print(f"\n{'='*70}")
    print(f"📊 RESULTADOS — BULL 3 ANOS (50 tickers)")
    print(f"{'='*70}")
    print(f"\n{'Métrica':<25} {'📈 PASSIVO':>15} {'🧠 SMART_PASSIVE':>20}")
    print(f"{'─'*65}")
    
    for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']:
        pv = pas_m.get(metric, 0)
        sv = sp_m.get(metric, 0)
        unit = '%' if metric in ['total_return', 'max_drawdown', 'volatility'] else ''
        winner = '🧠' if (metric in ['sharpe_ratio','total_return'] and sv > pv) or \
                        (metric in ['max_drawdown','volatility'] and sv < pv) else '📈'
        labels = {
            'total_return': 'Retorno Total',
            'sharpe_ratio': 'Sharpe Ratio',
            'max_drawdown': 'Max Drawdown',
            'volatility': 'Volatilidade',
        }
        print(f"  {labels[metric]:<25} {pv:>+13.2f}{unit} {sv:>+17.2f}{unit}  {winner}")
    
    alpha = sp_m.get('total_return',0) - pas_m.get('total_return',0)
    print(f"\n  🎯 Alpha vs Passivo: {alpha:+.2f}%")
    if alpha > 0:
        print(f"  ✅ SMART PASSIVE BATEU O PASSIVO em {len(price_data)} ações! 🏆")
    else:
        print(f"  ❌ SMART PASSIVE ficou atrás do passivo")
    
    print(f"\n{'='*70}")
    print(f"✅ Resultados salvos em smart_passive_backtest_results.txt")
    print(f"{'='*70}")
    
    # Save
    with open('smart_passive_backtest_results.txt', 'w') as f:
        f.write("SMART PASSIVE BACKTEST — Full 50-ticker universe\n")
        f.write(f"Período: {START.date()} → {END.date()}\n")
        f.write(f"Estratégia: Model picks top-10 by fused_score, equal weight, hold 6mo\n\n")
        f.write(f"PASSIVE:\n")
        for k,v in pas_m.items(): f.write(f"  {k}: {v:.2f}\n")
        f.write(f"\nSMART_PASSIVE:\n")
        for k,v in sp_m.items(): f.write(f"  {k}: {v:.2f}\n")
        f.write(f"\nAlpha: {alpha:.2f}%\n")


if __name__ == '__main__':
    main()
