#!/usr/bin/env python3
"""
COMPREHENSIVE BACKTEST — 3 configs × 2 market periods
+ Top-10 ticker identification by model signal quality.

Tests:
  1. CONSERVADOR: threshold=0.60, max 5 pos, trailing stop 8%
  2. ATUAL:       threshold=0.38 (bull) / 0.50 (else), max 10 pos
  3. HÍBRIDO:     regime-aware (bull=0.38, sideways=0.50, bear=0.65), max 5 pos
  4. SMART PASSIVE: model picks top-10 tickers, then buy & hold
  5. PASSIVE: buy & hold all

Runs on:
  - BULL period (last 3 years)
  - BEAR period (COVID 2020)
"""

import sys
sys.path.insert(0, '.')

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from production_simple import SimpleProductionRunner
import json
import warnings
warnings.filterwarnings('ignore')

# ── Full 50-ticker universe (from top_50_tickers.json) ──
def load_all_tickers():
    try:
        with open('data/top_50_tickers.json') as f:
            data = json.load(f)
        tickers = data.get('top_50', [])
        if not tickers:
            raise ValueError("Empty")
        return [f"{t}.SA" if not t.endswith('.SA') else t for t in tickers[:50]]
    except:
        return [
            'PETR4.SA','VALE3.SA','ITUB4.SA','BBDC4.SA','PETR3.SA','BBAS3.SA',
            'ABEV3.SA','WEGE3.SA','B3SA3.SA','RENT3.SA','SUZB3.SA','PRIO3.SA',
            'BBDC3.SA','EQTL3.SA','RADL3.SA','HAPV3.SA','RDOR3.SA','RAIL3.SA',
            'GGBR4.SA','SBSP3.SA','CSNA3.SA','LREN3.SA','CMIG4.SA','UGPA3.SA',
            'ITSA4.SA','HYPE3.SA','VBBR3.SA','RAIZ4.SA','SLCE3.SA','CPFE3.SA',
            'TAEE11.SA','EGIE3.SA','TOTS3.SA','BBSE3.SA','MRVE3.SA','CSAN3.SA',
            'CYRE3.SA','KLBN11.SA','MULT3.SA','BRKM5.SA','CMIN3.SA','GOAU4.SA',
            'USIM5.SA','DXCO3.SA','ALPA4.SA','ASAI3.SA','BEEF3.SA','BPAC11.SA',
            'BRAP4.SA','CAML3.SA',
        ]

ALL_TICKERS = load_all_tickers()
# Use top 10 for full backtest speed, full list for ticker ranking
BACKTEST_TICKERS = ALL_TICKERS[:10]

INITIAL_CAPITAL_PER_STOCK = 1000
TOTAL_INITIAL_CAPITAL = INITIAL_CAPITAL_PER_STOCK * len(BACKTEST_TICKERS)

# ── Date ranges ──
PERIODS = {
    'BULL_3Y': (datetime.now() - timedelta(days=3*365), datetime.now()),
    'BEAR_COVID': (datetime(2020, 1, 2), datetime(2020, 12, 31)),
}


def calc_sentiment(data, lookback=10):
    if len(data) < lookback: return 0.0
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    recent = data.tail(lookback)
    returns = recent['Close'].pct_change().dropna()
    if len(returns) == 0: return 0.0
    return np.clip(returns.mean() * 50, -1, 1)


class Config:
    def __init__(self, name, buy_thresh_bull, buy_thresh_side, buy_thresh_bear,
                 max_positions, trailing_stop_init):
        self.name = name
        self.buy_bull = buy_thresh_bull
        self.buy_side = buy_thresh_side
        self.buy_bear = buy_thresh_bear
        self.max_pos = max_positions
        self.trail_stop = trailing_stop_init

    def get_threshold(self, regime):
        if regime == 'bull': return self.buy_bull
        if regime == 'bear': return self.buy_bear
        return self.buy_side

CONFIGS = [
    Config('CONSERVADOR', 0.60, 0.60, 0.65, 5, 0.92),
    Config('ATUAL',      0.38, 0.50, 0.65, 10, 0.88),
    Config('HIBRIDO',   0.38, 0.50, 0.65, 5, 0.90),
]


class PassiveStrategy:
    def __init__(self, tickers, capital_per):
        self.tickers = tickers
        self.capital_per = capital_per
    def run(self, price_data):
        all_dates = sorted(price_data[self.tickers[0]].index)
        history = []
        positions = {}
        for t in self.tickers:
            if t in price_data:
                p = price_data[t].loc[:all_dates[0]]
                if len(p)>0:
                    px = p['Close'].iloc[-1]; 
                    if hasattr(px,'item'): px=px.item()
                    positions[t] = {'shares': self.capital_per/px, 'avg': px}
        for d in all_dates:
            v = 0
            for t, pos in positions.items():
                if t in price_data:
                    p = price_data[t].loc[:d]
                    if len(p)>0:
                        px = p['Close'].iloc[-1]
                        if hasattr(px,'item'): px=px.item()
                        v += pos['shares']*px
            history.append({'date': d, 'value': v})
        return history


class ActiveConfigStrategy:
    def __init__(self, config, tickers, capital_per):
        self.cfg = config
        self.tickers = tickers
        self.capital_per = capital_per
    def run(self, price_data):
        total_cap = self.capital_per * len(self.tickers)
        pos = {}
        cash = total_cap
        trades = []
        hist = []
        peaks = {}
        runner = SimpleProductionRunner(use_news=False, n_workers=1)

        all_dates = sorted(price_data[self.tickers[0]].index)
        reb_dates = all_dates[::5]

        for i, rebalance_date in enumerate(reb_dates):
            # Phase 0 — trailing stops
            for t, p in list(pos.items()):
                if t not in price_data: continue
                data = price_data[t].loc[:rebalance_date]
                if len(data)<2: continue
                px = data['Close'].iloc[-1]
                if hasattr(px,'item'): px=px.item()
                if t not in peaks: peaks[t]=px
                if px > peaks[t]: peaks[t] = px

                stop = p['avg'] * self.cfg.trail_stop
                if peaks[t] > p['avg']*1.05:
                    stop = max(stop, peaks[t]*self.cfg.trail_stop)
                if px < stop:
                    sval = p['shares']*px
                    pnl = sval - p['shares']*p['avg']
                    cash += sval
                    del pos[t]; peaks.pop(t,None)
                    trades.append({'a':'SELL','t':t,'px':px,'pnl':pnl,'why':'TRAIL'})

            # Phase 1 — analyze & collect
            cands = []
            for t in self.tickers:
                if t not in price_data: continue
                data = price_data[t].loc[:rebalance_date].copy()
                if len(data)<50: continue

                res = runner.analyze_ticker(t, data=data, as_of_date=rebalance_date)
                if res is None: continue

                px = data['Close'].iloc[-1]
                if hasattr(px,'item'): px=px.item()

                sig = res['signal']
                fus = res.get('fused_score', 0.0)
                regime = res.get('market_regime', 'sideways')
                conf = res.get('conviction', 0.0)

                # Override threshold based on config + detected regime
                thresh = self.cfg.get_threshold(regime)

                # SELL signal
                if t in pos and sig in ['SELL','STRONG_SELL']:
                    p = pos[t]; sval = p['shares']*px; pnl = sval-p['shares']*p['avg']
                    cash+=sval; del pos[t]; peaks.pop(t,None)
                    trades.append({'a':'SELL','t':t,'px':px,'pnl':pnl,'why':'SIGNAL'})
                    continue

                # BUY candidate (only if confidence > config threshold)
                if sig in ['BUY','STRONG_BUY'] and t not in pos and conf >= thresh:
                    cands.append({'t':t,'fus':fus,'conf':conf,'px':px})

            # Phase 2 — enter new positions (ranked by fused_score)
            if cands and len(pos) < self.cfg.max_pos:
                cands.sort(key=lambda x: x['fus'], reverse=True)
                for c in cands[:self.cfg.max_pos - len(pos)]:
                    alloc = cash / max(self.cfg.max_pos - len(pos), 1) * 0.3
                    alloc = min(alloc, cash*0.3)
                    if alloc<=0: continue
                    shares = alloc/c['px']
                    pos[c['t']] = {'shares': shares, 'avg': c['px']}
                    peaks[c['t']] = c['px']
                    cash -= alloc
                    trades.append({'a':'BUY','t':c['t'],'px':c['px'],'pnl':0,'why':'FUS=%.2f' % c['fus']})

            # Mark-to-market
            v = cash
            for t, p in pos.items():
                if t in price_data:
                    d = price_data[t].loc[:rebalance_date]
                    if len(d)>0:
                        px = d['Close'].iloc[-1]
                        if hasattr(px,'item'): px=px.item()
                        v += p['shares']*px
            hist.append({'date': rebalance_date, 'value': v})

        return hist, trades


class SmartPassive:
    """
    Model picks top-N tickers by fused_score at periodic intervals,
    then buys and HOLDS (no active trading).
    Simulates: "10 melhores tickers que o modelo sugere, comprado e segurado"
    """
    def __init__(self, tickers, top_n=10, rebalance_months=3):
        self.tickers = tickers       # full universe
        self.top_n = top_n
        self.reb_months = rebalance_months

    def run(self, price_data, period_start):
        total_cap = TOTAL_INITIAL_CAPITAL
        all_dates = sorted(price_data[self.tickers[0]].index)
        hist = []
        held = {}  # ticker -> shares
        cash = total_cap
        last_rebalance = None
        runner = SimpleProductionRunner(use_news=False, n_workers=1)

        for i, date in enumerate(all_dates):
            # Check if it's time to re-evaluate holdings
            should_rebalance = (last_rebalance is None or
                               (date - last_rebalance).days >= self.reb_months * 30)

            if should_rebalance:
                # Analyze all tickers at this date
                scored = []
                for t in self.tickers:
                    if t not in price_data: continue
                    data = price_data[t].loc[:date].copy()
                    if len(data) < 50: continue
                    res = runner.analyze_ticker(t, data=data, as_of_date=date)
                    if res is None: continue
                    fus = res.get('fused_score', 0.0)
                    trend = res.get('trend', 'SIDEWAYS')
                    signal = res.get('signal', 'HOLD')
                    px = data['Close'].iloc[-1]
                    if hasattr(px, 'item'): px = px.item()
                    scored.append({'t': t, 'fus': fus, 'trend': trend, 'signal': signal, 'px': px})

                # Pick top N by fused_score
                top = sorted(scored, key=lambda x: x['fus'], reverse=True)[:self.top_n]

                # Liquidate everything, re-buy top N equally weighted
                # (sell current holdings at today's price)
                for t, s in list(held.items()):
                    if t in price_data:
                        d = price_data[t].loc[:date]
                        if len(d)>0:
                            px = d['Close'].iloc[-1]
                            if hasattr(px,'item'): px=px.item()
                            cash += s * px
                held = {}

                # Equal weight among top N
                alloc_per = cash / max(len(top), 1)
                for c in top:
                    shares = alloc_per / c['px']
                    held[c['t']] = shares
                    cash -= alloc_per

                last_rebalance = date

            # Mark-to-market
            v = cash
            for t, s in held.items():
                if t in price_data:
                    d = price_data[t].loc[:date]
                    if len(d)>0:
                        px = d['Close'].iloc[-1]
                        if hasattr(px,'item'): px=px.item()
                        v += s * px
            hist.append({'date': date, 'value': v})

        return hist


def calc_metrics(hist, initial_cap):
    if not hist: return {'total_return':0,'volatility':0,'sharpe_ratio':0,'max_drawdown':0}
    vals = [h['value'] for h in hist]
    rets = [(vals[i]-vals[i-1])/vals[i-1] for i in range(1,len(vals))]
    total_ret = (vals[-1]/initial_cap - 1)*100
    vol = np.std(rets)*np.sqrt(252)*100 if len(rets)>1 else 0
    sharpe = (total_ret - 5)/vol if vol>0 else 0
    peak = vals[0]; max_dd = max((peak-v)/peak for v in vals)
    return {'total_return':total_ret,'volatility':vol,'sharpe_ratio':sharpe,'max_drawdown':max_dd*100}


def download_data(tickers, start, end):
    print(f"  Downloading {len(tickers)} tickers: {start.date()} → {end.date()}...")
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start.strftime('%Y-%m-%d'),
                           end=end.strftime('%Y-%m-%d'), progress=False)
            if not df.empty and len(df)>=50:
                data[t] = df
        except: pass
    print(f"  ✅ Got {len(data)}/{len(tickers)}\n")
    return data


def run_period(period_name, start, end, tickers):
    print("\n" + "╔" + "═"*68 + "╗")
    print(f"║  📅 {period_name}: {start.date()} → {end.date()}".ljust(69) + "║")
    print("╚" + "═"*68 + "╝")

    pd_data = download_data(tickers, start, end)
    if len(pd_data) < 5:
        print("  ❌ Not enough data\n")
        return None

    results = {}

    # 1. Passive
    pas = PassiveStrategy(list(pd_data.keys()), INITIAL_CAPITAL_PER_STOCK)
    pas_hist = pas.run(pd_data)
    results['PASSIVE'] = calc_metrics(pas_hist, TOTAL_INITIAL_CAPITAL)

    # 2. Active configs
    for cfg in CONFIGS:
        act = ActiveConfigStrategy(cfg, list(pd_data.keys()), INITIAL_CAPITAL_PER_STOCK)
        act_hist, trades = act.run(pd_data)
        metrics = calc_metrics(act_hist, TOTAL_INITIAL_CAPITAL)
        buys = sum(1 for tr in trades if tr['a']=='BUY')
        sells = sum(1 for tr in trades if tr['a']=='SELL')
        wins = sum(1 for tr in trades if tr['a']=='SELL' and tr['pnl']>0)
        pnl = sum(tr['pnl'] for tr in trades if tr['a']=='SELL')
        metrics['trades'] = len(trades)
        metrics['win_rate'] = wins/sells*100 if sells else 0
        metrics['pnl'] = pnl
        results[cfg.name] = metrics

    # 3. Smart Passive (model-picked top-10, buy & hold)
    spl = SmartPassive(list(pd_data.keys()), top_n=10, rebalance_months=6)
    spl_hist = spl.run(pd_data, start)
    results['SMART_PASSIVE'] = calc_metrics(spl_hist, TOTAL_INITIAL_CAPITAL)

    return results


def print_table(period_results, period_name):
    if not period_results: return
    print(f"\n{'─'*80}")
    print(f"  🏆 {period_name}")
    print(f"{'─'*80}")
    header = f"  {'Estratégia':<20} {'Retorno':>10} {'Sharpe':>8} {'Drawdown':>10} {'Vol':>8}  {'Extras':>15}"
    print(header)
    print(f"  {'─'*78}")

    passive = period_results.get('PASSIVE', {})
    for name in ['PASSIVE','CONSERVADOR','ATUAL','HIBRIDO','SMART_PASSIVE']:
        m = period_results.get(name)
        if not m: continue
        extras = ''
        if name != 'PASSIVE' and name != 'SMART_PASSIVE':
            extras = f"Trades:{m.get('trades',0)} WR:{m.get('win_rate',0):.0f}%"
        alpha = ''
        if name != 'PASSIVE':
            a = m['total_return'] - passive.get('total_return',0)
            alpha = f" α:{a:+.1f}%"

        print(f"  {name:<20} {m['total_return']:>+9.2f}% {m['sharpe_ratio']:>7.2f} {m['max_drawdown']:>9.2f}% {m['volatility']:>7.2f}% {extras}{alpha}")

    print(f"  {'─'*78}")


def identify_best_tickers():
    """
    Run the model on ALL 50 tickers with TODAY'S data
    and rank by fused_score to identify the 10 best.
    """
    print("\n" + "╔" + "═"*68 + "╗")
    print(f"║  🔍 TOP-10 TICKER IDENTIFICATION (today's data)".ljust(69) + "║")
    print("╚" + "═"*68 + "╝\n")

    runner = SimpleProductionRunner(use_news=False, n_workers=1)
    
    # Download 6 months of data for each ticker
    end = datetime.now()
    start = end - timedelta(days=365)
    
    scored = []
    for t in ALL_TICKERS:
        try:
            df = yf.download(t, start=start.strftime('%Y-%m-%d'),
                           end=end.strftime('%Y-%m-%d'), progress=False)
            if df.empty or len(df) < 50: continue
            
            res = runner.analyze_ticker(t, data=df)
            if res is None: continue
            
            px = df['Close'].iloc[-1]
            if hasattr(px, 'item'): px = px.item()
            
            scored.append({
                'ticker': t.replace('.SA',''),
                'price': px,
                'signal': res.get('signal','?'),
                'fused_score': res.get('fused_score', 0.0),
                'trend': res.get('trend','?'),
                'confidence': res.get('confidence',0.0),
                'conviction': res.get('conviction',0.0),
                'position_size': res.get('position_size',0.0),
                'fundamental_grade': res.get('fundamentals',{}).get('grade','?') if res.get('fundamentals') else '?',
            })
            print(f"  {t:<12} {res.get('signal','?'):<12} fused={res.get('fused_score',0):+.2f}  trend={res.get('trend','?')}")
        except Exception as e:
            print(f"  {t:<12} ❌ {e}")

    # Sort by fused_score
    scored.sort(key=lambda x: x['fused_score'], reverse=True)

    print(f"\n  {'─'*70}")
    print(f"  🏅 TOP 10 BY FUSED SCORE")
    print(f"  {'─'*70}")
    for i, s in enumerate(scored[:10]):
        emoji = '🚀' if s['signal']=='STRONG_BUY' else ('🟢' if s['signal']=='BUY' else '🟡')
        print(f"  {i+1:>2}. {emoji} {s['ticker']:<8} R${s['price']:<8.2f} fused={s['fused_score']:+.2f}  {s['trend']}  {s['signal']}")

    # Top-10 buy & hold simulation (if we'd bought these 6 months ago)
    print(f"\n  {'─'*70}")
    print(f"  📊 TOP-10 BUY & HOLD (6-month lookback)")
    print(f"  {'─'*70}")
    total_ret = 0
    for i, s in enumerate(scored[:10]):
        try:
            df = yf.download(s['ticker']+'.SA', start=start.strftime('%Y-%m-%d'),
                           end=end.strftime('%Y-%m-%d'), progress=False)
            if df.empty: continue
            entry = df['Close'].iloc[0]
            exit_px = df['Close'].iloc[-1]
            if hasattr(entry,'item'): entry=entry.item()
            if hasattr(exit_px,'item'): exit_px=exit_px.item()
            ret = (exit_px/entry - 1)*100
            total_ret += ret
            print(f"  {i+1:>2}. {s['ticker']:<8} R${entry:.2f} → R${exit_px:.2f}  {ret:+.1f}%")
        except: pass
    avg_ret = total_ret/10
    print(f"\n  🎯 Top-10 average 6-month return: {avg_ret:+.1f}%")

    return scored[:10]


def main():
    print("\n" + "═"*70)
    print("🚀 COMPREHENSIVE BACKTEST — KIPP Signal Validation")
    print("═"*70)
    print(f"Backtest tickers: {len(BACKTEST_TICKERS)} | Full universe: {len(ALL_TICKERS)}")
    print(f"Configs: Conservative, Current, Hybrid + Smart Passive\n")

    all_results = {}

    # Run both periods
    for pname, (start, end) in PERIODS.items():
        results = run_period(pname, start, end, BACKTEST_TICKERS)
        if results:
            all_results[pname] = results
            print_table(results, pname)

    # ── FINAL COMPARISON ──
    print(f"\n{'═'*80}")
    print(f"  🎯 FINAL VERDICT — Strategy Ranking")
    print(f"{'═'*80}")

    for pname in ['BULL_3Y','BEAR_COVID']:
        r = all_results.get(pname)
        if not r: continue
        passive_ret = r.get('PASSIVE',{}).get('total_return',0)
        print(f"\n  📅 {pname}:")
        
        strategies = []
        for name in ['PASSIVE','CONSERVADOR','ATUAL','HIBRIDO','SMART_PASSIVE']:
            m = r.get(name)
            if not m: continue
            strategies.append((name, m))
        
        # Rank by Sharpe
        by_sharpe = sorted(strategies, key=lambda x: x[1]['sharpe_ratio'], reverse=True)
        sharpe_parts = []
        for n, m in by_sharpe:
            sharpe_parts.append(f'{n}({m["sharpe_ratio"]:.2f})')
        print(f'    By Sharpe: {" > ".join(sharpe_parts)}')
        
        # Rank by return
        by_ret = sorted(strategies, key=lambda x: x[1]['total_return'], reverse=True)
        ret_parts = []
        for n, m in by_ret:
            ret_parts.append(f'{n}({m["total_return"]:+.1f}%)')
        print(f'    By Return: {" > ".join(ret_parts)}')
        
        # Rank by drawdown protection
        by_dd = sorted(strategies, key=lambda x: x[1]['max_drawdown'])
        dd_parts = []
        for n, m in by_dd:
            dd_parts.append(f'{n}({m["max_drawdown"]:.1f}%)')
        print(f'    By DD Prot: {" > ".join(dd_parts)}')

    # ── Identify best tickers ──
    top10 = identify_best_tickers()

    # Save results
    output = {
        'bull_3y': {},
        'bear_covid': {},
        'best_tickers': [{'ticker':t['ticker'],'fused_score':t['fused_score'],
                          'signal':t['signal']} for t in top10]
    }
    for pname, results in all_results.items():
        for sname, m in results.items():
            output[pname.lower()] = {k:v for k,v in m.items() if k not in ['trades','win_rate','pnl']}
    
    with open('comprehensive_backtest_results.json','w') as f:
        json.dump(output, f, indent=2)

    # Save summary text
    with open('comprehensive_backtest_summary.txt','w') as f:
        f.write("COMPREHENSIVE BACKTEST SUMMARY\n")
        f.write("="*70 + "\n\n")
        for pname, results in all_results.items():
            f.write(f"\n{pname}:\n")
            f.write(f"{'Strategy':<20} {'Return':>10} {'Sharpe':>8} {'DD':>10} {'Vol':>8}\n")
            f.write(f"{'-'*60}\n")
            for name in ['PASSIVE','CONSERVADOR','ATUAL','HIBRIDO','SMART_PASSIVE']:
                m = results.get(name)
                if m:
                    f.write(f"{name:<20} {m['total_return']:>+9.2f}% {m['sharpe_ratio']:>7.2f} {m['max_drawdown']:>9.2f}% {m['volatility']:>7.2f}%\n")
        f.write(f"\n\nBEST TICKERS (by fused_score):\n")
        for i, t in enumerate(top10[:10]):
            f.write(f"  {i+1}. {t['ticker']} — {t['signal']} — fused={t['fused_score']:+.2f}\n")
    
    print(f"\n{'═'*70}")
    print("✅ Results: comprehensive_backtest_results.json & .txt")
    print("═"*70)


if __name__ == '__main__':
    main()
