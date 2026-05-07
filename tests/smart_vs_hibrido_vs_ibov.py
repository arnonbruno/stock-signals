#!/usr/bin/env python3
"""
SMART PASSIVE vs HÍBRIDO ATIVO vs IBOV PASSIVO
Períodos: 1, 2, 3 anos
SMART_PASSIVE: modelo escolhe top-10 por fused_score, buy & hold 6 meses
HIBRIDO: regime-aware ativo com entradas/saídas (já temos os dados)
"""
import sys
sys.path.insert(0, '.')
import yfinance as yf, pandas as pd, numpy as np
from datetime import datetime, timedelta
from production_simple import SimpleProductionRunner
import json, io, os, contextlib, warnings
warnings.filterwarnings('ignore')

ALL_TICKERS = [
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
CAPITAL = 100_000; TOP_N = 10

def download(tickers, start, end):
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
            if not df.empty and len(df)>=50: data[t] = df
        except: pass
    return data

def ibov(start, end):
    try:
        d = yf.download('^BVSP', start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False)
        return d if not d.empty and len(d)>1 else None
    except: return None

def smart_passive(price_data, capital, top_n=10, reb_months=6):
    """Modelo escolhe top-N, buy & hold, reavalia a cada reb_months."""
    runner = SimpleProductionRunner(use_news=False, n_workers=1)
    tickers = list(price_data.keys())
    dates = sorted(price_data[tickers[0]].index)
    null = io.StringIO()
    
    held, cash, hist = {}, capital, []
    last_reb, next_reb = None, None
    
    for i, date in enumerate(dates):
        do_reb = (last_reb is None or (date - last_reb).days >= reb_months*30)
        
        if do_reb:
            # Analyze all tickers at this date
            scored = []
            with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
                for t in tickers:
                    if t not in price_data: continue
                    hd = price_data[t].loc[:date].copy()
                    if len(hd)<50: continue
                    try:
                        res = runner.analyze_ticker(t, data=hd, as_of_date=date)
                        if res is None: continue
                        fus = res.get('fused_score',0.0)
                        px = hd['Close'].iloc[-1]
                        if hasattr(px,'item'): px=px.item()
                        scored.append({'t':t,'fus':fus,'px':px})
                    except: pass
            
            # Pick top N
            top = sorted(scored, key=lambda x: x['fus'], reverse=True)[:top_n]
            
            # Liquidate & re-buy equally weighted
            for t, s in list(held.items()):
                if t in price_data:
                    d2 = price_data[t].loc[:date]
                    if len(d2)>0:
                        px = d2['Close'].iloc[-1]
                        if hasattr(px,'item'): px=px.item()
                        cash += s*px
            held = {}
            
            if top:
                alloc = cash / len(top)
                for c in top:
                    held[c['t']] = alloc / c['px']
                    cash -= alloc
            
            last_reb = date
        
        # MTM
        val = cash + sum(s*(price_data[t].loc[:date]['Close'].iloc[-1].item() if hasattr(price_data[t].loc[:date]['Close'].iloc[-1],'item') else price_data[t].loc[:date]['Close'].iloc[-1]) for t,s in held.items() if t in price_data and len(price_data[t].loc[:date])>0)
        hist.append({'data':date,'valor':round(val,2)})
    
    return hist

def metrics(hist, capital):
    if not hist or len(hist)<2: return {}
    vals = [h['valor'] for h in hist]
    rets = [(vals[i]-vals[i-1])/vals[i-1] for i in range(1,len(vals))]
    ret = (vals[-1]/capital-1)*100
    vol = np.std(rets)*np.sqrt(252)*100 if len(rets)>1 else 0
    sharpe = (ret/100-0.05)/(vol/100) if vol>0 else 0
    peak=vals[0]; dd=max((peak-v)/peak*100 for v in vals)
    return {'ret':round(ret,2),'vol':round(vol,2),'sharpe':round(sharpe,2),'dd':round(dd,2)}

# Reuse hibrido results from previous run
prev = None
try:
    with open('tests/rolling_portfolio_results.json') as f: prev = json.load(f)
except: pass

hibrido = {}
if prev:
    for k in ['1_ANO','2_ANOS','3_ANOS']:
        if k in prev:
            hibrido[k] = prev[k]['metrics']

print(f"\n{'='*60}")
print("🚀 SMART PASSIVE vs HÍBRIDO vs IBOV — 1/2/3 anos")
print(f"{'='*60}")
print(f"SMART_PASSIVE: modelo escolhe top-{TOP_N}, buy & hold 6 meses")
print(f"HIBRIDO: regime-aware ativo (mensal)")
print(f"Capital: R$ {CAPITAL:,.0f}\n")

hoje = datetime.now()
resultados = {}

for nome, delta in [('1_ANO',365),('2_ANOS',730),('3_ANOS',1095)]:
    start = hoje - timedelta(days=delta)
    print(f"\n{'─'*60}")
    print(f"📅 {nome}: {start.date()} → {hoje.date()}")
    print(f"{'─'*60}")
    
    pd_data = download(ALL_TICKERS, start, hoje)
    ib = ibov(start, hoje)
    
    # SMART PASSIVE
    print("  🧠 SMART_PASSIVE...")
    sp_hist = smart_passive(pd_data, CAPITAL, TOP_N, 6)
    sp_m = metrics(sp_hist, CAPITAL)
    
    # IBOV passive
    ib_m = {}
    if ib is not None and len(ib)>1:
        ib_close = ib['Close']
        if isinstance(ib.columns, pd.MultiIndex): ib_close = ib_close[ib_close.columns[0]]
        ib_ret = (ib_close.iloc[-1]/ib_close.iloc[0]-1)*100
        if hasattr(ib_ret,'item'): ib_ret=ib_ret.item()
        ib_m = {'ret':round(ib_ret,2)}
    
    # Hibrido from previous
    h = hibrido.get(nome, {})
    
    print(f"\n  {'Estratégia':<20} {'Retorno':>10} {'Sharpe':>8} {'DD':>10} {'ALPHA':>10}")
    print(f"  {'─'*58}")
    
    sp_alpha = sp_m['ret'] - ib_m.get('ret',0)
    print(f"  {'SMART_PASSIVE':<20} {sp_m['ret']:>+9.2f}% {sp_m['sharpe']:>7.2f} {sp_m['dd']:>9.2f}% {sp_alpha:>+9.2f}%")
    
    if ib_m:
        print(f"  {'IBOV PASSIVO':<20} {ib_m['ret']:>+9.2f}% {'':>7} {'':>9} {'':>9}")
    
    h_ret = h.get('retorno', 'N/A')
    h_sharpe = h.get('sharpe', 'N/A')
    h_dd = h.get('dd', 'N/A')
    h_alpha = h.get('alpha', 'N/A')
    if isinstance(h_ret, (int,float)): h_ret = f"{h_ret:+.2f}%"
    if isinstance(h_sharpe, (int,float)): h_sharpe = f"{h_sharpe:.2f}"
    if isinstance(h_dd, (int,float)): h_dd = f"{h_dd:.2f}%"
    if isinstance(h_alpha, (int,float)): h_alpha = f"{h_alpha:+.2f}%"
    print(f"  {'HIBRIDO ATIVO':<20} {str(h_ret):>10} {str(h_sharpe):>8} {str(h_dd):>10} {str(h_alpha):>10}")
    
    resultados[nome] = {'sp':sp_m,'ib':ib_m,'h':h,'sp_alpha':sp_alpha}

# Final summary
print(f"\n{'='*60}")
print("🏁 RESULTADO FINAL — Quem é o melhor?")
print(f"{'='*60}")

print(f"\n  {'Estratégia':<20} {'1 Ano':>12} {'2 Anos':>12} {'3 Anos':>12}")
print(f"  {'─'*56}")
for lbl, key, fmt in [('SMART PASSIVE','sp','ret'),('HIBRIDO ATIVO','h','retorno'),('IBOV PASSIVO','ib','ret')]:
    vals = []
    for p in ['1_ANO','2_ANOS','3_ANOS']:
        r = resultados.get(p,{})
        d = r.get(key,{})
        v = d.get(fmt,'N/A')
        if isinstance(v,(int,float)): v = f"{v:+.2f}%"
        vals.append(f"{str(v):>12}")
    print(f"  {lbl:<20} {'| '.join(vals)}")

# Count wins per strategy
print(f"\n  {'🏆':<20}", end="")
for p in ['1_ANO','2_ANOS','3_ANOS']:
    r = resultados.get(p,{})
    sp_ret = r.get('sp',{}).get('ret',-999)
    ib_ret = r.get('ib',{}).get('ret',-999)
    h_ret = r.get('h',{}).get('retorno',-999)
    if isinstance(sp_ret,(int,float)) and isinstance(ib_ret,(int,float)) and isinstance(h_ret,(int,float)):
        best = max(sp_ret, ib_ret, h_ret)
        if best == sp_ret: name = 'SMART 🥇'
        elif best == h_ret: name = 'HIBRIDO 🥇'
        else: name = 'IBOV 🥇'
        print(f"{name:>12}", end="")
    print(" | ", end="")
print()

# Save
with open('tests/smart_vs_hibrido_vs_ibov.json','w') as f:
    json.dump(resultados, f, indent=2, default=str)
print(f"\n✅ Salvo: tests/smart_vs_hibrido_vs_ibov.json")
