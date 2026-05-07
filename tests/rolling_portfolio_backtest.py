#!/usr/bin/env python3
"""
ROLLING PORTFOLIO BACKTEST — Modelo como Gestor de Carteira
Simula: o modelo analisa 50 tickers a cada mês, escolhe top-10, monta carteira.
Compara vs IBOVESPA buy & hold.
Períodos: 1 ano, 2 anos, 3 anos.
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

CAPITAL_INICIAL = 100_000
TOP_N = 10
REB_DAYS = 21  # mensal

def download_data(tickers, start, end):
    print(f"  Download {len(tickers)} tickers: {start.date()} → {end.date()}...")
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start.strftime('%Y-%m-%d'),
                           end=end.strftime('%Y-%m-%d'), progress=False)
            if not df.empty and len(df) >= 50:
                data[t] = df
        except: pass
    print(f"  ✅ {len(data)}/{len(tickers)}")
    return data

def baixar_ibov(start, end):
    try:
        ibov = yf.download('^BVSP', start=start.strftime('%Y-%m-%d'),
                          end=end.strftime('%Y-%m-%d'), progress=False)
        return ibov if not ibov.empty and len(ibov)>0 else None
    except: return None

def calc_metrics(hist, capital):
    if not hist or len(hist)<2: return {}
    vals = [h['valor'] for h in hist]
    rets = [(vals[i]-vals[i-1])/vals[i-1] for i in range(1,len(vals))]
    ret_total = (vals[-1]/capital-1)*100
    vol = np.std(rets)*np.sqrt(252)*100 if len(rets)>1 else 0
    sharpe = (ret_total/100-0.05)/(vol/100) if vol>0 else 0
    peak=vals[0]; dd_max=max((peak-v)/peak*100 for v in vals)
    return {'retorno':round(ret_total,2),'vol':round(vol,2),'sharpe':round(sharpe,2),'dd':round(dd_max,2)}

def simular(price_data, ibov_data, capital, top_n=10):
    runner = SimpleProductionRunner(use_news=False, n_workers=1)
    tickers_validos = list(price_data.keys())
    all_dates = sorted(price_data[tickers_validos[0]].index)
    reb_dates = all_dates[::REB_DAYS]
    
    posicoes, caixa, historico, trades = {}, capital, [], []
    ibov_start, ibov_curr = None, None
    
    print(f"  📊 {len(reb_dates)} rebalances...")
    null = io.StringIO()
    
    for i, data_atual in enumerate(reb_dates):
        if i%5==0: print(f"     {i}/{len(reb_dates)}...")
        
        # IBOV update
        if ibov_data is not None:
            ibov_hist = ibov_data.loc[:data_atual]
            if len(ibov_hist)>0:
                ibov_curr = ibov_hist['Close'].iloc[-1]; 
                if hasattr(ibov_curr,'item'): ibov_curr=ibov_curr.item()
                if ibov_start is None: ibov_start=ibov_curr
        
        # Analyze all tickers (suppress stdout)
        candidatos = []
        with contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
            for t in tickers_validos:
                if t not in price_data: continue
                hist = price_data[t].loc[:data_atual].copy()
                if len(hist)<50: continue
                try:
                    res = runner.analyze_ticker(t, data=hist, as_of_date=data_atual)
                    if res is None: continue
                    sig = res.get('signal','HOLD')
                    fus = res.get('fused_score',0.0)
                    conf = res.get('conviction',0.0) or res.get('confidence',0.0)
                    regime = res.get('market_regime','sideways')
                    px = hist['Close'].iloc[-1]
                    if hasattr(px,'item'): px=px.item()
                    thresh = {'bull':0.38,'sideways':0.50,'bear':0.65}.get(regime,0.50)
                    candidatos.append({'t':t,'sig':sig,'fus':fus,'conf':conf,'px':px,'thr':thresh})
                except: continue
        
        # Update current prices for holdings
        for t in posicoes:
            if t in price_data:
                h2 = price_data[t].loc[:data_atual]
                if len(h2)>0:
                    p = h2['Close'].iloc[-1]
                    if hasattr(p,'item'): p=p.item()
                    posicoes[t]['px'] = p
        
        # Ranking top N by fused_score
        candidatos.sort(key=lambda x: x['fus'], reverse=True)
        top = {c['t'] for c in candidatos[:top_n]}
        
        # Sell phase
        for t in list(posicoes.keys()):
            p = posicoes[t]; motivo = None
            info = next((c for c in candidatos if c['t']==t), None)
            if info is None: motivo = 'NO_DATA'
            elif info['sig'] in ['SELL','STRONG_SELL']: motivo = 'SELL_SIG'
            elif t not in top: motivo = 'OUT_TOP10'
            elif info['conf'] < info['thr']: motivo = 'LOW_CONF'
            if motivo:
                px_out = p.get('px', p['avg'])
                caixa += p['shares']*px_out
                trades.append({'a':'SELL','t':t,'d':str(data_atual.date()),'px':round(px_out,2),
                    'pnl':round(p['shares']*(px_out-p['avg']),2),'w':motivo,
                    'hold':(data_atual-p['entrada']).days})
                del posicoes[t]
        
        # Buy phase
        vagas = top_n - len(posicoes)
        if vagas>0:
            compras = [c for c in candidatos if c['t'] not in posicoes 
                      and c['sig'] in ['BUY','STRONG_BUY'] and c['conf']>=c['thr']]
            for c in compras[:vagas]:
                alloc = min(caixa*0.25, caixa/max(vagas,1))
                if alloc<=0: break
                s = alloc/c['px']; caixa-=alloc; vagas-=1
                posicoes[c['t']] = {'shares':s,'avg':c['px'],'entrada':data_atual,'px':c['px']}
                trades.append({'a':'BUY','t':c['t'],'d':str(data_atual.date()),
                    'px':round(c['px'],2),'pnl':0,'w':f"F={c['fus']:.2f}",'hold':0})
        
        # MTM
        val = caixa + sum(pos['shares']*pos.get('px',pos['avg']) for pos in posicoes.values())
        historico.append({'data':data_atual,'valor':round(val,2),'caixa':round(caixa,2),
            'pos':len(posicoes),'ibov':round(ibov_curr,2) if ibov_curr else 1,
            'ibov_ini':round(ibov_start,2) if ibov_start else 1,'ibov_fim':round(ibov_curr,2) if ibov_curr else 1})
    
    return historico, trades

def run_period(nome, start, end):
    print(f"\n{'='*60}\n📅 {nome}: {start.date()} → {end.date()}\n{'='*60}")
    pd_data = download_data(ALL_TICKERS, start, end)
    if len(pd_data)<10: print(f"  ❌ {len(pd_data)} tickers apenas"); return None
    ibov = baixar_ibov(start, end)
    
    hist, trades = simular(pd_data, ibov, CAPITAL_INICIAL, TOP_N)
    if not hist: print("  ❌ Simulation failed"); return None
    
    m = calc_metrics(hist, CAPITAL_INICIAL)
    buys = sum(1 for t in trades if t['a']=='BUY')
    sells = [t for t in trades if t['a']=='SELL']
    wins = sum(1 for s in sells if s['pnl']>0)
    m['trades']=len(trades); m['buys']=buys; m['sells']=len(sells)
    m['wins']=wins; m['loss']=len(sells)-wins
    m['wr']=round(wins/len(sells)*100,1) if sells else 0
    m['pnl']=round(sum(s['pnl'] for s in sells),2); m['dias']=(end-start).days
    
    # IBOV passive
    if ibov is not None and len(ibov)>1:
        ibov_close = ibov['Close']
        if isinstance(ibov.columns, pd.MultiIndex):
            ibov_close = ibov_close[ibov_close.columns[0]]
        ibov_ret = (ibov_close.iloc[-1]/ibov_close.iloc[0]-1)*100
        if hasattr(ibov_ret,'item'): ibov_ret=ibov_ret.item()
        m['ibov']=round(ibov_ret,2)
        m['alpha']=round(m['retorno']-ibov_ret,2)
    
    # Print
    print(f"\n  📊 CARTEIRA ATIVA (MODELO)")
    print(f"     Retorno: {m['retorno']:>+8.2f}% | Sharpe: {m['sharpe']:.2f} | DD: {m['dd']:.2f}%")
    if m.get('ibov'):
        print(f"  📈 IBOV PASSIVO")
        print(f"     Retorno: {m['ibov']:>+8.2f}%")
        a = m['alpha']
        icon = '✅' if a>0 else '❌'
        print(f"  {icon} ALPHA: {a:+.2f}%")
    print(f"  📋 Trades: {m['trades']} | Win Rate: {m['wr']:.1f}% | PnL: R$ {m['pnl']:+,.2f}")
    
    return {'nome':nome,'start':str(start.date()),'end':str(end.date()),'m':m,'trades':trades,'hist':hist}

def main():
    print(f"\n{'='*60}\n🚀 ROLLING PORTFOLIO BACKTEST — Modelo vs IBOV\n{'='*60}")
    print(f"Universo: {len(ALL_TICKERS)} ações | Carteira: {TOP_N} | Rebalance: {REB_DAYS} dias")
    print(f"Capital: R$ {CAPITAL_INICIAL:,.0f}\n")
    
    hoje = datetime.now()
    resultados = {}
    for nome, delta in [('1_ANO',365),('2_ANOS',730),('3_ANOS',1095)]:
        r = run_period(nome, hoje-timedelta(days=delta), hoje)
        resultados[nome] = r
    
    # Final summary
    print(f"\n{'='*60}\n🏁 RESUMO FINAL\n{'='*60}")
    print(f"  {'Estratégia':<20} {'1 Ano':>12} {'2 Anos':>12} {'3 Anos':>12}")
    print(f"  {'─'*56}")
    for lbl, key in [('Ret. Ativo','retorno'),('Ret. IBOV','ibov'),('ALPHA','alpha'),('Sharpe','sharpe'),('DD','dd'),('Win Rate','wr')]:
        vals = []
        for p in ['1_ANO','2_ANOS','3_ANOS']:
            r=resultados.get(p)
            v=r['m'].get(key) if r else None
            vals.append(f"{v:>+11.2f}"+( "%" if v is not None and 'Sharpe' not in lbl and 'DD' not in lbl and 'Win' not in lbl else ""))
        print(f"  {lbl:<20} {' | '.join(vals)}")
    
    # Verdict
    wins = sum(1 for r in resultados.values() if r and r['m'].get('alpha',0)>0)
    total = sum(1 for r in resultados.values() if r)
    print(f"\n  🏆 VERDITO: Modelo venceu IBOV em {wins}/{total} períodos")
    
    # Save
    out = {}
    for k,r in resultados.items():
        if r: out[k] = {'periodo':r['start']+'→'+r['end'],'metrics':r['m'],'last_trades':r['trades'][-100:]}
    with open('tests/rolling_portfolio_results.json','w') as f: json.dump(out,f,indent=2,default=str)
    print(f"✅ Salvo: tests/rolling_portfolio_results.json")

if __name__=='__main__': main()
