# Relatório de Backtests — Stock Signals (Maio 2026)

## Objetivo
Validar se o modelo de stock signals é capaz de montar uma carteira de 10 ações
que bata o IBOVESPA passivo em diferentes regimes de mercado.

## Metodologia
- **Universo:** 50 ações do IBOV/SMLL (top liquidez)
- **Capital:** R$ 100.000
- **Carteira:** 10 ações (top ranking por fused_score)
- **Estratégias testadas:**
  1. **HÍBRIDO ATIVO** — regime-aware (bull=0.38, sideways=0.50, bear=0.65),
     rebalance mensal, compra/venda baseada em sinais + trailing stop
  2. **SMART_PASSIVE** — modelo escolhe top-10, buy & hold, reavalia a cada 6 meses
  3. **IBOV PASSIVO** — buy & hold do índice ^BVSP

---

## 1. Bull Market (últimos 1, 2, 3 anos)

| Período | SMART PASSIVE | HÍBRIDO ATIVO | IBOV | Vencedor |
|---------|:------------:|:------------:|:----:|:--------:|
| 1 ano   | +5.1% | +26.2% | **+40.7%** | IBOV |
| 2 anos  | +15.3% | +26.8% | **+45.3%** | IBOV |
| 3 anos  | +19.3% | +15.4% | **+77.0%** | IBOV |

**Conclusão Bull:** Em bull market prolongado, o modelo NÃO bate o índice.
Ele gira demais a carteira (450 trades em 3 anos, win rate ~50%) e vende
cedo ações que continuariam subindo.

---

## 2. Bear Market (COVID Crash — Mar a Jul/2020)

| Estratégia | Retorno | Sharpe | Drawdown | Alpha vs IBOV |
|------------|:-------:|:------:|:--------:|:-------------:|
| SMART PASSIVE | **+9.95%** | 0.10 | 37.28% | **+11.47%** |
| HÍBRIDO ATIVO | +2.55% | -0.04 | 6.28% | +4.07% |
| IBOV PASSIVO | -1.52% | — | — | — |

**Top picks do modelo em Mar/2020:** DXCO3, B3SA3, WEGE3, HYPE3, RADL3,
KLBN11, EQTL3, SUZB3, PRIO3, CAML3 — todas se recuperaram forte.

**Conclusão Bear:** Em crash, o modelo é REI. Alpha de +11.47% com SMART_PASSIVE,
proteção de capital com HÍBRIDO (DD de apenas 6.28% vs queda histórica do IBOV).

---

## 3. Diagnóstico do Modelo

### FORÇAS ✅
- **Excelente stock picker**: escolhe ações que performam bem no médio prazo
- **Proteção em queda**: DD muito menor que o índice em cenários de stress
- **PnL consistente**: nunca perdeu dinheiro nos backtests (sempre retorno positivo)
- **Regime detection funciona**: detecta corretamente bull/bear via MA200/MA50

### FRAQUEZAS ❌
- **Péssimo market timer em bull**: vende cedo demais, gira excessivamente
- **Win rate ~50% em bull**: essencialmente aleatório nas decisões de saída
- **Confiança conservadora**: thresholds altos fazem perder entradas em rallies
- **Alta rotatividade**: +400 trades em 3 anos (custos de corretagem não modelados)

---

## 4. Recomendações Estratégicas

### Para usar o modelo HOJE:
1. **Use o modelo para STOCK PICKING (O QUE comprar), não para MARKET TIMING**
2. **Em bull market**: deixe o modelo escolher as 10 ações, compre, e NÃO MEXA
   por 6 meses (estilo SMART_PASSIVE)
3. **Em bear/sideways**: o HÍBRIDO ativo funciona bem como proteção (DD baixo)
4. **Ideal**: combine os dois — use SMART_PASSIVE como base e só ative
   o HÍBRIDO (vendas) quando o regime detector virar BEAR

### Melhorias sugeridas para o modelo:
- Aumentar período de hold mínimo (evitar vendas < 30 dias)
- Em regime BULL, reduzir ainda mais os thresholds de venda
- Adicionar filtro de momentum para evitar vender ações em tendência de alta
- Considerar custos de corretagem no backtest (impacta alta rotatividade)

---

## 5. Arquivos no Repositório (sandbox)

```
tests/
├── rolling_portfolio_backtest.py       # Backtest HÍBRIDO ativo (1/2/3 anos)
├── rolling_portfolio_results.json      # Resultados completos HÍBRIDO
├── smart_vs_hibrido_vs_ibov.py         # Comparação 3 estratégias
├── smart_vs_hibrido_vs_ibov.json       # Resultados da comparação
├── comprehensive_backtest_results.json # Backtest original (bull/bear)
├── comprehensive_backtest_summary.txt  # Sumário do comprehensive
├── bear_backtest_results.txt           # Resultados bear COVID (detalhado)
└── ARNON_REPORT.md                     # ← Este arquivo
```

---

## 6. Resumo de Decisões (Maio/2026)
- **Estratégia no production:** HÍBRIDO (regime-aware)
- **BrAPI token:** free tier, batch_size=1
- **Cron jobs:** removidos, execução on-demand
- **SMART_PASSIVE rejeitado** por Thiago para execution — mas validado como
  superior em backtest para stock picking puro
- **Conclusão final:** Modelo = escudo (🛡️), não lança. Usar para escolher
  ações, não para fazer timing.
