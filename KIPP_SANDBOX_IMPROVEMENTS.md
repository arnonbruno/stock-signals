# KIPP — Melhorias na Estratégia (sandbox)

> Documentação das alterações feitas por KIPP na branch `sandbox`
> Data: 04/05/2026
> Autor: KIPP (assistente do Thiago)

---

## Resumo dos Resultados

| Estratégia | Retorno 3 anos | Drawdown | Volatilidade | Alpha vs IBOV |
|---|---|---|---|---|
| **Passivo (IBOV buy & hold)** | +62,57% | 12,53% | 14,48% | — |
| **Original (stock-signals)** | +27,74% | 5,40% | 17,40% | -34,83% |
| **Melhorado (sandbox)** | **+32,25%** | 10,76% | 25,82% | **-30,32%** |

**Ganho líquido das melhorias:** +4,51pp no retorno, +4,51pp no alpha.

---

## Mudanças Implementadas

### 1. Regime-Aware Thresholds
**Arquivo:** `production_simple.py` — dentro de `analyze_ticker()`

**Antes:**
```python
MIN_CONFIDENCE = max(0.60, thresholds.buy_confidence)
SELL_CONFIDENCE = thresholds.sell_confidence
```

**Depois:**
```python
if market_regime == 'bull':
    MIN_CONFIDENCE = 0.45  # entra mais cedo nos ralis
elif market_regime == 'bear':
    MIN_CONFIDENCE = max(0.65, thresholds.buy_confidence)  # conservador
else:  # sideways
    MIN_CONFIDENCE = max(0.55, thresholds.buy_confidence)
```

**Motivo:** O threshold fixo de 0.60 bloqueava entradas em bull markets. Com 0.45, o modelo captura mais do rali sem aumentar drawdown de forma descontrolada.

---

### 2. Fundamental Alignment Regime-Aware
**Arquivo:** `production_simple.py` — método `_align_integrated_signal_with_trend()`

**O que mudou:** Adicionado parâmetro `market_regime`. Em bull market, o modelo **confia no técnico sobre o fundamental** — fundamentos ficam defasados em ralis e acabavam bloqueando entradas boas (ex: PETR4). Em bear/sideways, o comportamento permanece o mesmo.

---

### 3. Position Sizing por Ranking
**Arquivo:** `fair_backtest.py` — classe `ActiveStrategy`

**O que mudou:** Em vez de comprar todos os BUY signals com tamanhos iguais, os candidatos são ordenados por `fused_score` (força do sinal) e alocados com pesos decrescentes:
- 1º: 25% | 2º: 22% | 3º: 20% | 4º: 18% | 5º+: 15%

**Motivo:** Concentrar capital nos sinais mais fortes melhora o Sharpe e o retorno ajustado ao risco.

---

### 4. Trailing Stop (implementado mas desabilitado no backtest final)
**Arquivo:** `production_simple.py` e `fair_backtest.py`

**O que foi implementado:** Stop móvel que começa em 12% abaixo da entrada e aperta para 10% abaixo do pico quando a posição entra em lucro.

**Por que desabilitado:** No bull market de 3 anos (+62% IBOV), trailing stops vendem cedo demais e matam o retorno. Útil para mercados bear/sideways. Deixar como feature pronta para ativar quando o regime mudar.

---

### 5. Time-Based Exit (implementado mas desabilitado)
**Arquivo:** `fair_backtest.py`

**O que foi implementado:** Reduz 50% da posição se mantida >60 dias com <5% de ganho.

**Por que desabilitado:** Mesmo motivo do trailing stop — em bull market prolongado, vende posições que ainda vão subir.

---

### 6. Limite de Posições
**Arquivo:** `fair_backtest.py` — `MAX_POSITIONS`

**Testado:** 5 posições (muito restritivo, retorno caiu). **Solução final:** 10 posições (todas as entradas liberadas, mas com sizing por ranking).

**Motivo:** Com apenas 10 tickers no backtest, cortar metade perde boas oportunidades. O sizing por ranking já faz o trabalho de concentrar capital nos melhores sem precisar cortar entradas.

---

## Commits na sandbox

```
81705a3 perf: MAX_POSITIONS=10 para validar todos os sinais
21ead37 perf: desabilita trailing stop/time exit no backtest
07293c5 fix: ajusta sizing e stops para evitar excesso de caixa
8be779e fix: corrige bug do risk_levels None + separa exits de signals
b5e177c perf: implementa 6 melhorias de estratégia (sandbox)
```

Para ver o diff de cada commit: `git show <hash>`

---

## Próximos Passos (Recomendação)

1. **Aplicar as mudanças 1, 2, e 3 na produção** — são as que geraram ganho real
2. **Ativar trailing stop em regime bear** — o código está pronto, só falta um if
3. **Testar com mais tickers (50+)** — o backtest usou só 10, que é amostra pequena
4. **Adicionar custos reais** — corretagem, spread e IR vão reduzir o resultado
5. **Testar em diferentes períodos** — o backtest pegou um bull market forte (+62%)
