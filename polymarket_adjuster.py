#!/usr/bin/env python3
"""
Polymarket Final Adjuster — Ajustador de final de fluxo

Combina dois ajustadores que entram no FINAL da produção:

  1. HormuzAdjuster   → Ajusta oil tickers baseado no score Ormuz
  2. MacroAdjuster    → Ajusta convicção geral baseado em juros/inflação

Uso:
    from polymarket_adjuster import PolymarketAdjuster
    
    adjuster = PolymarketAdjuster()
    result = adjuster.adjust(results)
    # result.results_ajustados, result.summary, result.alert
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("polymarket_adjuster")

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURAÇÃO COMPARTILHADA
# ═══════════════════════════════════════════════════════════════════════

GAMMA_API = "https://gamma-api.polymarket.com"

# ── Grupos de tickers (mesma empresa, não dobram boost) ────────────────
# Ticker -> group_id. Mesmo group_id = compartilha capacidade de boost.
TICKER_GROUPS = {
    "PETR4": "petrobras",
    "PETR3": "petrobras",
}

# ── Sensibilidade ao petróleo por ticker ───────────────────────────────
# 0.0 = sem exposição, 1.0 = máxima
OIL_SENSITIVITY = {
    "PETR4": 1.0,   # Petrobras PN
    "PETR3": 1.0,   # Petrobras ON
    "PRIO3": 0.9,   # Prio (petróleo puro)
    "VBBR3": 0.7,   # Vibra (BR Distribuidora)
    "UGPA3": 0.6,   # Ultrapar (Ipiranga)
    "BRKM5": 0.5,   # Braskem (nafta)
    "CSAN3": 0.3,   # Cosan (Raízen, energia)
}

# ── Sensibilidade a juros/inflação por ticker ──────────────────────────
# Para o MacroAdjuster. Quanto o ticker sobe/desce com juros.
# 1.0 = muito sensível (bancos, consumo), 0.0 = imune (commodities)
RATE_SENSITIVITY = {
    # Bancos (ganham com juros altos)
    "ITUB4": 0.8, "ITSA4": 0.7, "BBDC4": 0.8, "BBDC3": 0.8,
    "BBAS3": 0.7, "SANB11": 0.7, "BPAC11": 0.6,
    # Consumo / Varejo (perdem com juros altos)
    "MGLU3": 0.7, "LREN3": 0.6, "AMAR3": 0.6, "ARZZ3": 0.5,
    "VIIA3": 0.7, "CEAB3": 0.6, "PETZ3": 0.5,
    "BHIA3": 0.5, "CRFB3": 0.4, "PCAR3": 0.5, "GMAT3": 0.5,
    # Construção (sensível a juros)
    "MRVE3": 0.7, "CYRE3": 0.6, "EZTC3": 0.6, "DIRR3": 0.6,
    "EVEN3": 0.6, "TEND3": 0.6, "CURY3": 0.5, "LAVV3": 0.5,
    # Concessionárias (juros afetam dívida)
    "EGIE3": 0.4, "NEOE3": 0.4, "TAEE11": 0.4, "CMIG4": 0.4,
    "SBSP3": 0.4, "ENBR3": 0.4, "EQTL3": 0.3,
}

# ═══════════════════════════════════════════════════════════════════════
# MACRO ADJUSTER
# ═══════════════════════════════════════════════════════════════════════

class MacroAdjuster:
    """
    Ajusta o cenário geral baseado em juros/inflação do Polymarket.
    
    Puxa scores macro do módulo polymarket_sentiment.py (tags: interest-rates,
    macro-indicators, macro-single) e ajusta:
    
    - cash_buffer: mais caixa se juros sobem (bearish), menos se caem (bullish)
    - conviction_mult: convicção geral (score macro alto = mais confiança)
    - setor financeiro vs consumo: rotaciona baseado em projeção de juros
    """

    def __init__(self, macro_score: Optional[float] = None,
                 macro_detalhes: Optional[Dict] = None):
        self.macro_score = macro_score
        self.macro_detalhes = macro_detalhes

    # ── Data fetching ─────────────────────────────────────────────────

    def _fetch_macro_scores(self) -> Dict:
        """Puxa scores macro do Polymarket (mesma lógica do polymarket_sentiment.py)."""
        TAGS = {
            "interest-rates": (0.35, "Taxa de Juros", False),
            "macro-indicators": (0.35, "Inflação / PIB", False),
            "macro-single": (0.30, "Macro Geral", True),
        }

        scores = {}
        scores_pond = []

        for slug, (peso, desc, high_is_bullish) in TAGS.items():
            try:
                r = requests.get(
                    f"{GAMMA_API}/events",
                    params={"tag_slug": slug, "closed": "false", "limit": 5},
                    timeout=10,
                )
                eventos = r.json()
                prices = []
                for ev in eventos:
                    markets = ev.get("markets") or []
                    for m in markets:
                        ps = m.get("outcomePrices", "[]")
                        try:
                            p = json.loads(ps) if isinstance(ps, str) else ps
                            yes = float(p[0]) if p else None
                            if yes is not None:
                                prices.append(yes)
                        except (ValueError, TypeError, json.JSONDecodeError):
                            pass

                if prices:
                    avg = sum(prices) / len(prices)
                    adj = avg if high_is_bullish else (1.0 - avg)
                    scores[slug] = {
                        "desc": desc,
                        "raw": round(avg, 3),
                        "adj": round(adj, 3),
                        "peso": peso,
                        "mercados": len(prices),
                    }
                    scores_pond.append((adj, peso))
            except Exception as e:
                logger.warning(f"⚠️ macro fetch {slug}: {e}")

        if not scores_pond:
            return {"composite": 0.5, "tags": {}}

        composite = sum(s * p for s, p in scores_pond) / sum(p for _, p in scores_pond)
        composite = max(0.0, min(1.0, composite))

        return {"composite": round(composite, 3), "tags": scores}

    def _get_macro_label(self, score: float) -> Tuple[str, str]:
        """Classifica o cenário macro."""
        if score >= 0.70:
            return ("🟢 MACRO BULLISH",
                    "Juros baixos, inflação controlada. Cenário favorável para bolsa. "
                    "Reduzir caixa, aumentar exposição a consumo e construção.")
        elif score >= 0.55:
            return ("🔵 MACRO LEVE",
                    "Cenário macro positivo. Manter posições, pequeno viés de alta.")
        elif score >= 0.40:
            return ("🟡 MACRO NEUTRO",
                    "Sinais mistos. Manter alocação padrão, sem viés macro forte.")
        elif score >= 0.25:
            return ("🟠 MACRO LEVE BAIXA",
                    "Juros podem subir. Aumentar caixa, favorecer bancos sobre consumo.")
        return ("🔴 MACRO BEARISH",
                "Inflação ou juros subindo. Aumentar caixa, favorecer setores defensivos.")

    def _calc_cash_adjustment(self, score: float) -> float:
        """
        Ajuste no cash buffer baseado no score macro.
        
        Score 0.0 (bearish) → +20% cash
        Score 0.5 (neutro) → 0% ajuste
        Score 1.0 (bullish) → -10% cash (investir mais)
        """
        return round((0.5 - score) * 0.40, 3)

    def _calc_conviction_mult(self, score: float) -> float:
        """
        Multiplicador de convicção baseado no score macro.
        
        Score 0.0 → 0.80x (menos confiança nos sinais)
        Score 0.5 → 1.00x
        Score 1.0 → 1.15x (mais confiança)
        """
        return round(0.80 + score * 0.35, 3)

    # ── Public API ────────────────────────────────────────────────────

    def get_score(self) -> Dict:
        if self.macro_score is not None:
            return {"composite": self.macro_score, "tags": self.macro_detalhes or {}}
        result = self._fetch_macro_scores()
        self.macro_score = result["composite"]
        self.macro_detalhes = result["tags"]
        return result

    def adjust(self, results: List[Dict]) -> Dict:
        """
        Aplica ajustes macro nos resultados.
        
        Returns:
            results_ajustados, summary (cash_adj, conviction_mult, etc.)
        """
        macro = self.get_score()
        score = macro["composite"]
        label, desc = self._get_macro_label(score)
        cash_adj = self._calc_cash_adjustment(score)
        conviction_mult = self._calc_conviction_mult(score)

        ajustes = []
        for r in results:
            ticker = r.get("ticker", "")
            conviction = r.get("conviction", 0.5)
            pos = r.get("position_size", 0.0)

            rate_sens = RATE_SENSITIVITY.get(ticker, 0.0)
            r["_rate_sensitivity"] = rate_sens

            # Rate-sensitive adjustment: se juros vão cair (macro bullish),
            # consumo e construção sobem; se vão subir, bancos sobem
            if rate_sens > 0 and pos > 0:
                # interest-rate tag específica
                ir_tag = macro.get("tags", {}).get("interest-rates", {})
                ir_raw = ir_tag.get("raw", 0.5)

                if rate_sens >= 0.6:
                    # Alta sensibilidade: se juros vão cair (raw baixo), 
                    # consumo sobe; se vão subir, bancos sobem
                    if ticker in RATE_SENSITIVITY and RATE_SENSITIVITY[ticker] >= 0.6:
                        is_bank = ticker in ("ITUB4", "ITSA4", "BBDC4", "BBDC3",
                                             "BBAS3", "SANB11", "BPAC11")
                        if is_bank:
                            # Bancos: quanto maior a chance de juros subirem, melhor
                            bank_boost = 1.0 + (ir_raw - 0.3) * 0.3
                            r["position_size"] = round(min(pos * bank_boost, 0.50), 3)
                            r["_macro_setor"] = "bancos"
                        else:
                            # Consumo/construção: juros caindo = bom
                            cons_boost = 1.0 + (0.5 - ir_raw) * 0.3
                            r["position_size"] = round(min(pos * cons_boost, 0.40), 3)
                            r["_macro_setor"] = "consumo"

            # Convicção ajustada
            r["_conviction_original"] = round(conviction, 3)
            r["conviction"] = round(conviction * conviction_mult, 3)
            r["_macro_score"] = score
            r["_macro_label"] = label
            r["_conviction_mult"] = conviction_mult
            ajustes.append(r)

        return {
            "results_ajustados": ajustes,
            "summary": {
                "macro_score": score,
                "label": label,
                "descricao": desc,
                "cash_adjustment": cash_adj,
                "conviction_multiplier": conviction_mult,
                "detalhes_tags": macro.get("tags", {}),
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# HORMUZ ADJUSTER (refatorado com grupos)
# ═══════════════════════════════════════════════════════════════════════

SCORE_ACTIVATION = 0.25


class HormuzAdjuster:
    """
    Ajusta oil tickers baseado no score de disrupção do Estreito de Ormuz.
    Agora com suporte a grupos (PETR3/PETR4 não dobram boost).
    """

    def __init__(self, score: Optional[float] = None, detalhes: Optional[Dict] = None):
        self.score = score
        self.detalhes = detalhes
        self._prev_score = self._load_prev_score()

    def _load_prev_score(self) -> Optional[float]:
        mf = Path(__file__).parent / "hormuz_correlacao_history.json"
        if mf.exists():
            try:
                with open(mf) as f:
                    h = json.load(f)
                if len(h) >= 2:
                    return h[-2].get("hormuz_score")
            except Exception:
                pass
        return None

    def _fetch_score(self) -> Tuple[float, Dict]:
        """Puxa score Ormuz atual do Polymarket."""
        HORMUZ = [
            ("strait-of-hormuz-traffic-returns-to-normal-by-may-15", 0.20),
            ("strait-of-hormuz-traffic-returns-to-normal-by-end-of-may", 0.20),
            ("strait-of-hormuz-traffic-returns-to-normal-by-end-of-june", 0.15),
            ("trump-announces-us-blockade-of-hormuz-lifted-by", 0.15),
            ("kharg-island-no-longer-under-iranian-control-by-march-31", 0.10),
            ("bab-el-mandeb-strait-effectively-closed-by", 0.10),
            ("iran-agrees-to-unrestricted-shipping-through-hormuz-by-may-31", 0.10),
        ]

        scores = []
        mercados = []

        for slug, peso in HORMUZ:
            try:
                r = requests.get(f"{GAMMA_API}/events", params={"slug": slug, "limit": 1}, timeout=10)
                events = r.json()
                if not events:
                    continue
                ev = events[0]
                eid = ev.get("id")
                if not eid:
                    continue
                r2 = requests.get(f"{GAMMA_API}/events/{eid}", timeout=10)
                ev_full = r2.json()
                prices = []
                for m in ev_full.get("markets", []):
                    ps = m.get("outcomePrices", "[]")
                    try:
                        parsed = json.loads(ps) if isinstance(ps, str) else ps
                        yes = float(parsed[0]) if parsed else None
                        if yes is not None:
                            prices.append(yes)
                    except (ValueError, TypeError, json.JSONDecodeError):
                        pass
                if prices:
                    avg_yes = sum(prices) / len(prices)
                    disruption = 1.0 - avg_yes
                    scores.append((disruption, peso))
                    mercados.append({
                        "slug": slug,
                        "prob_normalizar": round(avg_yes, 3),
                        "score_disrupcao": round(disruption, 3),
                    })
            except Exception as e:
                logger.warning(f"⚠️ fetch {slug}: {e}")

        if not scores:
            return 0.5, {"score": 0.5, "mercados": []}

        pt = sum(p for _, p in scores)
        score = sum(s * p for s, p in scores) / pt
        score = max(0.0, min(1.0, score))
        return round(score, 3), {"score": round(score, 3), "mercados": mercados}

    def _calc_boost(self, score: float) -> float:
        if score < SCORE_ACTIVATION:
            return 1.0
        return 1.0 + (score ** 1.5)

    def _calc_reduction(self, score: float) -> float:
        if score < SCORE_ACTIVATION:
            return 1.0
        return 1.0 - (score ** 1.5) * 0.15

    def _detect_drop(self, current: float, prev: Optional[float]) -> Optional[str]:
        if prev is None or current is None:
            return None
        drop = prev - current
        if drop >= 0.20:
            return f"🚨 CRÍTICO: Score Ormuz caiu {drop:.0%}. Risco geopolítico despencando. REALIZE LUCRO em petróleo."
        if drop >= 0.10:
            return f"⚠️ ALERTA: Score Ormuz caiu {drop:.0%}. Tendência de normalização. Ajuste posições em oil."
        return None

    def _get_label(self, score: float) -> Tuple[str, str]:
        if score >= 0.80:
            return ("🔥 DISRUPÇÃO EXTREMA", "Oil super-pesado. Prêmio de risco máximo no petróleo.")
        if score >= 0.60:
            return ("⛽ DISRUPÇÃO ALTA", "Oil sobreponderado. Ormuz bloqueado, petróleo sustentado.")
        if score >= 0.40:
            return ("⚠️ RISCO MODERADO", "Oil levemente sobreponderado. Cenário geopolítico incerto.")
        if score >= SCORE_ACTIVATION:
            return ("✅ RISCO BAIXO", "Sem ajuste significativo. Normalização próxima.")
        return ("✅ NORMAL", "Mercado de petróleo sem disrupção geopolítica.")

    def get_score(self) -> Tuple[float, Dict]:
        if self.score is not None:
            return self.score, self.detalhes or {}
        s, d = self._fetch_score()
        self.score = s
        self.detalhes = d
        return s, d

    def adjust(self, results: List[Dict]) -> Dict:
        """
        Ajusta oil tickers com suporte a grupos.
        Mesmo grupo = compartilha boost (PETR3/PETR4 não dobram).
        """
        score, detalhes = self.get_score()

        if score < SCORE_ACTIVATION:
            return {
                "results_ajustados": results,
                "summary": {"score": score, "ajuste_aplicado": False,
                            "motivo": f"Score {score:.3f} abaixo do threshold"},
                "alert": None,
            }

        boost = self._calc_boost(score)
        reduction = self._calc_reduction(score)
        label, desc = self._get_label(score)
        drop_alert = self._detect_drop(score, self._prev_score)

        # Track grupos já boostados
        grupos_boostados = set()
        ajustes = []
        adjusted = []

        for r in results:
            ticker = r.get("ticker", "")
            pos = r.get("position_size", 0.0)
            signal = r.get("signal", "HOLD")
            sens = OIL_SENSITIVITY.get(ticker, 0.0)
            grupo = TICKER_GROUPS.get(ticker)

            if sens > 0 and pos > 0:
                # Verifica se o grupo já foi boostado
                if grupo and grupo in grupos_boostados:
                    # Já boostado neste grupo: boost reduzido
                    effective_boost = 1.0 + (boost - 1.0) * 0.3
                else:
                    effective_boost = boost
                    if grupo:
                        grupos_boostados.add(grupo)

                new_pos = pos * (1.0 + (effective_boost - 1.0) * sens)
                new_pos = min(new_pos, 0.70)

                new_signal = signal
                if sens >= 0.7 and score >= 0.6:
                    if signal == "HOLD":
                        new_signal = "BUY"
                    elif signal == "BUY":
                        new_signal = "STRONG_BUY"

                r["position_size"] = round(new_pos, 3)
                if new_signal != signal:
                    r["signal"] = new_signal
                    r["_hormuz_upgrade"] = f"{signal}→{new_signal}"
                r["_hormuz_boost"] = round(effective_boost, 3)
                r["_hormuz_sens"] = sens

                ajustes.append({
                    "ticker": ticker,
                    "grupo": grupo or ticker,
                    "sens": sens,
                    "boost": round(effective_boost, 3),
                    "pos_original": round(pos, 3),
                    "pos_ajustada": round(new_pos, 3),
                    "signal_original": signal,
                    "signal_ajustado": new_signal,
                })

            elif sens == 0 and pos > 0:
                nr = max(reduction, 0.88 if score >= 0.7 else 0.90)
                new_pos = pos * nr
                if abs(new_pos - pos) > 0.01:
                    r["position_size"] = round(new_pos, 3)

            r["_hormuz_score"] = score
            r["_hormuz_label"] = label
            adjusted.append(r)

        return {
            "results_ajustados": adjusted,
            "summary": {
                "score": score,
                "label": label,
                "descricao": desc,
                "ajuste_aplicado": True,
                "oil_boost": round(boost, 3),
                "non_oil_reduction": round(reduction, 3),
                "tickers_ajustados": len(ajustes),
                "detalhes_ajustes": ajustes,
                "grupos_boostados": list(grupos_boostados),
                "drop_alert": drop_alert,
            },
            "alert": drop_alert,
        }


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED ADJUSTER
# ═══════════════════════════════════════════════════════════════════════

class PolymarketAdjuster:
    """
    Ajustador unificado que roda Hormuz + Macro no final do fluxo.
    
    Ordem de execução:
      1. HormuzAdjuster  → ajusta oil tickers
      2. MacroAdjuster    → ajusta convicção geral + setores
    
    Returns:
        results_ajustados, summary (unificado), alerts
    """

    def __init__(self):
        self.hormuz = HormuzAdjuster()
        self.macro = MacroAdjuster()

    def adjust(self, results: List[Dict]) -> Dict:
        # Passo 1: Hormuz
        h_result = self.hormuz.adjust(results)
        h_results = h_result["results_ajustados"]
        h_summary = h_result["summary"]

        # Passo 2: Macro
        m_result = self.macro.adjust(h_results)
        m_results = m_result["results_ajustados"]
        m_summary = m_result["summary"]

        # Monta resumo unificado
        score_ormuz = h_summary.get("score", 0.5)
        score_macro = m_summary.get("macro_score", 0.5)

        oil_boost = h_summary.get("oil_boost", 1.0)

        alerts = []
        if h_result.get("alert"):
            alerts.append(h_result["alert"])

        # Alerta de conflito: macro bearish + oil em alta
        if score_macro < 0.40 and score_ormuz > 0.60:
            alerts.append(
                "⚠️ CONFLITO: Macro bearish (juros/inflação) mas Ormuz em disrupção. "
                "Hedge contraditório: oil sobe com Ormuz, mas juros altos pressionam bolsa geral."
            )

        return {
            "results_ajustados": m_results,
            "alerts": alerts,
            "summary": {
                "hormuz": {
                    "score": score_ormuz,
                    "label": h_summary.get("label", "?"),
                    "oil_boost": oil_boost,
                    "ajustes": h_summary.get("detalhes_ajustes", []),
                    "aplicado": h_summary.get("ajuste_aplicado", False),
                },
                "macro": {
                    "score": score_macro,
                    "label": m_summary.get("label", "?"),
                    "conviction_mult": m_summary.get("conviction_multiplier", 1.0),
                    "cash_adj": m_summary.get("cash_adjustment", 0.0),
                    "tags": m_summary.get("detalhes_tags", {}),
                },
                "tickers_ajustados": sum(
                    1 for r in m_results
                    if r.get("_hormuz_boost", 1.0) > 1.0 or r.get("_conviction_mult", 1.0) != 1.0
                ),
            },
        }


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════

def print_summary(adj_result: Dict):
    s = adj_result["summary"]
    h = s["hormuz"]
    m = s["macro"]
    alerts = adj_result.get("alerts", [])

    print(f"\n{'='*70}")
    print(f"  🛢️ 📊 AJUSTADOR POLYMARKET — RESUMO FINAL")
    print(f"{'='*70}")

    if h["aplicado"]:
        print(f"\n  🌊 ORMUZ: {h['score']:.3f} — {h['label']}")
        print(f"     Boost oil: {h['oil_boost']:.2f}x")
        for a in h["ajustes"]:
            sig = a.get("signal_ajustado", "")
            orig = a.get("signal_original", "")
            sd = f" {orig}→{sig}" if sig != orig else ""
            print(f"     • {a['ticker']:<6}: {a['pos_original']:.0%} → {a['pos_ajustada']:.0%} ({a['boost']:.2f}x){sd}")
    else:
        print(f"\n  🌊 ORMUZ: {h['score']:.3f} — sem ajuste (abaixo do threshold)")

    print(f"\n  📊 MACRO: {m['score']:.3f} — {m['label']}")
    print(f"     Convicção: {m['conviction_mult']:.2f}x")
    cash = m.get("cash_adj", 0)
    print(f"     Ajuste caixa: {cash:+.0%}")
    for slug, tag in m.get("tags", {}).items():
        print(f"     • {tag.get('desc','?')}: {tag.get('adj',0):.3f}")

    if alerts:
        print(f"\n  🚨 ALERTAS:")
        for a in alerts:
            print(f"     • {a}")

    print(f"\n  📋 Total tickers ajustados: {s['tickers_ajustados']}")
    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════
# STANDALONE
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Simula com dados mockados
    mock = []
    for t in list(OIL_SENSITIVITY.keys())[:4]:
        mock.append({"ticker": t, "position_size": 0.20, "signal": "BUY",
                      "conviction": 0.65, "trend": "uptrend", "composite_score": 70})
    for t in ["VALE3", "ITUB4", "WEGE3", "BBAS3", "B3SA3", "MGLU3", "MRVE3"]:
        mock.append({"ticker": t, "position_size": 0.15, "signal": "HOLD",
                      "conviction": 0.50, "trend": "consolidation", "composite_score": 55})

    adj = PolymarketAdjuster()
    result = adj.adjust(mock)
    print_summary(result)