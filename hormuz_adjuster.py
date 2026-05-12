#!/usr/bin/env python3
"""
HormuzAdjuster — Ajustador geopolítico de mercado

Entra no FINAL do fluxo de produção e ajusta:
  - position_size para tickers ligados a petróleo
  - Sinais (BUY/SELL) baseado no risco de disrupção do Estreito de Ormuz
  
Filosofia:
  O score Ormuz não dita o regime de mercado — ele ADICIONA um viés setorial
  no topo da análise fundamentalista/técnica já feita.

Uso:
    from hormuz_adjuster import HormuzAdjuster
    
    adjuster = HormuzAdjuster()
    results = adjuster.adjust(results)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("hormuz_adjuster")

# ── Sensibilidade ao petróleo por ticker ───────────────────────────────
# Multiplicador de quanto o score Ormuz afeta o position_size do ticker
# 0.0 = sem exposição a petróleo, 1.0 = exposição direta máxima
OIL_SENSITIVITY = {
    # —— Exploração & Produção (Petrobras) ——
    "PETR4": 1.0,   # PN, liquidez máxima
    "PETR3": 1.0,   # ON
    # —— Exploração & Produção (Privadas) ——
    "PRIO3": 0.9,   # Prio (ex-QGEP), petróleo puro
    # —— Distribuição / Combustíveis ——
    "VBBR3": 0.7,   # Vibra (ex-BR Distribuidora)
    "UGPA3": 0.6,   # Ultrapar (Ipiranga, Oxiteno)
    # —— Petroquímica ——
    "BRKM5": 0.5,   # Braskem (nafta = petróleo)
    # —— Conglomerados com energia ——
    "CSAN3": 0.3,   # Cosan (Raízen, diesel, mas diversificado)
}

# Score mínimo pra ativar o ajuste (abaixo disso: normalidade, sem ajuste)
SCORE_ACTIVATION_THRESHOLD = 0.25


class HormuzAdjuster:
    """
    Ajustador de portfólio baseado no risco de disrupção do Estreito de Ormuz.
    
    Como funciona:
      1. Puxa score Ormuz do Polymarket (ou usa score passado)
      2. Se score > threshold, aumenta posição em oil tickers
      3. Se score cai abruptamente (>0.15 em 1 dia), gera alerta de take-profit
         em oil tickers
      4. Ajusta a distribuição geral: mais peso em oil = menos nos outros
    """

    def __init__(self, score: Optional[float] = None, detalhes: Optional[Dict] = None):
        self.score = score
        self.detalhes = detalhes
        self._prev_score = self._load_prev_score()
        self.oil_tickers = list(OIL_SENSITIVITY.keys())

    # ── Score loading ──────────────────────────────────────────────────

    def _load_prev_score(self) -> Optional[float]:
        """Carrega o último score registrado no monitor."""
        monitor_file = Path(__file__).parent / "hormuz_correlacao_history.json"
        if monitor_file.exists():
            try:
                with open(monitor_file) as f:
                    history = json.load(f)
                if len(history) >= 2:
                    return history[-2].get("hormuz_score")
                elif len(history) == 1:
                    return history[-1].get("hormuz_score")
            except Exception:
                pass
        return None

    def _fetch_score(self) -> Tuple[float, Dict]:
        """Puxa score atual do Polymarket."""
        GAMMA_API = "https://gamma-api.polymarket.com"

        HORMUZ_MARKETS = [
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

        for slug, peso in HORMUZ_MARKETS:
            try:
                r = requests.get(
                    f"{GAMMA_API}/events", params={"slug": slug, "limit": 1}, timeout=10
                )
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

        peso_total = sum(p for _, p in scores)
        score = sum(s * p for s, p in scores) / peso_total
        score = max(0.0, min(1.0, score))

        return round(score, 3), {
            "score": round(score, 3),
            "mercados": mercados,
        }

    # ── Core adjustment logic ──────────────────────────────────────────

    def _calc_oil_boost(self, score: float) -> float:
        """
        Calcula o fator de boost para oil tickers baseado no score Ormuz.

        Score 0.0 → boost 1.0x (sem ajuste)
        Score 0.5 → boost 1.3x (risco moderado)
        Score 0.73 → boost 1.6x (disrupção alta, como agora)
        Score 1.0 → boost 2.0x (disrupção máxima)

        Curva: 1.0 + score^1.5 (curva côncava: impacto maior em scores altos)
        """
        if score < SCORE_ACTIVATION_THRESHOLD:
            return 1.0
        return 1.0 + (score ** 1.5)

    def _calc_oil_reduction(self, score: float) -> float:
        """
        Fator de redução para tickers NÃO-oil quando score está alto.
        Redistribui o peso extra que foi pra oil tickers.

        Score 0.0 → 1.0 (sem redução)
        Score 0.73 → 0.92 (redução de 8% nos não-oil)
        Score 1.0 → 0.85 (redução de 15%)
        """
        if score < SCORE_ACTIVATION_THRESHOLD:
            return 1.0
        return 1.0 - (score ** 1.5) * 0.15

    def _detect_score_drop(self, current: float, prev: Optional[float]) -> Optional[str]:
        """
        Detecta queda abrupta no score Ormuz.
        Se caiu muito rápido, pode indicar que o bloqueio está se resolvendo
        → sinal de TAKE PROFIT em oil.
        """
        if prev is None or current is None:
            return None

        drop = prev - current
        if drop >= 0.20:
            return f"CRÍTICO: Score Ormuz caiu {drop:.0%} em 1 dia. Risco geopolítico diminuindo RAPIDAMENTE. Considere realizar lucro em petróleo."
        elif drop >= 0.10:
            return f"ALERTA: Score Ormuz caiu {drop:.0%}. Tendência de normalização. Ajuste posições em oil."
        return None

    def _get_signal_adjustment(self, score: float) -> Tuple[str, str]:
        """
        Sugere ajuste de sinal baseado no score.
        Retorna (label, descrição)
        """
        if score >= 0.80:
            return ("🔥 DISRUPÇÃO EXTREMA",
                    "Oil super-pesado. Petróleo deve continuar subindo com prêmio de risco.")
        elif score >= 0.60:
            return ("⛽ DISRUPÇÃO ALTA",
                    "Oil sobreponderado. Ormuz bloqueado, petróleo sustentado.")
        elif score >= 0.40:
            return ("⚠️ RISCO MODERADO",
                    "Oil levemente sobreponderado. Cenário geopolítico incerto.")
        elif score >= SCORE_ACTIVATION_THRESHOLD:
            return ("✅ RISCO BAIXO",
                    "Sem ajuste significativo em oil. Normalização próxima.")
        return ("✅ NORMALIDADE", "Mercado de petróleo sem disrupção geopolítica.")

    # ── Public API ─────────────────────────────────────────────────────

    def get_score(self) -> Tuple[float, Dict]:
        """Retorna score Ormuz atual."""
        if self.score is not None:
            return self.score, self.detalhes or {}
        score, detalhes = self._fetch_score()
        self.score = score
        self.detalhes = detalhes
        return score, detalhes

    def adjust(self, results: List[Dict]) -> Dict:
        """
        Ponto de entrada principal.
        
        Args:
            results: Lista de resultados da produção (cada um com 'ticker',
                    'position_size', 'signal', etc.)
        
        Returns:
            Dict com:
              - results_ajustados: Lista ajustada
              - summary: Resumo dos ajustes feitos
              - alert: Alerta de take-profit se aplicável
        """
        score, detalhes = self.get_score()

        if score < SCORE_ACTIVATION_THRESHOLD:
            return {
                "results_ajustados": results,
                "summary": {
                    "score": score,
                    "ajuste_aplicado": False,
                    "motivo": f"Score {score:.3f} abaixo do threshold {SCORE_ACTIVATION_THRESHOLD}",
                },
                "alert": None,
            }

        oil_boost = self._calc_oil_boost(score)
        non_oil_reduction = self._calc_oil_reduction(score)
        label, desc = self._get_signal_adjustment(score)
        drop_alert = self._detect_score_drop(score, self._prev_score)

        ajustes = []
        adjusted_results = []

        for r in results:
            ticker = r.get("ticker", "")
            pos = r.get("position_size", 0.0)
            signal = r.get("signal", "HOLD")

            sensitivity = OIL_SENSITIVITY.get(ticker, 0.0)

            if sensitivity > 0 and pos > 0:
                # Aplica boost proporcional à sensibilidade
                new_pos = pos * (1.0 + (oil_boost - 1.0) * sensitivity)
                new_pos = min(new_pos, 0.80)  # Cap em 80%

                # Upgrade de sinal se posição > 2x ou score muito alto
                new_signal = signal
                if sensitivity >= 0.7 and score >= 0.6:
                    if signal == "HOLD":
                        new_signal = "BUY"
                    elif signal == "BUY":
                        new_signal = "STRONG_BUY"

                r["position_size"] = round(new_pos, 3)
                if new_signal != signal:
                    r["signal"] = new_signal
                    r["_hormuz_signal_upgrade"] = f"{signal}→{new_signal}"

                r["_hormuz_boost"] = round(oil_boost, 3)
                r["_hormuz_sensitivity"] = sensitivity

                ajustes.append({
                    "ticker": ticker,
                    "sensitivity": sensitivity,
                    "pos_original": round(pos, 3),
                    "pos_ajustada": round(new_pos, 3),
                    "boost": round(oil_boost, 3),
                    "signal_original": signal,
                    "signal_ajustado": new_signal,
                })

            elif sensitivity == 0 and pos > 0:
                # Reduz posição em tickers não-oil pra compensar
                new_pos = pos * non_oil_reduction
                # Mas não reduz abaixo do que já estava
                new_pos = max(new_pos, pos * 0.85)
                r["position_size"] = round(new_pos, 3)

            r["_hormuz_score"] = score
            r["_hormuz_label"] = label
            adjusted_results.append(r)

        total_oil_weight = sum(
            r["position_size"]
            for r in adjusted_results
            if OIL_SENSITIVITY.get(r.get("ticker", ""), 0) >= 0.7
        )

        summary = {
            "score": score,
            "label": label,
            "descricao": desc,
            "ajuste_aplicado": True,
            "oil_boost": round(oil_boost, 3),
            "non_oil_reduction": round(non_oil_reduction, 3),
            "tickers_ajustados": len(ajustes),
            "detalhes_ajustes": ajustes,
            "peso_total_oil": round(total_oil_weight, 3),
            "drop_alert": drop_alert,
        }

        return {
            "results_ajustados": adjusted_results,
            "summary": summary,
            "alert": drop_alert,
        }


def print_adjuster_summary(adjustment_result: Dict):
    """Exibe o resumo do ajuste no terminal."""
    summary = adjustment_result.get("summary", {})
    alert = adjustment_result.get("alert")

    print(f"\n{'='*70}")
    print(f"🛢️  AJUSTADOR ORMUZ - RESUMO")
    print(f"{'='*70}")

    if not summary.get("ajuste_aplicado"):
        print(f"\n   ⏭️  {summary.get('motivo', 'Nenhum ajuste necessário')}")
        print(f"{'='*70}\n")
        return

    print(f"\n   🌊 Score Ormuz: {summary.get('score', 0):.3f} — {summary.get('label', '?')}")
    print(f"   📈 Boost oil: {summary.get('oil_boost', 1):.2f}x")
    print(f"   📉 Redução não-oil: {(1 - summary.get('non_oil_reduction', 1))*100:.0f}%")
    print(f"   💰 Peso total oil: {(summary.get('peso_total_oil', 0)*100):.0f}%")

    ajustes = summary.get("detalhes_ajustes", [])
    if ajustes:
        print(f"\n   📋 Tickers ajustados ({len(ajustes)}):")
        print(f"   {'Ticker':<8} {'Sens':>5} {'Pos Antes':>10} {'Pos Depois':>11} {'Sinal':>8}")
        print(f"   {'-'*8} {'-'*5} {'-'*10} {'-'*11} {'-'*8}")
        for a in ajustes:
            sig = a.get("signal_ajustado", "")
            sig_orig = a.get("signal_original", "")
            if sig != sig_orig:
                sig_display = f"{sig_orig}→{sig}"
            else:
                sig_display = sig
            print(f"   {a['ticker']:<8} {a['sensitivity']:>5.1f} {a['pos_original']:>10.1%} {a['pos_ajustada']:>10.1%}  {sig_display:>8}")

    if alert:
        print(f"\n   🚨 {alert}")

    print(f"\n   💡 {summary.get('descricao', '')}")
    print(f"{'='*70}\n")


# ── Standalone test hook ───────────────────────────────────────────────
def simulate_adjustment(score: float, mock_results: Optional[List[Dict]] = None):
    """
    Simula o ajuste com um score arbitrário para testar o comportamento.
    """
    from hormuz_adjuster import HormuzAdjuster

    if mock_results is None:
        mock_results = []
        for ticker, sens in OIL_SENSITIVITY.items():
            mock_results.append({
                "ticker": ticker,
                "position_size": 0.20,
                "signal": "BUY",
                "price": 0,
                "trend": "uptrend",
                "composite_score": 70,
            })
        # Add some non-oil tickers
        for ticker in ["VALE3", "ITUB4", "WEGE3", "BBAS3"]:
            mock_results.append({
                "ticker": ticker,
                "position_size": 0.20,
                "signal": "BUY",
                "price": 0,
                "trend": "uptrend",
                "composite_score": 70,
            })

    adjuster = HormuzAdjuster(score=score)
    result = adjuster.adjust(mock_results)
    print_adjuster_summary(result)
    return result


if __name__ == "__main__":
    import sys
    score = float(sys.argv[1]) if len(sys.argv) > 1 else 0.73
    simulate_adjustment(score)