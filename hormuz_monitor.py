#!/usr/bin/env python3
"""
Monitor de Correlação — Hormuz Score × PETR4 × WTI

Roda diariamente (ou on-demand) pra:
  1. Puxar score de disrupção Ormuz do Polymarket
  2. Puxar PETR4.SA e WTI (CL=F) do Yahoo
  3. Acumular histórico e calcular correlações

Uso:
    python3 hormuz_monitor.py          # Atualiza + mostra correlação
    python3 hormuz_monitor.py --stats  # Só mostra estatísticas (sem fetch)
    python3 hormuz_monitor.py --csv    # Exporta CSV
"""

import json
import math
import sys
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("hormuz_monitor")

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
HISTORY_FILE = BASE_DIR / "hormuz_correlacao_history.json"

# ── Sources ────────────────────────────────────────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com"
YAHOO_UA = "Mozilla/5.0"

# Mercados Ormuz monitorados (slug -> peso no score)
HORMUZ_MARKETS = [
    ("strait-of-hormuz-traffic-returns-to-normal-by-may-15", 0.20),
    ("strait-of-hormuz-traffic-returns-to-normal-by-end-of-may", 0.20),
    ("strait-of-hormuz-traffic-returns-to-normal-by-end-of-june", 0.15),
    ("trump-announces-us-blockade-of-hormuz-lifted-by", 0.15),
    ("kharg-island-no-longer-under-iranian-control-by-march-31", 0.10),
    ("bab-el-mandeb-strait-effectively-closed-by", 0.10),
    ("iran-agrees-to-unrestricted-shipping-through-hormuz-by-may-31", 0.10),
]


# ── Data Fetchers ──────────────────────────────────────────────────────
def fetch_hormuz_score() -> Tuple[Optional[float], Optional[Dict]]:
    """Retorna (score_ormuz, detalhes)"""
    try:
        scores_ponderados = []
        mercados_info = []

        for slug, peso in HORMUZ_MARKETS:
            r = requests.get(
                f"{GAMMA_API}/events", params={"slug": slug, "limit": 1}, timeout=15
            )
            results = r.json()
            if not results:
                continue

            ev = results[0]
            eid = ev.get("id")
            if not eid:
                continue

            r2 = requests.get(f"{GAMMA_API}/events/{eid}", timeout=15)
            ev_full = r2.json()
            markets = ev_full.get("markets", [])

            prices = []
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
                avg_yes = sum(prices) / len(prices)
                # Inverte: YES alto = normalização = disrupção baixa
                disruption = 1.0 - avg_yes
                scores_ponderados.append((disruption, peso))
                mercados_info.append({
                    "slug": slug,
                    "prob_normalizar": round(avg_yes, 3),
                    "score_disrupcao": round(disruption, 3),
                })

        if not scores_ponderados:
            return None, None

        peso_total = sum(p for _, p in scores_ponderados)
        score = sum(s * p for s, p in scores_ponderados) / peso_total
        score = max(0.0, min(1.0, score))

        detalhes = {
            "score": round(score, 3),
            "mercados": mercados_info,
        }
        return round(score, 3), detalhes
    except Exception as e:
        logger.warning(f"⚠️ fetch_hormuz_score: {e}")
        return None, None


def fetch_petr4() -> Optional[float]:
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/PETR4.SA",
            params={"range": "1d", "interval": "1d"},
            headers={"User-Agent": YAHOO_UA},
            timeout=15,
        )
        data = r.json()
        closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        return float(closes[-1])
    except Exception as e:
        logger.warning(f"⚠️ fetch_petr4: {e}")
        return None


def fetch_wti() -> Optional[float]:
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/CL=F",
            params={"range": "5d", "interval": "1d"},
            headers={"User-Agent": YAHOO_UA},
            timeout=15,
        )
        data = r.json()
        closes = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        # Last non-None close
        for c in reversed(closes):
            if c is not None:
                return float(c)
        return None
    except Exception as e:
        logger.warning(f"⚠️ fetch_wti: {e}")
        return None


# ── History ────────────────────────────────────────────────────────────
def load_history() -> List[Dict]:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []


def save_history(history: List[Dict]):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)


def update_history():
    """Faz fetch dos dados atuais e adiciona ao histórico."""
    history = load_history()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Check if already recorded today
    if any(h["date"] == today for h in history):
        logger.info(f"📅 Dados de {today} já registrados. Pulando.")
        return history

    logger.info("🔍 Buscando dados...")
    h_score, h_details = fetch_hormuz_score()
    petr4 = fetch_petr4()
    wti = fetch_wti()

    if h_score is None or petr4 is None or wti is None:
        logger.warning("⚠️ Falha ao buscar dados completos.")
        if h_score is None:
            logger.warning("   Hormuz: sem dados")
        if petr4 is None:
            logger.warning("   PETR4: sem dados")
        if wti is None:
            logger.warning("   WTI: sem dados")
        return history

    entry = {
        "date": today,
        "hormuz_score": h_score,
        "hormuz_detalhes": h_details,
        "petr4": round(petr4, 2),
        "wti": round(wti, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Calcula variações se houver dia anterior
    if history:
        prev = history[-1]
        if prev.get("petr4") and prev.get("wti") and prev.get("hormuz_score"):
            entry["petr4_var_pct"] = round((petr4 / prev["petr4"] - 1) * 100, 2)
            entry["wti_var_pct"] = round((wti / prev["wti"] - 1) * 100, 2)
            entry["hormuz_var"] = round(h_score - prev["hormuz_score"], 3)

    history.append(entry)
    save_history(history)
    logger.info(f"✅ Dados de {today} salvos")
    return history


# ── Statistics ─────────────────────────────────────────────────────────
def calc_stats(history: List[Dict]) -> Dict:
    if len(history) < 3:
        return {"error": "poucos dados para análise"}

    # Daily changes
    changes = []
    for i in range(1, len(history)):
        h = history[i]
        prev = history[i - 1]

        h_var = h.get("hormuz_score", 0) - prev.get("hormuz_score", 0)
        p4_var = h.get("petr4_var_pct", 0)
        wti_var = h.get("wti_var_pct", 0)

        # Also calculate from raw prices if var not available
        if p4_var == 0 and prev.get("petr4"):
            p4_var = (h["petr4"] / prev["petr4"] - 1) * 100
        if wti_var == 0 and prev.get("wti"):
            wti_var = (h["wti"] / prev["wti"] - 1) * 100

        changes.append({
            "date": h["date"],
            "hormuz_var": h_var,
            "petr4_var": p4_var,
            "wti_var": wti_var,
        })

    if len(changes) < 3:
        return {"error": "poucas variações diárias para correlação"}

    n = len(changes)

    # Hormuz × PETR4
    h_vars = [c["hormuz_var"] for c in changes]
    p_vars = [c["petr4_var"] for c in changes]
    w_vars = [c["wti_var"] for c in changes]

    def correlation(x, y):
        mx = sum(x) / n
        my = sum(y) / n
        cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / n
        sx = math.sqrt(sum((xi - mx) ** 2 for xi in x) / n) if n > 0 else 1
        sy = math.sqrt(sum((yi - my) ** 2 for yi in y) / n) if n > 0 else 1
        if sx == 0 or sy == 0:
            return 0
        return cov / (sx * sy)

    corr_h_p4 = correlation(h_vars, p_vars)
    corr_h_wti = correlation(h_vars, w_vars)
    corr_p4_wti = correlation(p_vars, w_vars)

    # Current state
    latest = history[-1]
    first = history[0]

    return {
        "amostras": n,
        "periodo": f"{first['date']} → {latest['date']}",
        "correlacoes": {
            "hormuz_petr4": round(corr_h_p4, 3),
            "hormuz_wti": round(corr_h_wti, 3),
            "petr4_wti": round(corr_p4_wti, 3),
        },
        "valores_atuais": {
            "hormuz_score": latest.get("hormuz_score"),
            "petr4": latest.get("petr4"),
            "wti": latest.get("wti"),
        },
        "acumulado": {
            "petr4": round((latest["petr4"] / first["petr4"] - 1) * 100, 1),
            "wti": round((latest["wti"] / first["wti"] - 1) * 100, 1),
        },
    }


# ── Output ─────────────────────────────────────────────────────────────
def print_summary(history: List[Dict], stats: Dict):
    print(f"\n{'='*55}")
    print(f"  🛢️  MONITOR ORMUZ × PETR4 × WTI")
    print(f"{'='*55}")

    if stats and "error" not in stats:
        print(f"\n  📊 Correlações (variação diária):")
        print(f"     Hormuz × PETR4:  {stats['correlacoes']['hormuz_petr4']:+.3f}")
        print(f"     Hormuz × WTI:    {stats['correlacoes']['hormuz_wti']:+.3f}")
        print(f"     PETR4 × WTI:     {stats['correlacoes']['petr4_wti']:+.3f}")
        print(f"\n  📈 Acumulado ({stats['periodo']}):")
        print(f"     PETR4: {stats['acumulado']['petr4']:+.1f}%")
        print(f"     WTI:   {stats['acumulado']['wti']:+.1f}%")

    if history:
        print(f"\n  📅 Últimos registros:")
        print(f"  {'Data':<12} {'Hormuz':>7} {'PETR4':>8} {'WTI':>7} {'P4Δ%':>7} {'WTIΔ%':>7} {'HΔ':>6}")
        print(f"  {'-'*12} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
        for h in history[-7:]:
            hp = h.get("petr4_var_pct", 0) or 0
            wp = h.get("wti_var_pct", 0) or 0
            hv = h.get("hormuz_var", 0) or 0
            print(f"  {h['date']:<12} {h.get('hormuz_score',0):>7.3f} "
                  f"{h.get('petr4',0):>8.2f} {h.get('wti',0):>7.2f} "
                  f"{hp:>+6.2f}% {wp:>+6.2f}% {hv:>+5.3f}")

    if history:
        latest = history[-1]
        hs = latest.get("hormuz_score", 0)
        if hs >= 0.65:
            print(f"\n  🚨 SITUAÇÃO: Disrupção ALTA ⛽ (score {hs:.3f})")
        elif hs >= 0.45:
            print(f"\n  ⚠️  SITUAÇÃO: Risco moderado (score {hs:.3f})")
        else:
            print(f"\n  ✅ SITUAÇÃO: Normalidade (score {hs:.3f})")

    print(f"\n{'='*55}\n")


def export_csv(history: List[Dict], path: str = None):
    if not history:
        print("Sem dados para exportar")
        return
    path = path or str(BASE_DIR / "hormuz_correlacao.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Data", "HormuzScore", "PETR4", "WTI", "PETR4_Var%", "WTI_Var%", "Hormuz_Var"])
        for h in history:
            w.writerow([
                h["date"],
                h.get("hormuz_score", ""),
                h.get("petr4", ""),
                h.get("wti", ""),
                h.get("petr4_var_pct", ""),
                h.get("wti_var_pct", ""),
                h.get("hormuz_var", ""),
            ])
    print(f"📁 CSV exportado: {path}")


# ── Main ───────────────────────────────────────────────────────────────
def main():
    args = set(sys.argv[1:])

    if "--stats" in args or "--csv" in args:
        history = load_history()
    else:
        history = update_history()

    stats = calc_stats(history)

    if "--csv" in args:
        export_csv(history)
    else:
        print_summary(history, stats)


if __name__ == "__main__":
    main()