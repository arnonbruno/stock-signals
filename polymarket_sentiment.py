#!/usr/bin/env python3
"""
Polymarket Sentiment Integration — módulo auxiliar para o stock signals.

Puxa mercados de prediction markets do Polymarket via API pública,
calcula scores de sentimento macro e oferece funções de consulta
para o pipeline principal de análise.

APIs usadas:
  Gamma API (https://gamma-api.polymarket.com) — eventos, mercados, tags
  CLOB API (https://clob.polymarket.com) — preços atuais (sem auth)
  Data API (https://data-api.polymarket.com) — trades, posições

Nenhuma autenticação necessária para leitura.

Uso:
    python3 polymarket_sentiment.py           # Diagnóstico + lista eventos
    python3 polymarket_sentiment.py --score   # Score agregado de sentimento
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("polymarket")

# ── Endpoints ───────────────────────────────────────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# Tags comprovadamente funcionais no Polymarket
# slug -> (peso, descrição, direção: high_is_bullish)
TAGS_RELEVANTES = {
    "interest-rates": (0.30, "Taxa de Juros / Política Monetária", False),
    "macro-indicators": (0.30, "Inflação / PIB / Fed Rate", False),
    "macro-single": (0.25, "Indicadores Macroeconômicos", True),
    # tags financeiras genéricas com volume alto
    # "finance", "economy" - muito genéricas, filtramos manualmente
}

# Cache
CACHE_FILE = Path(__file__).parent / "polymarket_cache.json"
CACHE_TTL = 60 * 10  # 10 minutos


class PolymarketClient:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self._cache = self._load_cache()

    # ── Cache ──────────────────────────────────────────────────────────
    def _load_cache(self) -> Dict:
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self):
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump(self._cache, f, indent=2)
        except OSError as e:
            logger.warning(f"⚠️ Cache IO: {e}")

    def _cache_get(self, key: str) -> Optional[any]:
        entry = self._cache.get(key)
        if not entry:
            return None
        age = datetime.now(timezone.utc).timestamp() - entry.get("ts", 0)
        if age > CACHE_TTL:
            return None
        return entry.get("data")

    def _cache_put(self, key: str, data: any):
        self._cache[key] = {
            "data": data,
            "ts": datetime.now(timezone.utc).timestamp(),
        }
        self._save_cache()

    # ── Gamma API ──────────────────────────────────────────────────────
    def get_events_by_tag(self, tag_slug: str, limit: int = 5) -> List[Dict]:
        """Busca eventos ativos por tag."""
        cache_key = f"events:{tag_slug}:{limit}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        try:
            resp = requests.get(
                f"{GAMMA_API}/events",
                params={"tag_slug": tag_slug, "closed": "false", "limit": limit},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            self._cache_put(cache_key, data)
            return data
        except Exception as e:
            logger.warning(f"⚠️ get_events({tag_slug}): {e}")
            return []

    def get_markets(self, event_id: str, limit: int = 5) -> List[Dict]:
        """Busca mercados de um evento específico."""
        cache_key = f"markets:{event_id}:{limit}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        try:
            resp = requests.get(
                f"{GAMMA_API}/markets",
                params={"event_id": event_id, "closed": "false", "limit": limit},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            self._cache_put(cache_key, data)
            return data
        except Exception as e:
            logger.warning(f"⚠️ get_markets({event_id}): {e}")
            return []

    # ── CLOB API ───────────────────────────────────────────────────────
    def get_last_price(self, market: Dict) -> Optional[float]:
        """
        Pega o último preço de um mercado.
        O preço já vem no response da Gamma API.
        """
        price = market.get("lastTradePrice")
        if price is not None:
            try:
                return float(price)
            except (ValueError, TypeError):
                pass
        # Fallback: outcomePrices
        outcomes = market.get("outcomePrices")
        if outcomes and len(outcomes) > 0:
            try:
                return float(outcomes[0])
            except (ValueError, TypeError):
                pass
        return None

    # ── Cálculo de sentimento ──────────────────────────────────────────
    def calculate_sentiment_score(self) -> Dict:
        """
        Calcula score de sentimento (0.0 a 1.0) baseado em mercados ativos.

        0.0 = pessimista / risco alto
        0.5 = neutro
        1.0 = otimista / risco baixo
        """
        agrupado = {}
        scores_com_peso: List[Tuple[float, float, str]] = []

        for tag_slug, (peso, desc, high_is_bullish) in TAGS_RELEVANTES.items():
            eventos = self.get_events_by_tag(tag_slug)
            if not eventos:
                continue

            prices = []
            for ev in eventos:
                ev_id = ev.get("id")
                if not ev_id:
                    continue
                markets = ev.get("markets") or self.get_markets(ev_id)
                for m in markets:
                    p = self.get_last_price(m)
                    if p is not None:
                        prices.append(p)

            if prices:
                avg = sum(prices) / len(prices)
                # Se high_is_bullish é False (ex: inflação alta é bearish), inverte
                score = avg if high_is_bullish else (1.0 - avg)
                scores_com_peso.append((score, peso, f"{tag_slug} ({desc})"))
                agrupado[tag_slug] = {
                    "descricao": desc,
                    "score_bruto": round(avg, 3),
                    "score_ajustado": round(score, 3),
                    "peso": peso,
                    "mercados_analisados": len(prices),
                    "high_is_bullish": high_is_bullish,
                }
                logger.info(
                    f"   {desc}: raw={avg:.3f} → adj={score:.3f} "
                    f"(peso {peso}, {len(prices)} mercados)"
                )

        if not scores_com_peso:
            return {"score": 0.5, "interpretacao": "neutro (sem dados)", "detalhes": {}}

        score_final = sum(s * p for s, p, _ in scores_com_peso) / sum(
            p for _, p, _ in scores_com_peso
        )
        score_final = max(0.0, min(1.0, score_final))

        if score_final >= 0.65:
            interp = "otimista 📈"
        elif score_final >= 0.45:
            interp = "neutro ➖"
        else:
            interp = "pessimista 📉"

        return {
            "score": round(score_final, 3),
            "interpretacao": interp,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "detalhes": agrupado,
        }

    # ── Diagnóstico ────────────────────────────────────────────────────
    def run_diagnostics(self):
        logger.info("\n🔌 POLYMARKET — Diagnóstico\n")

        # Testa APIs
        for nome, url in [("Gamma", f"{GAMMA_API}/events?limit=1"),
                          ("CLOB", f"{CLOB_API}/last-trade-price")]:
            try:
                r = requests.get(url, timeout=10)
                logger.info(f"✅ {nome} API: {r.status_code}" if r.ok
                           else f"⚠️  {nome} API: {r.status_code}")
            except Exception as e:
                logger.warning(f"❌ {nome} API: {e}")

        # Lista eventos por tag
        logger.info("\n📋 Eventos por tag:\n")
        for tag, (peso, desc, _) in TAGS_RELEVANTES.items():
            eventos = self.get_events_by_tag(tag)
            logger.info(f"   🏷️  {tag} ({desc}): {len(eventos)} eventos")
            for ev in eventos[:4]:
                t = ev.get("title", "?")
                vol = float(ev.get("volume", 0))
                logger.info(f"       • {t[:60]} (vol: R${vol/1e6:.1f}M)")
                markets = ev.get("markets") or []
                if markets:
                    for m in markets[:2]:
                        q = m.get("question", "?")
                        logger.info(f"         └ {q[:55]}")


def main():
    client = PolymarketClient()

    if "--score" in sys.argv:
        logger.info("\n📊 Score de Sentimento Polymarket\n")
        result = client.calculate_sentiment_score()
        logger.info(f"\n   Score agregado: {result['score']} — {result['interpretacao']}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        client.run_diagnostics()
        logger.info("\n💡 Use --score para sentimento agregado.")


if __name__ == "__main__":
    main()