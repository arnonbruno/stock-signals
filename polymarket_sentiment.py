#!/usr/bin/env python3
"""
Polymarket Sentiment Integration — módulo auxiliar para o stock signals.

Puxa mercados de prediction markets do Polymarket via API pública,
calcula scores de sentimento macro e score de risco geopolítico (Ormuz/petróleo).

APIs usadas:
  Gamma API (https://gamma-api.polymarket.com) — eventos, mercados, tags
  CLOB API (https://clob.polymarket.com) — preços atuais (sem auth)
  Data API (https://data-api.polymarket.com) — trades, posições

Nenhuma autenticação necessária para leitura.

Uso:
    python3 polymarket_sentiment.py           # Diagnóstico + lista eventos
    python3 polymarket_sentiment.py --score   # Score agregado de sentimento
    python3 polymarket_sentiment.py --hormuz  # Score específico de risco Ormuz
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
    "geopolitics": (0.20, "Geopolítica Geral (conflitos, guerras)", False),
}

# Mercados específicos do Estreito de Ormuz / disrupção de petróleo
# Cada entry: (slug, descrição, peso, probabilidade_sinaliza_disrupcao)
# peso = quanto este mercado importa no score agregado
# probabilidade_sinaliza_disrupcao = True se YES significa "disrupção ativa"
HORMUZ_MARKETS = [
    # —— Prazos curtos (peso maior, mais impacto imediato) ——
    ("strait-of-hormuz-traffic-returns-to-normal-by-may-15",
     "Tráfego Ormuz normaliza até 15/Mai", 0.20, False),
    ("strait-of-hormuz-traffic-returns-to-normal-by-end-of-may",
     "Tráfego Ormuz normaliza até 31/Mai", 0.20, False),
    # —— Prazos médios ——
    ("strait-of-hormuz-traffic-returns-to-normal-by-end-of-june",
     "Tráfego Ormuz normaliza até 30/Jun", 0.15, False),
    # —— Bloqueio EUA ——
    ("trump-announces-us-blockade-of-hormuz-lifted-by",
     "Trump anuncia fim do bloqueio em...", 0.15, False),
    # —— Kharg Island (terminal iraniano) ——
    ("kharg-island-no-longer-under-iranian-control-by-march-31",
     "Kharg Island fora do controle iraniano", 0.10, True),
    # —— Bab el-Mandeb ——
    ("bab-el-mandeb-strait-effectively-closed-by",
     "Estreito Bab el-Mandeb efetivamente fechado", 0.10, True),
    # —— Acordo Irã ——
    ("iran-agrees-to-unrestricted-shipping-through-hormuz-by-may-31",
     "Irã permite navegação irrestrita em Ormuz", 0.10, False),
]

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
    def get_event_by_slug(self, slug: str) -> Optional[Dict]:
        """Busca um evento específico pelo slug, incluindo markets."""
        cache_key = f"event:slug:{slug}"
        cached = self._cache_get(cache_key)
        if cached:
            return cached

        try:
            resp = requests.get(
                f"{GAMMA_API}/events",
                params={"slug": slug, "limit": 1},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            results = resp.json()
            if not results:
                return None

            ev = results[0]
            eid = ev.get("id")
            if eid:
                full = requests.get(
                    f"{GAMMA_API}/events/{eid}", timeout=self.timeout
                )
                if full.status_code == 200:
                    data = full.json()
                    self._cache_put(cache_key, data)
                    return data
            self._cache_put(cache_key, ev)
            return ev
        except Exception as e:
            logger.warning(f"⚠️ get_event_by_slug({slug}): {e}")
            return None

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

    # ── Price helpers ──────────────────────────────────────────────────
    def _parse_prices(self, market: Dict) -> Tuple[Optional[float], Optional[float]]:
        """
        Extrai os preços YES e NO de um mercado.
        outcomePrices vem como string JSON: '["0.085","0.915"]'
        """
        prices_raw = market.get("outcomePrices")
        if not prices_raw:
            return None, None
        try:
            prices = json.loads(prices_raw) if isinstance(prices_raw, str) else prices_raw
            yes = float(prices[0]) if len(prices) > 0 else None
            no = float(prices[1]) if len(prices) > 1 else None
            return yes, no
        except (ValueError, TypeError, json.JSONDecodeError):
            return None, None

    def get_last_price(self, market: Dict) -> Optional[float]:
        """
        Pega o último preço (YES) de um mercado.
        """
        price = market.get("lastTradePrice")
        if price is not None:
            try:
                return float(price)
            except (ValueError, TypeError):
                pass
        # Fallback: outcomePrices
        yes, _ = self._parse_prices(market)
        return yes

    # ── Hormuz / Oil Disruption Risk ───────────────────────────────────
    def calculate_hormuz_risk_score(self) -> Dict:
        """
        Calcula um score de disrupção no Estreito de Ormuz (0.0 a 1.0).

        0.0 = navegação normal, sem risco (bom pra petróleo → pressão baixista)
        0.5 = neutro
        1.0 = disrupção máxima (petróleo dispara)

        A lógica:
        - Mercados que medem "volta ao normal" (ex: tráfego normal): 
          YES alto = disrupção baixa → inverte (1-YES)
        - Mercados que medem "disrupção" (ex: Kharg sem controle iraniano):
          YES alto = disrupção alta → mantém
        """
        resultados = []
        scores_ponderados: List[Tuple[float, float, str]] = []

        for slug, desc, peso, disrupt_is_high in HORMUZ_MARKETS:
            ev = self.get_event_by_slug(slug)
            if not ev:
                continue

            markets = ev.get("markets", [])
            prices = []
            for m in markets:
                p = self.get_last_price(m)
                if p is not None:
                    prices.append(p)

            if not prices:
                continue

            # YES médio dos sub-mercados
            avg_yes = sum(prices) / len(prices)

            if disrupt_is_high:
                # YES alto = disrupção ativa
                disruption_score = avg_yes
            else:
                # YES alto = normalização (disrupção baixa)
                disruption_score = 1.0 - avg_yes

            scores_ponderados.append((disruption_score, peso, slug, desc, avg_yes, disrupt_is_high))
            resultados.append({
                "slug": slug,
                "descricao": desc,
                "prob_yes_media": round(avg_yes, 3),
                "score_disrupcao": round(disruption_score, 3),
                "peso": peso,
                "mercados": len(prices),
                "yes_significa_disrupcao": disrupt_is_high,
            })

        if not scores_ponderados:
            return {
                "score": 0.5,
                "interpretacao": "neutro (sem dados)",
                "mercados": [],
            }

        peso_total = sum(p for _, p, _, _, _, _ in scores_ponderados)
        score_final = sum(s * p for s, p, _, _, _, _ in scores_ponderados) / peso_total
        score_final = max(0.0, min(1.0, score_final))

        if score_final >= 0.65:
            interp = "disrupção ALTA ⛽📈 (petróleo sobe)"
        elif score_final >= 0.45:
            interp = "disrupção MODERADA ⚠️"
        elif score_final >= 0.25:
            interp = "risco BAIXO ✅"
        else:
            interp = "normalidade ✅✅ (petróleo sem risco geopolítico)"

        return {
            "score": round(score_final, 3),
            "interpretacao": interp,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mercados": resultados,
        }

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

        # Inclui score Ormuz no sentimento geral (peso 15% no agregado)
        hormuz = self.calculate_hormuz_risk_score()
        h_score = hormuz.get("score", 0.5)
        # Inverte: disrupção alta = bearish para bolsa
        h_adj = 1.0 - h_score
        scores_com_peso.append((h_adj, 0.15, "hormuz-ormuz (Risco Estreito de Ormuz)"))
        agrupado["hormuz-ormuz"] = {
            "descricao": "Risco Estreito de Ormuz (impacto petróleo)",
            "score_bruto": round(h_score, 3),
            "score_ajustado": round(h_adj, 3),
            "peso": 0.15,
            "sub_mercados": hormuz.get("mercados", []),
            "high_is_bullish": False,
        }

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
            "hormuz_score": hormuz,
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
                logger.info(f"       • {t[:60]} (vol: ${vol/1e6:.1f}M)")
                markets = ev.get("markets") or []
                if markets:
                    for m in markets[:2]:
                        q = m.get("question", "?")
                        logger.info(f"         └ {q[:55]}")

        # Lista mercados de Ormuz
        logger.info("\n🛢️  Mercados Estreito de Ormuz:\n")
        for slug, desc, peso, _ in HORMUZ_MARKETS:
            ev = self.get_event_by_slug(slug)
            if not ev:
                logger.info(f"   ⚠️  {slug}: não encontrado")
                continue
            title = ev.get("title", "?")[:60]
            vol = float(ev.get("volume", 0))
            markets = ev.get("markets", [])
            logger.info(f"   🌊 {title}")
            logger.info(f"      (vol: ${vol/1e6:.1f}M, {len(markets)} sub-mercados)")
            for m in markets[:3]:
                q = m.get("question", "?")[:55]
                yes_p, no_p = self._parse_prices(m)
                if yes_p is not None:
                    logger.info(f"         ├ YES: {yes_p*100:.1f}% | NO: {no_p*100:.1f}%")
                else:
                    logger.info(f"         ├ sem preço")
            if len(markets) > 3:
                logger.info(f"         └ ... +{len(markets)-3} sub-mercados")


def main():
    client = PolymarketClient()

    if "--hormuz" in sys.argv:
        logger.info("\n🛢️  Score de Risco — Estreito de Ormuz\n")
        result = client.calculate_hormuz_risk_score()
        logger.info(f"\n   Score: {result['score']} — {result['interpretacao']}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif "--score" in sys.argv:
        logger.info("\n📊 Score de Sentimento Polymarket\n")
        result = client.calculate_sentiment_score()
        logger.info(f"\n   Score agregado: {result['score']} — {result['interpretacao']}")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        client.run_diagnostics()
        logger.info("\n💡 Use --score para sentimento agregado ou --hormuz para risco Ormuz.")


if __name__ == "__main__":
    main()