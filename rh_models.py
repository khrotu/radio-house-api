from __future__ import annotations
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import requests
SITE_ORIGIN = "https://www.tryingopen.com"
SITE_URL = SITE_ORIGIN + "/"
CATALOG_TTL_S = 300.0
FAILURE_RETRY_S = 60.0
HTTP_TIMEOUT = 30
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36"
ROUTE_COOLDOWN_S = 90.0
ROUTE_FAILURE_LIMIT = 3
PROVIDER = "tryingopen"
COMPARE_PROVIDER = "tryingopen-compare"
COMPARE_MODELS: Tuple[Dict[str, Any], ...] = (
    {
        "id": "openai/gpt-5.6-terra",
        "name": "GPT-5.6 Terra",
        "context_window": 1050000,
        "features": ("tool-use",),
        "price_per_mtok": 2.0,
        "provider": COMPARE_PROVIDER,
    },
    {
        "id": "anthropic/claude-sonnet-5",
        "name": "Claude Sonnet 5",
        "context_window": 1000000,
        "features": ("tool-use",),
        "price_per_mtok": 2.0,
        "provider": COMPARE_PROVIDER,
    },
)
_CHUNK_RE = re.compile(r"/_next/static/chunks/[A-Za-z0-9._~\-]+\.js(?:\?[^\s\"']*)?")
_RECORD_HEAD = re.compile(r"\{id:\"([a-z0-9][a-z0-9.\-]*/[a-z0-9][a-z0-9.\-]*)\",name:\"([^\"]+)\"")
_CONTEXT_RE = re.compile(r"\bcontext:\"([^\"]+)\"")
_PRICE_RE = re.compile(r"\bpricePerMTok:([0-9.]+)")
def _slugify(pid: str) -> str:
    s = pid.strip().lower()
    s = re.sub(r"[@+_/]", "-", s)
    s = re.sub(r"[^a-z0-9\-.]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    s = re.sub(r"\.{2,}", ".", s)
    s = s.strip("-.")
    return s or "model"
def _standardize_model_id(model_id: str) -> str:
    s = model_id.strip().lower()
    s = re.sub(r"^(qwen)(\d+)-(\d+)(?=-|$)", r"\1\2.\3", s)
    s = re.sub(r"^(glm)-(\d+)-(\d+)(?=-|$|\.)", r"\1-\2.\3", s)
    s = re.sub(r"^(nemotron)-(\d+)-(\d+)(?=-|$)", r"\1-\2.\3", s)
    s = re.sub(r"-(\d+)-(\d+)(t)(?=-|$)", r"-\1.\2\3", s)
    if re.search(r"-(\d)-(\d)(?=-[a-z])", s):
        s = re.sub(r"-(\d)-(\d)(?=-[a-z])", r"-\1.\2", s, count=1)
    return s
def _parse_context(label: str) -> int:
    m = re.match(r"([0-9.]+)\s*([kKmM]?)", label.strip())
    if not m:
        return 0
    value = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "k":
        return int(value * 1024)
    if unit == "m":
        return int(value * 1024 * 1024)
    return int(value)
@dataclass(frozen=True)
class Route:
    provider: str
    raw_id: str
    mid: str
    pid: str
    slug: str
    context_window: int = 0
    max_output_tokens: int = 0
    features: Tuple[str, ...] = ()
    input_modalities: Tuple[str, ...] = ()
    output_modalities: Tuple[str, ...] = ()
    supported_parameters: Tuple[str, ...] = ()
    label: str = ""
    aliases: Tuple[str, ...] = ()
@dataclass(frozen=True)
class ModelCard:
    id: str
    label: str
    routes: Tuple[Route, ...]
    context_window: int = 0
    owned_by: str = "radio-house"
    aliases: Tuple[str, ...] = ()
@dataclass
class RouteHealth:
    failures: int = 0
    cooldown_until: float = 0.0
    def available(self, now: float) -> bool:
        return now >= self.cooldown_until
    def report(self, ok: bool, hard: bool = False) -> None:
        if ok:
            self.failures = 0
            self.cooldown_until = 0.0
            return
        self.failures += 1
        if hard or self.failures >= ROUTE_FAILURE_LIMIT:
            self.cooldown_until = time.time() + ROUTE_COOLDOWN_S
            self.failures = 0
def _fetch_text(url: str) -> str:
    resp = requests.get(url, timeout=HTTP_TIMEOUT, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    resp.raise_for_status()
    return resp.text
def _parse_catalog(js: str) -> List[Dict[str, Any]]:
    heads = list(_RECORD_HEAD.finditer(js))
    records: List[Dict[str, Any]] = []
    seen: set = set()
    for i, m in enumerate(heads):
        raw_id, name = m.group(1), m.group(2)
        if raw_id in seen:
            continue
        seg_end = heads[i + 1].start() if i + 1 < len(heads) else min(len(js), m.end() + 4000)
        seg = js[m.start():seg_end]
        cm = _CONTEXT_RE.search(seg)
        if not cm:
            continue
        features = ["tool-use"] if "supportsTools:!0" in seg else []
        if "supportsImages:!0" in seg:
            features.append("images")
        pm = _PRICE_RE.search(seg)
        records.append({
            "id": raw_id,
            "name": name,
            "context_window": _parse_context(cm.group(1)),
            "features": tuple(features),
            "price_per_mtok": float(pm.group(1)) if pm else None,
        })
        seen.add(raw_id)
    return records
def _fetch_catalog() -> List[Dict[str, Any]]:
    html = _fetch_text(SITE_URL)
    urls: List[str] = []
    seen: set = set()
    for u in _CHUNK_RE.findall(html):
        if u not in seen:
            seen.add(u)
            urls.append(u)
    for path in urls:
        try:
            js = _fetch_text(SITE_ORIGIN + path)
        except Exception:
            continue
        if "supportsTools" not in js:
            continue
        records = _parse_catalog(js)
        if records:
            return sorted(records, key=lambda r: r["price_per_mtok"] if r["price_per_mtok"] is not None else 9999.0)
    return []
def _build_cards(rows: List[Dict[str, Any]]) -> Tuple[List[ModelCard], Dict[str, ModelCard]]:
    known = {row["id"] for row in rows}
    rows = list(rows) + [dict(r) for r in COMPARE_MODELS if r["id"] not in known]
    cards: List[ModelCard] = []
    by_key: Dict[str, ModelCard] = {}
    used_ids: Dict[str, int] = {}
    for row in rows:
        raw_id = row["id"]
        suffix = raw_id.split("/", 1)[1] if "/" in raw_id else raw_id
        base = _slugify(suffix)
        n = used_ids.get(base, 0) + 1
        used_ids[base] = n
        card_id = base if n == 1 else f"{base}-{n}"
        name = row["name"]
        base_aliases = tuple(dict.fromkeys(a for a in (raw_id, suffix, name, _slugify(name)) if a))
        variant_aliases: List[str] = []
        for a in base_aliases:
            low = a.lower()
            if "." in low:
                hv = low.replace(".", "-")
                if hv != low and hv not in variant_aliases:
                    variant_aliases.append(hv)
            std = _standardize_model_id(low)
            if std != low and std not in variant_aliases:
                variant_aliases.append(std)
            std_hv = std.replace(".", "-") if "." in std else std
            if std_hv != low and std_hv != std and std_hv not in variant_aliases:
                variant_aliases.append(std_hv)
        if "." in card_id:
            hv = card_id.replace(".", "-")
            if hv not in variant_aliases and hv.lower() != card_id.lower():
                variant_aliases.append(hv)
        aliases = tuple(dict.fromkeys(list(base_aliases) + variant_aliases))
        provider = row.get("provider") or PROVIDER
        route = Route(
            provider=provider,
            raw_id=raw_id,
            mid=raw_id,
            pid=name,
            slug=card_id,
            context_window=row["context_window"],
            features=row["features"],
            label=f"{name} @ {provider}",
            aliases=aliases,
        )
        card = ModelCard(
            id=card_id,
            label=name,
            routes=(route,),
            context_window=row["context_window"],
            aliases=aliases,
        )
        cards.append(card)
        by_key[card_id.lower()] = card
        for a in aliases:
            by_key[a.lower()] = card
        std_card = _standardize_model_id(card_id)
        if std_card.lower() != card_id.lower():
            by_key[std_card.lower()] = card
        hv_card = card_id.replace(".", "-")
        if hv_card.lower() != card_id.lower():
            by_key[hv_card.lower()] = card
        by_key[_slugify(card_id).lower()] = card
        if hv_card != card_id:
            by_key[_slugify(hv_card).lower()] = card
    return cards, by_key
class ModelRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cards: List[ModelCard] = []
        self._by_key: Dict[str, ModelCard] = {}
        self._route_health: Dict[str, RouteHealth] = {}
        self._fetched_at = 0.0
        self._retry_after = 0.0
        self._refreshing = False
    def refresh(self, force: bool = True) -> int:
        if force:
            with self._lock:
                self._retry_after = 0.0
        rows = _fetch_catalog()
        if rows:
            cards, by_key = _build_cards(rows)
            with self._lock:
                self._cards = cards
                self._by_key = by_key
                self._fetched_at = time.time()
        return len(self.cards)
    def _attempt_refresh(self) -> None:
        now = time.time()
        with self._lock:
            if self._refreshing or now < self._retry_after:
                return
            self._refreshing = True
        try:
            self.refresh(force=False)
        finally:
            with self._lock:
                self._refreshing = False
    def _ensure(self) -> None:
        with self._lock:
            has = bool(self._cards)
            stale = time.time() - self._fetched_at >= CATALOG_TTL_S
        if has and not stale:
            return
        if has:
            threading.Thread(target=self._attempt_refresh, daemon=True).start()
        else:
            self._attempt_refresh()
    @property
    def cards(self) -> List[ModelCard]:
        self._ensure()
        with self._lock:
            return list(self._cards)
    def default(self) -> Optional[ModelCard]:
        cards = self.cards
        if not cards:
            return None
        for preferred in ("qwen3.8-27b", "deepseek-v4-flash-0731", "glm-5.3", "kimi-k3"):
            card = self.resolve(preferred, strict=True)
            if card:
                return card
        return cards[0]
    def resolve(self, model_id: Optional[str], strict: bool = False) -> Optional[ModelCard]:
        if not model_id:
            return None if strict else self.default()
        key = model_id.strip().lower()
        if key in {"", "auto", "default", "router/default"}:
            return None if strict else self.default()
        self._ensure()
        std = _standardize_model_id(key)
        hv = key.replace(".", "-") if "." in key else std.replace(".", "-") if "." in std else None
        norm = _slugify(key)
        std_norm = _standardize_model_id(norm)
        slug_std = _slugify(std) if std != key else None
        candidates: List[str] = []
        for cand in (key, std, hv, norm, std_norm, slug_std):
            if cand and cand not in candidates:
                candidates.append(cand)
                if "." in cand:
                    hv_c = cand.replace(".", "-")
                    if hv_c not in candidates:
                        candidates.append(hv_c)
                else:
                    dot_c = _standardize_model_id(cand)
                    if dot_c != cand and dot_c not in candidates:
                        candidates.append(dot_c)
        with self._lock:
            for cand in candidates:
                card = self._by_key.get(cand)
                if card:
                    return card
            for card in self._cards:
                for cand in candidates:
                    if card.id == cand:
                        return card
                    if _slugify(card.label) == cand:
                        return card
                    for a in card.aliases:
                        if _slugify(a) == cand or a.lower() == cand:
                            return card
                        if _standardize_model_id(a.lower()) == cand:
                            return card
        return None
    def route_health(self, mid: str) -> RouteHealth:
        with self._lock:
            health = self._route_health.get(mid)
            if health is None:
                health = RouteHealth()
                self._route_health[mid] = health
            return health
    def report_route(self, mid: str, ok: bool, hard: bool = False) -> None:
        self.route_health(mid).report(ok, hard=hard)
    def ordered_routes(self, card: ModelCard, prefer_tools: bool = False) -> List[Route]:
        now = time.time()
        healthy = [r for r in card.routes if self.route_health(r.mid).available(now)]
        if not healthy:
            healthy = list(card.routes)
        if prefer_tools:
            healthy.sort(key=lambda r: (0 if "tool-use" in r.features else 1, r.raw_id))
        return healthy
REGISTRY = ModelRegistry()
