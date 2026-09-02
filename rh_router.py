from __future__ import annotations
import codecs
import itertools
import json
import random as _random
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import httpx
import requests
from rh_credentials import (
    Credential,
    CredentialFile,
    DEFAULT_OUTPUT,
    DEFAULT_USER_AGENT,
    ensure_fresh,
)
from rh_models import COMPARE_PROVIDER, ModelCard, ModelRegistry, Route, REGISTRY
CHAT_ENDPOINT = "https://www.tryingopen.com/api/open"
COMPARE_ENDPOINT = "https://www.tryingopen.com/api/compare"
SITE_ORIGIN = "https://www.tryingopen.com"
EFFORT_MAP = {"low": "quick", "minimal": "quick", "quick": "quick", "medium": "balanced", "balanced": "balanced", "high": "deep", "xhigh": "deep", "max": "deep", "deep": "deep"}
DEFAULT_EFFORT = "balanced"
MAX_FAILURES = 3
COOLDOWN_BASE = 15.0
COOLDOWN_MAX = 300.0
HTTP_TIMEOUT = (10, 120)
MAX_CRED_ATTEMPTS = 3
RETRY_BASE_DELAY = 1.0
RETRY_MAX_DELAY = 20.0
MAX_MESSAGES = 40
MESSAGE_TRIM_TARGET = 30
PROMPT_TOKEN_CAP = 30000
MAX_STREAM_EVENTS = 4000
REPEAT_STREAK_LIMIT = 12
SITE_OUTAGE_RETRY_DELAY = 8.0
class _StreamError(RuntimeError):
    pass
class _RouteRejected(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status
_HTTP2_CLIENT: Optional[httpx.Client] = None
_HTTP2_CLIENT_LOCK = threading.Lock()
def _http2_client() -> httpx.Client:
    global _HTTP2_CLIENT
    with _HTTP2_CLIENT_LOCK:
        if _HTTP2_CLIENT is None:
            _HTTP2_CLIENT = httpx.Client(http2=True, timeout=httpx.Timeout(10.0, read=300.0))
        return _HTTP2_CLIENT
def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: List[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text", "")))
            elif isinstance(part, str):
                parts.append(part)
    return "".join(parts)
_MIME_BY_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp", ".pdf": "application/pdf"}
def _guess_media_type(url: str) -> str:
    path = url.split("?", 1)[0].lower()
    for ext, mime in _MIME_BY_EXT.items():
        if path.endswith(ext):
            return mime
    return "image/png"
def _part_text_len(msg: Dict[str, Any]) -> int:
    return sum(len(p.get("text", "")) for p in msg["parts"] if p["type"] == "text")
def _truncate_parts(msg: Dict[str, Any], limit: int) -> Dict[str, Any]:
    new_parts = []
    for p in msg["parts"]:
        if p["type"] == "text" and len(p.get("text", "")) > limit:
            p = {**p, "text": p["text"][:limit]}
        new_parts.append(p)
    return {**msg, "parts": new_parts}
def _fold_systems(out: List[Dict[str, Any]], system_texts: List[str]) -> List[Dict[str, Any]]:
    if not system_texts:
        return out
    preamble = "\n\n".join(t.strip() for t in system_texts if t.strip())
    if not preamble:
        return out
    wrapped = f"[SYSTEM INSTRUCTIONS]\n{preamble}\n[/SYSTEM INSTRUCTIONS]"
    for msg in out:
        if msg["role"] == "user":
            msg["parts"] = [{"type": "text", "text": wrapped}] + msg["parts"]
            return out
    return [{"id": "sys-0", "role": "user", "parts": [{"type": "text", "text": wrapped}]}] + out
def to_ui_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    system_texts: List[str] = []
    for i, msg in enumerate(messages):
        role = msg.get("role") or "user"
        content = msg.get("content")
        if role == "system":
            text = _flatten_content(content)
            if text:
                system_texts.append(text)
            continue
        parts: List[Dict[str, Any]] = []
        if role == "assistant" and msg.get("tool_calls"):
            text = _flatten_content(content)
            calls = []
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                calls.append(f"{fn.get('name', 'tool')}({fn.get('arguments', '')})")
            combined = (text + ("\n" if text else "") + "[called " + "; ".join(calls) + "]").strip()
            if combined:
                parts.append({"type": "text", "text": combined})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    text = str(part.get("text", ""))
                    if text:
                        parts.append({"type": "text", "text": text})
                elif ptype == "image_url":
                    iu = part.get("image_url")
                    url = iu.get("url") if isinstance(iu, dict) else iu
                    if not url:
                        continue
                    if str(url).startswith("data:"):
                        head, _, b64 = str(url).partition(",")
                        media_type = head[5:].split(";", 1)[0] or "image/png"
                        parts.append({"type": "file", "mediaType": media_type, "url": f"data:{media_type};base64,{b64}"})
                    else:
                        parts.append({"type": "file", "mediaType": _guess_media_type(str(url)), "url": str(url)})
        else:
            text = _flatten_content(content)
            if text:
                parts.append({"type": "text", "text": text})
        if parts:
            out.append({"id": msg.get("id") or f"msg-{i}-{uuid.uuid4().hex[:8]}", "role": role, "parts": parts})
    return _fold_systems(out, system_texts)
def _effort_for(reasoning_effort: Optional[str]) -> str:
    if not reasoning_effort:
        return DEFAULT_EFFORT
    return EFFORT_MAP.get(str(reasoning_effort).strip().lower(), DEFAULT_EFFORT)
def _tool_spec_text(tools: List[Any], tool_choice: Optional[Any]) -> str:
    full = json.dumps(tools, ensure_ascii=False)
    if len(full) <= 6000:
        spec = full
    else:
        lines: List[str] = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function") if isinstance(t.get("function"), dict) else {}
            name = fn.get("name") or t.get("name") or "tool"
            desc = str(fn.get("description") or "")[:200]
            params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
            required = params.get("required") or []
            if not isinstance(required, list):
                required = []
            lines.append(f"{name}({', '.join(str(r) for r in required)}) - {desc}")
        spec = "\n".join(lines)[:6000]
    text = f"Available tools:\n{spec}"
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        if name:
            text += f"\nYou MUST call the tool named \"{name}\"."
    elif tool_choice == "required":
        text += "\nYou MUST call one or more tools."
    return text
def _emulation_messages(messages: List[Dict[str, Any]], tools: List[Any], tool_choice: Optional[Any], followup: bool = False) -> List[Dict[str, Any]]:
    if followup:
        instruction = (
            "\n\nYour previous reply did not include a tool call, but the task requires it.\n"
            "If a tool is needed, output ONLY this JSON object and nothing else:\n"
            '{"tool_call": {"name": "<exact tool name>", "arguments": {<arguments>}}}\n'
            "Do not narrate, do not explain, do not use markdown fences."
        )
    else:
        instruction = (
            "\n\n[TOOL CALLING MODE]\n"
            + _tool_spec_text(tools, tool_choice)
            + "\n\nIf you need to call a tool, respond with ONLY a single JSON object and no other text, no markdown fences:\n"
            '{"tool_call": {"name": "<exact tool name>", "arguments": {<arguments matching the tool schema>}}}\n\n'
            "If no tool call is needed, answer normally as plain text."
        )
    out = [dict(m) for m in messages]
    out.append({"role": "system", "content": instruction})
    return out
def _looks_like_tool_call(obj: Any) -> bool:
    items = obj if isinstance(obj, list) else [obj]
    for item in items:
        if not isinstance(item, dict):
            continue
        inner = item.get("tool_call") if isinstance(item.get("tool_call"), dict) else item
        if isinstance(inner.get("name") or inner.get("tool") or inner.get("tool_name"), str):
            return True
    return False
def _last_json_object(text: str) -> Tuple[Optional[Any], Optional[int]]:
    if not text:
        return None, None
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.startswith("json"):
            t = t[4:]
        t = t.strip()
    starts = [m.start() for m in re.finditer(r"\{", t)]
    decoder = json.JSONDecoder()
    best: Optional[Tuple[Any, int]] = None
    for idx in starts:
        try:
            obj, _ = decoder.raw_decode(t, idx)
        except Exception:
            continue
        if isinstance(obj, (dict, list)):
            if _looks_like_tool_call(obj):
                return obj, idx
            if best is None:
                best = (obj, idx)
    return best if best is not None else (None, None)
def _json_position(text: str) -> Optional[int]:
    _, idx = _last_json_object(text)
    return idx
def _parse_plaintext_tool_calls(text: str) -> Optional[List[Dict[str, Any]]]:
    found, _ = _last_json_object(text)
    if found is None:
        return None
    items = found if isinstance(found, list) else [found]
    calls: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        inner = item.get("tool_call") if isinstance(item.get("tool_call"), dict) else item
        name = inner.get("name") or inner.get("tool") or inner.get("tool_name")
        args = inner.get("arguments")
        if args is None:
            args = inner.get("args")
        if args is None:
            args = inner.get("parameters")
        if not isinstance(name, str) or not name:
            continue
        if isinstance(args, dict):
            args = json.dumps(args, ensure_ascii=False)
        if not isinstance(args, str):
            args = "{}"
        calls.append({"name": name, "arguments": args})
    return calls or None
def _plain_compare_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for msg in messages:
        parts = msg.get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("type") == "text")
        if text:
            out.append({"role": msg.get("role") or "user", "content": text})
    return out

class TryingOpenClient:
    def __init__(self, credential: Credential, timeout: Tuple[int, int] = HTTP_TIMEOUT) -> None:
        self.credential = credential
        self.timeout = timeout
        self.session = requests.Session()
    def _headers(self, accept: str = "text/event-stream") -> Dict[str, str]:
        ensure_fresh(self.credential)
        headers = {
            "Accept": accept,
            "Content-Type": "application/json",
            "Origin": SITE_ORIGIN,
            "Referer": SITE_ORIGIN + "/",
            "User-Agent": self.credential.user_agent or DEFAULT_USER_AGENT,
        }
        if self.credential.cookie_header:
            headers["Cookie"] = self.credential.cookie_header
        return headers
    @staticmethod
    def _build_payload(messages: List[Dict[str, Any]], route: Route, effort: str) -> Dict[str, Any]:
        now_hex = uuid.uuid4().hex
        return {
            "id": f"chat-{now_hex[:16]}",
            "trigger": "submit-message",
            "messageId": f"msg-{uuid.uuid4().hex[:24]}",
            "model": route.raw_id,
            "effort": effort,
            "messages": messages,
        }
    def stream(self, messages: List[Dict[str, Any]], route: Route, effort: str = DEFAULT_EFFORT) -> Generator[Dict[str, Any], None, None]:
        payload = self._build_payload(messages, route, effort)
        with self.session.post(
            CHAT_ENDPOINT, headers=self._headers(), json=payload,
            stream=True, timeout=self.timeout,
        ) as resp:
            resp.encoding = "utf-8"
            if resp.status_code != 200:
                self._raise_for_rejection(resp)
            event_count = 0
            repeat_streak = 0
            repeat_text = None
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                stripped = line.strip()
                if stripped.startswith(":"):
                    continue
                if stripped.startswith("data:"):
                    stripped = stripped[5:].strip()
                if not stripped:
                    continue
                if stripped == "[DONE]":
                    yield {"type": "done"}
                    return
                if not stripped.startswith("{"):
                    continue
                try:
                    obj = json.loads(stripped)
                except Exception:
                    continue
                event_count += 1
                if event_count > MAX_STREAM_EVENTS:
                    raise _RouteRejected("upstream stream exceeded max chunks (possible loop)")
                for ev in self._events_from_obj(obj):
                    if ev.get("type") == "content":
                        text = ev.get("text", "")
                        if text and text == repeat_text:
                            repeat_streak += 1
                            if repeat_streak >= REPEAT_STREAK_LIMIT:
                                raise _RouteRejected("upstream repetition loop (model stuck)")
                        else:
                            repeat_text = text if text else repeat_text
                            repeat_streak = 1
                    yield ev
    def stream_compare(self, messages: List[Dict[str, Any]], route: Route) -> Generator[Dict[str, Any], None, None]:
        payload: Dict[str, Any] = {
            "model": route.raw_id,
            "messages": _plain_compare_messages(messages),
        }
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            with _http2_client().stream(
                "POST", COMPARE_ENDPOINT, headers=self._headers(accept="*/*"), json=payload,
            ) as resp:
                if resp.status_code != 200:
                    body = resp.read().decode("utf-8", errors="replace")
                    self._raise_compare_rejection(resp.status_code, body, resp.headers.get("Retry-After"))
                event_count = 0
                repeat_streak = 0
                repeat_text = None
                for chunk in resp.iter_raw():
                    if not chunk:
                        continue
                    text = decoder.decode(chunk)
                    if not text:
                        continue
                    event_count += 1
                    if event_count > MAX_STREAM_EVENTS:
                        raise _RouteRejected("upstream stream exceeded max chunks (possible loop)")
                    if text == repeat_text:
                        repeat_streak += 1
                        if repeat_streak >= REPEAT_STREAK_LIMIT:
                            raise _RouteRejected("upstream repetition loop (model stuck)")
                    else:
                        repeat_text = text
                        repeat_streak = 1
                    yield {"type": "content", "text": text}
        except httpx.HTTPError as exc:
            raise requests.HTTPError(f"compare transport error: {exc}") from exc
    def _raise_compare_rejection(self, status: int, body: str, retry_after: Optional[str] = None) -> None:
        message = body[:300]
        try:
            message = str(json.loads(body).get("error") or message)
        except Exception:
            pass
        if status == 429:
            suffix = f" (retry after {retry_after}s)" if retry_after else ""
            raise _RouteRejected(f"compare quota exhausted: {message}{suffix}", status=429)
        if 500 <= status < 600:
            raise requests.HTTPError(f"upstream {status}: {message}")
        raise _RouteRejected(f"status {status}: {message}", status=status)
    @staticmethod
    def _raise_for_rejection(resp: requests.Response) -> None:
        status = resp.status_code
        try:
            body = resp.text or ""
        except Exception:
            body = ""
        if status == 429:
            raise requests.HTTPError(f"rate limited: {body[:200]}", response=resp)
        if 500 <= status < 600:
            raise requests.HTTPError(f"upstream {status}: {body[:200]}", response=resp)
        raise _RouteRejected(f"status {status}: {body[:300]}", status=status)
    @staticmethod
    def _events_from_obj(obj: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
        etype = obj.get("type")
        if etype == "error":
            msg = obj.get("errorText") or obj.get("message") or json.dumps(obj)[:300]
            yield {"type": "error", "message": str(msg)}
            return
        if etype == "reasoning-delta":
            delta = obj.get("delta")
            if isinstance(delta, str) and delta:
                yield {"type": "reasoning", "text": delta}
            return
        if etype == "text-delta":
            delta = obj.get("delta")
            if isinstance(delta, str) and delta:
                yield {"type": "content", "text": delta}
            return
        if etype == "finish":
            meta = obj.get("messageMetadata") or {}
            usage: Dict[str, Any] = {}
            for src, dst in (("inputTokens", "prompt_tokens"), ("outputTokens", "completion_tokens"), ("totalTokens", "total_tokens"), ("reasoningTokens", "reasoning_tokens")):
                if isinstance(meta.get(src), int):
                    usage[dst] = meta[src]
            if usage:
                yield {"type": "usage", "usage": usage}
            finish = obj.get("finishReason")
            if finish and str(finish).lower() != "error":
                yield {"type": "finish", "finish_reason": str(finish)}
            elif finish:
                yield {"type": "error", "message": "upstream stream ended with finish_reason=error"}
@dataclass
class CredentialState:
    credential: Credential
    failures: int = 0
    successes: int = 0
    depleted: bool = False
    cooldown_until: float = 0.0
    last_used: float = 0.0
    last_error: Optional[str] = None
    def available(self, now: float) -> bool:
        return (not self.depleted) and now >= self.cooldown_until
class NoCredentialsError(RuntimeError):
    pass
class AllCredentialsBusyError(RuntimeError):
    pass
class CredentialPool:
    def __init__(self, path: Path = DEFAULT_OUTPUT) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._states: List[CredentialState] = []
        self._cycle = itertools.cycle([])
        self.reload()
        if not self._states:
            self._states = [CredentialState(credential=Credential(id="anon-default"))]
            self._rebuild_cycle()
    def reload(self) -> int:
        creds = CredentialFile(self.path).load()
        with self._lock:
            existing = {s.credential.id: s for s in self._states}
            states: List[CredentialState] = []
            for cred in creds:
                if cred.id in existing:
                    state = existing[cred.id]
                    state.credential = cred
                    states.append(state)
                else:
                    states.append(CredentialState(credential=cred))
            self._states = states
            self._rebuild_cycle()
        return len(creds)
    def _rebuild_cycle(self) -> None:
        self._cycle = itertools.cycle(range(len(self._states))) if self._states else itertools.cycle([])
    def _save(self) -> None:
        creds = [s.credential for s in self._states if s.credential.source != "anonymous" or s.credential.cookie_header]
        CredentialFile(self.path).save(creds)
    def total(self) -> int:
        with self._lock:
            return len(self._states)
    def working(self) -> int:
        with self._lock:
            return sum(1 for s in self._states if not s.depleted)
    def prune(self) -> int:
        with self._lock:
            before = len(self._states)
            self._states = [s for s in self._states if not s.depleted]
            self._rebuild_cycle()
            if not self._states:
                self._states = [CredentialState(credential=Credential(id="anon-default"))]
                self._rebuild_cycle()
            return before - len(self._states)
    def stats(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            return {
                "total": len(self._states),
                "working": sum(1 for s in self._states if not s.depleted),
                "available": sum(1 for s in self._states if s.available(now)),
                "credentials": [
                    {
                        "id": s.credential.id,
                        "source": s.credential.source,
                        "has_cookies": bool(s.credential.cookie_header),
                        "successes": s.successes,
                        "failures": s.failures,
                        "depleted": s.depleted,
                        "cooldown_remaining": max(0.0, round(s.cooldown_until - now, 1)),
                        "last_error": s.last_error,
                    }
                    for s in self._states
                ],
            }
    def acquire(self, exclude_ids: Optional[set] = None, allow_cooldown: bool = False) -> Optional[CredentialState]:
        now = time.time()
        exclude_ids = exclude_ids or set()
        with self._lock:
            n = len(self._states)
            picked: Optional[CredentialState] = None
            for _ in range(n):
                idx = next(self._cycle)
                state = self._states[idx]
                if state.credential.id in exclude_ids:
                    continue
                if not allow_cooldown and not state.available(now):
                    continue
                if allow_cooldown and state.depleted:
                    continue
                state.last_used = now
                picked = state
                break
        return picked
    def available(self) -> int:
        now = time.time()
        with self._lock:
            return sum(1 for s in self._states if s.available(now))
    def report_result(self, state: CredentialState, ok: bool, depleted: bool = False, error: Optional[str] = None, neutral: bool = False) -> None:
        if neutral:
            return
        changed = False
        with self._lock:
            if ok:
                state.successes += 1
                state.failures = 0
                state.cooldown_until = 0.0
                state.last_error = None
            else:
                state.failures += 1
                state.last_error = error
                if depleted or state.failures >= MAX_FAILURES:
                    state.depleted = True
                    changed = True
                else:
                    backoff = min(COOLDOWN_BASE * (2 ** (state.failures - 1)), COOLDOWN_MAX)
                    state.cooldown_until = time.time() + backoff
            if changed and any(not s.depleted for s in self._states):
                self._states = [s for s in self._states if not s.depleted]
                self._rebuild_cycle()
                self._save()
    def save_tokens(self, state: CredentialState) -> None:
        self._save()
    def ensure_ready(self, block: bool = True) -> None:
        self.prune()
class RHRouter:
    def __init__(self, pool: Optional[CredentialPool] = None, registry: ModelRegistry = REGISTRY) -> None:
        self.pool = pool or CredentialPool()
        self.registry = registry
    def ensure_ready(self, block: bool = True) -> None:
        self.pool.ensure_ready(block=block)
    def stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        *,
        tools: Optional[List[Any]] = None,
        tool_choice: Optional[Any] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        response_format: Optional[Any] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        if tools and tool_choice == "none":
            tools = None
            tool_choice = None
        card = self.registry.resolve(model)
        if card is None:
            raise NoCredentialsError("model registry is empty - no curated models available")
        ui_messages = to_ui_messages(messages)
        if len(ui_messages) > MAX_MESSAGES:
            system_msgs = [m for m in ui_messages if m["role"] == "system"]
            rest = [m for m in ui_messages if m["role"] != "system"]
            ui_messages = system_msgs + rest[-MESSAGE_TRIM_TARGET:]
        total_chars = sum(_part_text_len(m) for m in ui_messages)
        if total_chars > PROMPT_TOKEN_CAP * 4:
            system_msgs = [m for m in ui_messages if m["role"] == "system"]
            rest = [m for m in ui_messages if m["role"] != "system"]
            budget = PROMPT_TOKEN_CAP * 4
            kept_system = [_truncate_parts(m, budget // max(1, len(system_msgs))) for m in system_msgs]
            tail: List[Dict[str, Any]] = []
            running = 0
            for m in reversed(rest):
                trimmed = _truncate_parts(m, 4096)
                running += _part_text_len(trimmed)
                tail.append(trimmed)
                if running >= budget // 2:
                    break
            ui_messages = kept_system + list(reversed(tail))
        effort = _effort_for(reasoning_effort)
        last_error: Optional[Exception] = None
        emulate_allowed = bool(tools) and tool_choice != "none"
        routing_pass = 0
        while True:
            routing_pass += 1
            saw_transient = False
            saw_hard = False
            saw_site_outage = False
            for route in self.registry.ordered_routes(card, prefer_tools=bool(tools)):
                emitted = False
                for _attempt in range(MAX_CRED_ATTEMPTS):
                    state = self.pool.acquire(allow_cooldown=routing_pass > 1)
                    if state is None and any(not s.depleted for s in self.pool._states):
                        time.sleep(5.0)
                        state = self.pool.acquire(allow_cooldown=routing_pass > 1)
                    if state is None:
                        if any(not s.depleted for s in self.pool._states):
                            raise AllCredentialsBusyError("All working credentials are cooling down; try again shortly.")
                        raise NoCredentialsError("No credentials available.")
                    client = TryingOpenClient(state.credential)
                    try:
                        buf: List[Dict[str, Any]] = []
                        reason_buf: List[Dict[str, Any]] = []
                        usage_buf: Optional[Dict[str, Any]] = None
                        finish_buf: Optional[str] = None
                        upstream_messages = ui_messages
                        if emulate_allowed:
                            upstream_messages = to_ui_messages(_emulation_messages(messages, tools, tool_choice))
                        calls: Optional[List[Dict[str, Any]]] = None
                        preamble = ""
                        if route.provider == COMPARE_PROVIDER:
                            source = client.stream_compare(upstream_messages, route)
                        else:
                            source = client.stream(upstream_messages, route, effort=effort)
                        for ev in source:
                            etype = ev.get("type")
                            if etype == "error":
                                msg = str(ev.get("message", "upstream error"))
                                if _looks_route_scoped(msg):
                                    raise _RouteRejected(msg)
                                raise _StreamError(msg)
                            if etype == "content":
                                buf.append(ev)
                            elif etype == "usage":
                                usage_buf = ev.get("usage")
                            elif etype == "finish":
                                finish_buf = ev.get("finish_reason")
                            elif etype == "reasoning":
                                reason_buf.append(ev)
                        self.pool.report_result(state, ok=True)
                        self.pool.save_tokens(state)
                        self.registry.report_route(route.mid, ok=True)
                        if not emitted:
                            emitted = True
                            yield {
                                "type": "route",
                                "credential_id": state.credential.id,
                                "mid": route.mid,
                                "pid": route.pid,
                                "provider": route.provider,
                                "model": card.id,
                            }
                        if emulate_allowed:
                            joined = "".join(e.get("text", "") for e in buf)
                            calls = _parse_plaintext_tool_calls(joined)
                            if calls is None:
                                reason_text = "".join(e.get("text", "") for e in reason_buf)
                                calls = _parse_plaintext_tool_calls(reason_text)
                            if calls is not None:
                                json_pos = _json_position(joined)
                                preamble = joined[:json_pos].strip() if json_pos is not None else ""
                        if calls:
                            if preamble:
                                yield {"type": "content", "text": preamble}
                            for i, call in enumerate(calls):
                                yield {
                                    "type": "tool_call_delta",
                                    "index": i,
                                    "id": f"call_{uuid.uuid4().hex[:24]}",
                                    "name": call["name"],
                                    "arguments": call["arguments"],
                                }
                            if usage_buf is not None:
                                yield {"type": "usage", "usage": usage_buf}
                            yield {"type": "finish", "finish_reason": "tool_calls"}
                        else:
                            for ev in reason_buf:
                                yield ev
                            for ev in buf:
                                yield ev
                            if usage_buf is not None:
                                yield {"type": "usage", "usage": usage_buf}
                            yield {"type": "finish", "finish_reason": finish_buf or "stop"}
                        return
                    except _RouteRejected as exc:
                        last_error = exc
                        hard = _is_tool_error(str(exc)) or _is_site_credit_outage(str(exc)) or (exc.status in (400, 401, 403, 404, 429))
                        saw_hard = saw_hard or hard
                        saw_transient = saw_transient or not hard
                        self.registry.report_route(route.mid, ok=False, hard=hard)
                        break
                    except requests.HTTPError as exc:
                        code = exc.response.status_code if exc.response is not None else None
                        depleted = code in (401, 403)
                        rate_limited = code == 429
                        saw_transient = True
                        if code in (500, 502, 503, 504):
                            saw_site_outage = saw_site_outage or _is_site_outage(str(exc))
                            self.registry.report_route(route.mid, ok=False, hard=False)
                            self.pool.report_result(state, ok=False, depleted=depleted, error=f"http {code}", neutral=True)
                            last_error = exc
                            if emitted:
                                raise
                            break
                        self.pool.report_result(state, ok=False, depleted=depleted, error=f"http {code}", neutral=not (depleted or rate_limited))
                        last_error = exc
                        if emitted:
                            raise
                        _sleep_jitter(_attempt, exc.response.headers.get("Retry-After") if exc.response is not None else None)
                        continue
                    except _StreamError as exc:
                        saw_transient = True
                        self.pool.report_result(state, ok=False, depleted=False, error=str(exc), neutral=True)
                        last_error = exc
                        if emitted:
                            raise
                        _sleep_jitter(_attempt)
                        continue
                    except Exception as exc:
                        saw_transient = True
                        self.pool.report_result(state, ok=False, depleted=False, error=str(exc), neutral=True)
                        last_error = exc
                        if emitted:
                            raise
                        _sleep_jitter(_attempt)
                        continue
            max_passes = 2
            if routing_pass >= max_passes or saw_hard or not saw_transient:
                break
            time.sleep(SITE_OUTAGE_RETRY_DELAY if saw_site_outage else 5.0)
        raise RuntimeError(f"All routes for '{card.id}' failed. Last error: {last_error}")
    def collect(self, messages: List[Dict[str, Any]], model: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_acc: Dict[int, Dict[str, Any]] = {}
        order: List[int] = []
        usage: Optional[Dict[str, Any]] = None
        finish_reason = "stop"
        route_info: Optional[Dict[str, Any]] = None
        for ev in self.stream(messages, model=model, **kwargs):
            etype = ev.get("type")
            if etype == "route":
                route_info = ev
            elif etype == "content":
                text_parts.append(ev.get("text", ""))
            elif etype == "reasoning":
                reasoning_parts.append(ev.get("text", ""))
            elif etype == "tool_call_delta":
                index = ev.get("index", 0)
                if index not in tool_acc:
                    tool_acc[index] = {
                        "id": ev.get("id") or f"call_{int(time.time()*1000):x}{len(order):02x}",
                        "name": ev.get("name") or "",
                        "arguments": "",
                    }
                    order.append(index)
                if ev.get("id"):
                    tool_acc[index]["id"] = ev["id"]
                if ev.get("name"):
                    tool_acc[index]["name"] = ev["name"]
                tool_acc[index]["arguments"] += ev.get("arguments") or ""
            elif etype == "usage":
                usage = ev.get("usage")
            elif etype == "finish":
                finish_reason = ev.get("finish_reason", finish_reason)
        tool_calls = [
            {
                "id": tool_acc[i]["id"],
                "type": "function",
                "function": {"name": tool_acc[i]["name"], "arguments": tool_acc[i]["arguments"]},
            }
            for i in order
        ]
        if tool_calls:
            finish_reason = "tool_calls"
        return {
            "text": "".join(text_parts),
            "reasoning": "".join(reasoning_parts),
            "tool_calls": tool_calls,
            "usage": usage,
            "route": route_info,
            "finish_reason": finish_reason,
        }
def _sleep_jitter(attempt: int, retry_after: Optional[str] = None) -> None:
    delay = min(RETRY_MAX_DELAY, RETRY_BASE_DELAY * (2 ** attempt)) + _random.random()
    if retry_after:
        try:
            requested = float(retry_after)
            delay = min(max(delay, requested), 15.0)
        except ValueError:
            pass
    time.sleep(delay)
def _looks_route_scoped(message: str) -> bool:
    low = message.lower()
    markers = (
        "payment", "402", "insufficient balance", "credit balance",
        "not found", "404", "does not exist",
        "unauthorized", "401", "403", "key limit",
        "billing", "entitlement",
        "finish_reason=error",
        "not supported", "input too long", "context length",
        "no available channel", "run out of api credit", "out of credit",
        "try again later", "didn't respond", "did not respond",
        "overloaded", "temporarily unavailable",
    )
    return any(m in low for m in markers)
def _is_site_credit_outage(message: str) -> bool:
    low = message.lower()
    markers = (
        "run out of api credit", "out of credit", "insufficient credit",
        "quota exceeded", "quota exhausted", "compare quota exhausted",
    )
    return any(m in low for m in markers)
def _is_site_outage(message: str) -> bool:
    low = message.lower()
    markers = (
        "no available channel", "run out of api credit", "out of credit",
        "try again later", "overloaded", "temporarily unavailable",
        "upstream 502", "upstream 503", "upstream 504",
    )
    return any(m in low for m in markers)
def _is_tool_error(message: str) -> bool:
    low = message.lower()
    markers = (
        "tool choice is none", "tool_use_failed", "tool use failed",
        "function calling", "tool calling", "not supported",
        "insufficient credits", "input too long",
    )
    return any(m in low for m in markers)