"""LLM client — no mock, no silent fallbacks.

Design rules (learned from production agent systems):
  * A missing key or a failed call is a VISIBLE error, never a canned response.
  * Rate limits (429) and transient 5xx get exponential-backoff retries,
    honoring Retry-After when the provider sends it.
  * JSON tasks use the provider's native json_object mode when available,
    with one repair pass before giving up.

Env (.env):
    LLM_API_KEY   = provider key (required)
    LLM_BASE_URL  = OpenAI-compatible base (Groq/OpenRouter/NIM/Cerebras/...)
    LLM_MODEL     = model id
    LLM_FALLBACK_MODELS = optional comma-separated models tried in order when the
                          primary is rate-limited or stalling (same provider/key)
    LLM_MODEL_<ROLE>    = per-role ordered model chain on the primary provider

  Secondary provider (separate rate-limit bucket — cross-provider failover):
    LLM_API_KEY_2, LLM_BASE_URL_2 = a second OpenAI-compatible provider (e.g.
                          Groq / Gemini / Cerebras). When the primary provider's
                          whole chain is rate-limited, calls fail over to here.
    LLM_MODELS_2       = comma-separated models on the secondary, tried for ALL
                          roles as the failover tail.
"""
from __future__ import annotations

import json
import os
import re
import time


class LLMError(Exception):
    """Surfaced to the user in the event stream — never swallowed."""


class _Provider:
    """One OpenAI-compatible endpoint (its own key = its own rate-limit bucket)."""

    def __init__(self, name: str, key: str, base_url: str, default: str,
                 fallbacks: list[str], role_models: dict[str, list[str]]) -> None:
        from openai import OpenAI

        self.name = name
        self.default = default
        self.fallbacks = fallbacks
        self.role_models = role_models
        # Thinking models (Gemini 2.5, o-series) spend max_tokens on hidden
        # reasoning and truncate the answer; reasoning_effort="none" routes the
        # whole budget to output. Flipped off if a provider rejects the param.
        self.supports_reasoning_effort = True
        self.supports_required_tool = True
        # Client-side pacing: on a tight per-minute limit (Gemini free ~20/min),
        # keep a minimum gap between calls so we never burst past it. Set via
        # LLM_MIN_INTERVAL_<NAME> seconds; 0 = no throttle.
        self.min_interval = float(os.getenv(f"LLM_MIN_INTERVAL_{name.upper()}", "0") or 0)
        self._last_call = 0.0
        # Short per-request timeout: a stalled provider fails fast and visibly.
        self.client = OpenAI(api_key=key, base_url=base_url, timeout=35, max_retries=0)

    def throttle(self) -> None:
        if self.min_interval <= 0:
            return
        gap = time.time() - self._last_call
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last_call = time.time()

    def models(self, role: str | None) -> list[str]:
        if role and role in self.role_models:
            return self.role_models[role]
        return [self.default, *self.fallbacks] if self.default else self.fallbacks


def _role_chains(suffix: str) -> dict[str, list[str]]:
    chains: dict[str, list[str]] = {}
    for role in ("researcher", "coder", "critic", "viz", "reporter", "reader"):
        raw = os.getenv(f"LLM_MODEL_{role.upper()}{suffix}", "")
        models = [m.strip() for m in raw.split(",") if m.strip()]
        if models:
            chains[role] = models
    return chains


class LLM:
    def __init__(self) -> None:
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.base_url = os.getenv("LLM_BASE_URL", "").strip()
        self.model = os.getenv("LLM_MODEL", "").strip()
        self.fallbacks = [m.strip() for m in
                          os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()]
        self.role_models = _role_chains("")

        self.providers: list[_Provider] = []
        if self.api_key and self.base_url and self.model:
            self.providers.append(_Provider(
                "primary", self.api_key, self.base_url, self.model,
                self.fallbacks, self.role_models))
        # Optional secondary provider — a separate rate-limit bucket.
        key2 = os.getenv("LLM_API_KEY_2", "").strip()
        base2 = os.getenv("LLM_BASE_URL_2", "").strip()
        models2 = [m.strip() for m in os.getenv("LLM_MODELS_2", "").split(",") if m.strip()]
        if key2 and base2 and models2:
            self.providers.append(_Provider(
                "secondary", key2, base2, models2[0], models2[1:], _role_chains("_2")))

    def _attempts(self, role: str | None = None) -> list[tuple[_Provider, str]]:
        """Every (provider, model) to try, in order: primary chain, then secondary."""
        return [(p, m) for p in self.providers for m in p.models(role)]

    def _models(self, role: str | None = None) -> list[str]:
        return [f"{p.name}:{m}" for p, m in self._attempts(role)]

    @property
    def configured(self) -> bool:
        return bool(self.providers)

    @property
    def mode(self) -> str:
        return "live" if self.configured else "unconfigured"

    # ------------------------------------------------------------------ core
    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.4,
        max_tokens: int = 4096,
        json_mode: bool = False,
        attempts: int = 4,
        role: str | None = None,
    ) -> str:
        if not self.configured:
            raise LLMError(
                "No LLM configured. Set LLM_API_KEY, LLM_BASE_URL and LLM_MODEL in backend/.env "
                "(e.g. Groq: https://api.groq.com/openai/v1, llama-3.3-70b-versatile)."
            )
        kwargs: dict = dict(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_err: Exception | None = None
        for prov, model in self._attempts(role):
            kwargs["model"] = model
            kwargs.pop("reasoning_effort", None)
            if prov.supports_reasoning_effort:
                kwargs["reasoning_effort"] = "none"
            for attempt in range(attempts):
                try:
                    prov.throttle()
                    resp = prov.client.chat.completions.create(**kwargs)
                    return resp.choices[0].message.content or ""
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    msg = str(e)
                    if "reasoning_effort" in msg and "reasoning_effort" in kwargs:
                        prov.supports_reasoning_effort = False
                        kwargs.pop("reasoning_effort", None)
                        continue
                    if json_mode and ("response_format" in msg or "json_object" in msg):
                        kwargs.pop("response_format", None)
                        json_mode = False
                        continue
                    status = getattr(e, "status_code", None)
                    retriable = (status in (429, 500, 502, 503)
                                 or "rate" in msg.lower() or "timed out" in msg.lower())
                    if not retriable or attempt == attempts - 1:
                        break   # next (provider, model)
                    wait = self._retry_after(e)
                    if wait and wait > 75:
                        break   # daily/long limit here -> fail over to the next
                    time.sleep(wait or min(2 ** attempt * 2.0, 12.0))
        raise LLMError(
            f"LLM unavailable (tried {', '.join(self._models(role))}): {last_err}"
        ) from last_err

    def chat_tools(self, system: str, messages: list[dict], tools: list[dict], *,
                   temperature: float = 0.3, max_tokens: int = 2300,
                   attempts: int = 4, role: str | None = None,
                   tool_choice: str = "auto") -> dict:
        """Tool-calling turn. Returns {"content": str, "tool_calls": [{id,name,arguments}]}.

        Same retry/backoff policy as chat(); failures raise LLMError.
        """
        if not self.configured:
            raise LLMError("No LLM configured — set LLM_API_KEY/LLM_BASE_URL/LLM_MODEL.")
        full = [{"role": "system", "content": system}, *messages]
        last_err: Exception | None = None
        for prov, model in self._attempts(role):
            for attempt in range(attempts):
                try:
                    prov.throttle()
                    extra = ({"reasoning_effort": "none"}
                             if prov.supports_reasoning_effort else {})
                    tc = tool_choice if prov.supports_required_tool else "auto"
                    resp = prov.client.chat.completions.create(
                        model=model, messages=full, tools=tools, tool_choice=tc,
                        temperature=temperature, max_tokens=max_tokens, **extra,
                    )
                    m = resp.choices[0].message
                    return {
                        "content": m.content or "",
                        "model": f"{prov.name}:{model}",   # which provider+model answered
                        "tool_calls": [
                            {"id": t.id, "name": t.function.name, "arguments": t.function.arguments}
                            for t in (m.tool_calls or [])
                        ],
                    }
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    msg = str(e)
                    if "reasoning_effort" in msg and prov.supports_reasoning_effort:
                        prov.supports_reasoning_effort = False
                        continue    # retry this model without the param
                    if "tool_choice" in msg and prov.supports_required_tool:
                        prov.supports_required_tool = False
                        continue    # provider rejects "required" -> fall back to auto
                    # Groq/llama sometimes writes a correct call the parser rejects
                    # (nested quotes in code). The intended call is in
                    # failed_generation — salvage it instead of failing.
                    body = getattr(e, "body", None) or {}
                    failed = None
                    if isinstance(body, dict):
                        failed = (body.get("failed_generation")
                                  or (body.get("error", {}) or {}).get("failed_generation"))
                    if not failed and "failed_generation" in msg:
                        m = re.search(r"'failed_generation': '(.*)'}}\s*$", msg, re.S)
                        failed = m.group(1) if m else None
                    if failed and "tool_use_failed" in msg:
                        args = failed
                        marker = "<function=run_python>"
                        if marker in args:
                            args = args.split(marker, 1)[1]
                        return {"content": "", "tool_calls": [
                            {"id": "recovered", "name": "run_python", "arguments": args.strip()}
                        ]}
                    status = getattr(e, "status_code", None)
                    retriable = (status in (429, 500, 502, 503)
                                 or "rate" in msg.lower() or "timed out" in msg.lower())
                    if not retriable or attempt == attempts - 1:
                        break
                    wait = self._retry_after(e)
                    if wait and wait > 75:
                        break
                    time.sleep(wait or min(2 ** attempt * 2.0, 12.0))
        raise LLMError(
            f"LLM tool call unavailable (tried {', '.join(self._models(role))}): {last_err}"
        ) from last_err

    def chat_json(self, system: str, user: str, *, temperature: float = 0.4,
                  max_tokens: int = 2048, role: str | None = None) -> dict:
        """Chat expecting a JSON object; one repair pass, then a visible error."""
        raw = self.chat(system, user, temperature=temperature,
                        max_tokens=max_tokens, json_mode=True, role=role)
        obj = self.json_from(raw)
        if obj is not None:
            return obj
        repaired = self.chat(
            "You fix malformed JSON. Return ONLY the corrected JSON object, nothing else.",
            raw[:4000], temperature=0.0, max_tokens=max_tokens, json_mode=True,
        )
        obj = self.json_from(repaired)
        if obj is not None:
            return obj
        raise LLMError(f"Model returned unparseable JSON: {raw[:200]!r}")

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _retry_after(err: Exception) -> float | None:
        msg = str(err)
        # Groq: "Please try again in 50.39s" · OpenRouter: "'retry_after_seconds': 29"
        for pat in (r"try again in ([\d.]+)s",
                    r"retry_after_seconds'?: ([\d.]+)",
                    r"'Retry-After': '([\d.]+)'"):
            m = re.search(pat, msg, re.IGNORECASE)
            if m:
                return float(m.group(1)) + 1.0
        headers = getattr(getattr(err, "response", None), "headers", None)
        if headers and headers.get("retry-after"):
            try:
                return float(headers["retry-after"]) + 0.5
            except ValueError:
                pass
        return None

    @staticmethod
    def json_from(text: str) -> dict | None:
        if not text:
            return None
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            return json.loads(text[start:end])
        except Exception:  # noqa: BLE001
            return None


_llm: LLM | None = None


def get_llm() -> LLM:
    global _llm
    if _llm is None:
        _llm = LLM()
    return _llm
