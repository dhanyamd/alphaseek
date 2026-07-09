"""LLM client — no mock, no silent fallbacks.

Supports both OpenAI-compatible providers (OpenRouter, Groq, etc.) AND
Anthropic's native API (which is NOT OpenAI-compatible).

Detection: if LLM_BASE_URL contains "anthropic.com", the provider uses the
Anthropic SDK directly. Everything else uses the OpenAI SDK.

Env (.env):
    LLM_API_KEY   = provider key (required)
    LLM_BASE_URL  = base URL (https://api.anthropic.com for Claude, or
                    OpenAI-compatible URL for Groq/OpenRouter/...)
    LLM_MODEL     = model id (Anthropic: claude-opus-4-8, OpenAI: gpt-4o-mini)
    LLM_FALLBACK_MODELS = optional comma-separated models tried in order when the
                          primary is rate-limited or stalling (same provider/key)
    LLM_MODEL_<ROLE>    = per-role ordered model chain (on the default provider for that role)
    LLM_PROVIDER_<ROLE> = route a role to a specific provider: "primary" or "secondary"
                          (default: tries all providers in order)

  Secondary provider (separate rate-limit bucket — cross-provider failover):
    LLM_API_KEY_2, LLM_BASE_URL_2, LLM_MODELS_2
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

    def __init__(
        self,
        name: str,
        key: str,
        base_url: str,
        default: str,
        fallbacks: list[str],
        role_models: dict[str, list[str]],
    ) -> None:
        from openai import OpenAI

        self.name = name
        self.default = default
        self.fallbacks = fallbacks
        self.role_models = role_models
        self.supports_reasoning_effort = False
        self.supports_required_tool = True
        self.min_interval = float(os.getenv(f"LLM_MIN_INTERVAL_{name.upper()}", "0") or 0)
        self._last_call = 0.0
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


class _AnthropicProvider:
    """Anthropic provider — uses anthropic SDK directly (NOT OpenAI-compatible)."""

    def __init__(
        self,
        name: str,
        key: str,
        default: str,
        fallbacks: list[str],
        role_models: dict[str, list[str]],
    ) -> None:
        from anthropic import Anthropic

        self.name = name
        self.default = default
        self.fallbacks = fallbacks
        self.role_models = role_models
        self.client = Anthropic(api_key=key)
        self.min_interval = float(os.getenv(f"LLM_MIN_INTERVAL_{name.upper()}", "0") or 0)
        self._last_call = 0.0
        self.supports_reasoning_effort = False
        self.supports_required_tool = True

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
    for role in ("researcher", "coder", "viz", "reporter", "reader", "exporter"):
        raw = os.getenv(f"LLM_MODEL_{role.upper()}{suffix}", "")
        models = [m.strip() for m in raw.split(",") if m.strip()]
        if models:
            chains[role] = models
    return chains


def _is_anthropic(base_url: str) -> bool:
    return "anthropic.com" in base_url.lower()


class LLM:
    def __init__(self) -> None:
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.base_url = os.getenv("LLM_BASE_URL", "").strip()
        self.model = os.getenv("LLM_MODEL", "").strip()
        self.fallbacks = [
            m.strip() for m in os.getenv("LLM_FALLBACK_MODELS", "").split(",") if m.strip()
        ]
        self.role_models = _role_chains("")
        self.role_providers: dict[str, str] = {}
        for role in ("researcher", "coder", "viz", "reporter", "reader", "exporter"):
            p = os.getenv(f"LLM_PROVIDER_{role.upper()}", "").strip().lower()
            if p:
                self.role_providers[role] = p

        self.providers: list[_Provider | _AnthropicProvider] = []
        if self.api_key and self.base_url and self.model:
            if _is_anthropic(self.base_url):
                self.providers.append(
                    _AnthropicProvider(
                        "primary", self.api_key, self.model, self.fallbacks, self.role_models
                    )
                )
            else:
                self.providers.append(
                    _Provider(
                        "primary",
                        self.api_key,
                        self.base_url,
                        self.model,
                        self.fallbacks,
                        self.role_models,
                    )
                )
        # Optional secondary provider — a separate rate-limit bucket.
        key2 = os.getenv("LLM_API_KEY_2", "").strip()
        base2 = os.getenv("LLM_BASE_URL_2", "").strip()
        models2 = [m.strip() for m in os.getenv("LLM_MODELS_2", "").split(",") if m.strip()]
        if key2 and base2 and models2:
            if _is_anthropic(base2):
                self.providers.append(
                    _AnthropicProvider(
                        "secondary", key2, models2[0], models2[1:], _role_chains("_2")
                    )
                )
            else:
                self.providers.append(
                    _Provider("secondary", key2, base2, models2[0], models2[1:], _role_chains("_2"))
                )

    def _attempts(
        self, role: str | None = None
    ) -> list[tuple[_Provider | _AnthropicProvider, str]]:
        if role and role in self.role_providers:
            target = self.role_providers[role]
            preferred = [
                (p, m) for p in self.providers if p.name.lower() == target for m in p.models(role)
            ]
            others = [
                (p, m) for p in self.providers if p.name.lower() != target for m in p.models(role)
            ]
            return preferred + others
        return [(p, m) for p in self.providers for m in p.models(role)]

    def _models(self, role: str | None = None) -> list[str]:
        return [f"{p.name}:{m}" for p, m in self._attempts(role)]

    @property
    def configured(self) -> bool:
        return bool(self.providers)

    @property
    def mode(self) -> str:
        return "live" if self.configured else "unconfigured"

    #  core

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

        last_err: Exception | None = None
        for prov, model in self._attempts(role):
            for attempt in range(attempts):
                try:
                    prov.throttle()
                    if isinstance(prov, _AnthropicProvider):
                        resp = prov.client.messages.create(
                            model=model,
                            system=system,
                            messages=[{"role": "user", "content": user}],
                            max_tokens=max_tokens,
                        )
                        text = ""
                        for block in resp.content:
                            if block.type == "text":
                                text += block.text
                        return text
                    else:
                        kwargs: dict = dict(
                            model=model,
                            messages=[
                                {"role": "system", "content": system},
                                {"role": "user", "content": user},
                            ],
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                        if json_mode:
                            kwargs["response_format"] = {"type": "json_object"}
                        kwargs.pop("reasoning_effort", None)
                        if prov.supports_reasoning_effort:
                            kwargs["reasoning_effort"] = "none"
                        resp = prov.client.chat.completions.create(**kwargs)
                        return resp.choices[0].message.content or ""
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    msg = str(e)
                    if not isinstance(prov, _AnthropicProvider):
                        if "reasoning_effort" in msg and prov.supports_reasoning_effort:
                            prov.supports_reasoning_effort = False
                            continue
                        if json_mode and ("response_format" in msg or "json_object" in msg):
                            json_mode = False
                            continue
                    status = getattr(e, "status_code", None)
                    retriable = (
                        status in (429, 500, 502, 503)
                        or "rate" in msg.lower()
                        or "timed out" in msg.lower()
                    )
                    if not retriable or attempt == attempts - 1:
                        break
                    wait = self._retry_after(e)
                    if wait and wait > 75:
                        break
                    time.sleep(wait or min(2**attempt * 2.0, 12.0))
        raise LLMError(
            f"LLM unavailable (tried {', '.join(self._models(role))}): {last_err}"
        ) from last_err

    def chat_tools(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2300,
        attempts: int = 4,
        role: str | None = None,
        tool_choice: str = "auto",
    ) -> dict:
        """Tool-calling turn. Returns {"content": str, "tool_calls": [{id,name,arguments}]}.

        Same retry/backoff policy as chat(); failures raise LLMError.
        """
        if not self.configured:
            raise LLMError("No LLM configured — set LLM_API_KEY/LLM_BASE_URL/LLM_MODEL.")

        last_err: Exception | None = None
        for prov, model in self._attempts(role):
            for attempt in range(attempts):
                try:
                    prov.throttle()
                    if isinstance(prov, _AnthropicProvider):
                        return self._anthropic_tool_call(
                            prov,
                            model,
                            system,
                            messages,
                            tools,
                            temperature,
                            max_tokens,
                            tool_choice,
                        )
                    else:
                        full = [{"role": "system", "content": system}, *messages]
                        extra = (
                            {"reasoning_effort": "none"} if prov.supports_reasoning_effort else {}
                        )
                        tc = tool_choice if prov.supports_required_tool else "auto"
                        resp = prov.client.chat.completions.create(
                            model=model,
                            messages=full,
                            tools=tools,
                            tool_choice=tc,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            **extra,
                        )
                        m = resp.choices[0].message
                        return {
                            "content": m.content or "",
                            "model": f"{prov.name}:{model}",
                            "tool_calls": [
                                {
                                    "id": t.id,
                                    "name": t.function.name,
                                    "arguments": t.function.arguments,
                                }
                                for t in (m.tool_calls or [])
                            ],
                        }
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    msg = str(e)
                    if isinstance(prov, _AnthropicProvider):
                        pass  # no OpenAI-specific degradation needed
                    else:
                        if "reasoning_effort" in msg and prov.supports_reasoning_effort:
                            prov.supports_reasoning_effort = False
                            continue
                        if "tool_choice" in msg and prov.supports_required_tool:
                            prov.supports_required_tool = False
                            continue
                        body = getattr(e, "body", None) or {}
                        failed = None
                        if isinstance(body, dict):
                            failed = body.get("failed_generation") or (
                                body.get("error", {}) or {}
                            ).get("failed_generation")
                        if not failed and "failed_generation" in msg:
                            m = re.search(r"'failed_generation': '(.*)'}}\s*$", msg, re.S)
                            failed = m.group(1) if m else None
                        if failed and "tool_use_failed" in msg:
                            args = failed
                            marker = "<function=run_python>"
                            if marker in args:
                                args = args.split(marker, 1)[1]
                            return {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "recovered",
                                        "name": "run_python",
                                        "arguments": args.strip(),
                                    }
                                ],
                            }
                    status = getattr(e, "status_code", None)
                    retriable = (
                        status in (429, 500, 502, 503)
                        or "rate" in msg.lower()
                        or "timed out" in msg.lower()
                    )
                    if not retriable or attempt == attempts - 1:
                        break
                    wait = self._retry_after(e)
                    if wait and wait > 75:
                        break
                    time.sleep(wait or min(2**attempt * 2.0, 12.0))
        raise LLMError(
            f"LLM tool call unavailable (tried {', '.join(self._models(role))}): {last_err}"
        ) from last_err

    #  Anthropic internals

    def _anthropic_tool_call(
        self,
        prov: _AnthropicProvider,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
        temperature: float,
        max_tokens: int,
        tool_choice: str,
    ) -> dict:
        """Make an Anthropic tool-calling turn with full message format conversion."""
        # Convert tool definitions from OpenAI format to Anthropic format
        anthro_tools = []
        for t in tools:
            fn = t.get("function", t)
            anthro_tools.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", fn.get("input_schema", {})),
                }
            )

        # Convert tool_choice
        anthro_tc: dict = {"type": "auto"}
        if tool_choice == "required":
            anthro_tc = {"type": "any"}

        # Convert messages from OpenAI format to Anthropic format
        anthro_messages = self._to_anthropic_messages(messages)

        resp = prov.client.messages.create(
            model=model,
            system=system,
            messages=anthro_messages,
            tools=anthro_tools,
            tool_choice=anthro_tc,
            max_tokens=max_tokens,
        )

        # Parse response
        text = ""
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text += block.text
            elif block.type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "name": block.name,
                        "arguments": json.dumps(block.input),
                    }
                )

        return {
            "content": text,
            "model": f"{prov.name}:{model}",
            "tool_calls": tool_calls,
        }

    @staticmethod
    def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
        """Convert OpenAI-format messages to Anthropic format.

        OpenAI::
          {"role": "system"|"user"|"assistant"|"tool",
           "content": str,
           "tool_calls": [{"id", "function": {"name", "arguments"}}]}

        Anthropic::
          {"role": "user"|"assistant",
           "content": [{"type": "text"|"tool_use"|"tool_result", ...}]}
        """
        result: list[dict] = []
        for msg in messages:
            role = msg["role"]

            if role == "system":
                continue

            elif role == "user":
                content = msg.get("content") or ""
                if isinstance(content, str):
                    blocks = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    blocks = content
                else:
                    blocks = [{"type": "text", "text": str(content)}]
                result.append({"role": "user", "content": blocks})

            elif role == "assistant":
                content = msg.get("content") or ""
                blocks: list[dict] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", tc)
                    try:
                        args = (
                            json.loads(fn["arguments"])
                            if isinstance(fn["arguments"], str)
                            else fn["arguments"]
                        )
                    except (json.JSONDecodeError, KeyError):
                        args = {}
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": fn["name"],
                            "input": args,
                        }
                    )
                result.append({"role": "assistant", "content": blocks})

            elif role == "tool":
                result.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.get("tool_call_id", ""),
                                "content": msg.get("content", ""),
                            }
                        ],
                    }
                )

        return result

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.4,
        max_tokens: int = 2048,
        role: str | None = None,
    ) -> dict:
        """Chat expecting a JSON object; one repair pass, then a visible error."""
        raw = self.chat(
            system, user, temperature=temperature, max_tokens=max_tokens, json_mode=True, role=role
        )
        obj = self.json_from(raw)
        if obj is not None:
            return obj
        repaired = self.chat(
            "You fix malformed JSON. Return ONLY the corrected JSON object, nothing else.",
            raw[:4000],
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=True,
        )
        obj = self.json_from(repaired)
        if obj is not None:
            return obj
        raise LLMError(f"Model returned unparseable JSON: {raw[:200]!r}")

    #  helpers
    @staticmethod
    def _retry_after(err: Exception) -> float | None:
        msg = str(err)
        for pat in (
            r"try again in ([\d.]+)s",
            r"retry_after_seconds'?: ([\d.]+)",
            r"'Retry-After': '([\d.]+)'",
        ):
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
        # Strip markdown fences (```json ... ```) that LLMs sometimes wrap around JSON
        import re

        text = re.sub(r"^```(?:json)?\s*\n?", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"\n?```\s*$", "", text.strip())
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
