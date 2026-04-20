"""Protocol + registry + Transform adapter for tool_result interceptors."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from headroom.cache.compression_cache import (
    _extract_tool_result_content,
    _is_tool_result_message,
    _swap_tool_result_content,
)
from headroom.config import TransformResult
from headroom.tokenizer import Tokenizer
from headroom.transforms.base import Transform

logger = logging.getLogger(__name__)


@runtime_checkable
class ToolResultInterceptor(Protocol):
    """A stateless rewriter for a single tool_result's text content.

    Implementations MUST be idempotent and MUST return either a strictly
    smaller string (measured in tokens) or None to pass through. Never raise
    — errors should be caught internally and logged; the pipeline always
    tolerates a no-op interceptor.

    Interceptors MAY implement `progressive_disclosure_key()` to opt into
    one-shot behavior: the framework tracks which keys have already been
    rewritten in the current conversation, and skips subsequent matches on
    the same key so that the model gets full content if it asks again.
    """

    name: str  # e.g. "ast-grep", "difft", "scc"

    def matches(
        self,
        tool_name: str | None,
        tool_input: dict[str, Any],
        tool_output: str,
    ) -> bool: ...

    def transform(
        self,
        tool_name: str | None,
        tool_input: dict[str, Any],
        tool_output: str,
    ) -> str | None: ...

    def progressive_disclosure_key(
        self,
        tool_name: str | None,
        tool_input: dict[str, Any],
    ) -> str | None:
        """Optional: return a stable content key (e.g. file path).

        If a key is returned and the same (interceptor.name, key) pair was
        already successfully rewritten earlier in the messages, subsequent
        occurrences pass through unchanged. Return None to opt out.
        """
        ...


@dataclass(frozen=True)
class TransformSpan:
    """Per-interceptor measurement emitted for dashboard/metrics."""

    tool: str
    tokens_before: int
    tokens_after: int

    @property
    def tokens_saved(self) -> int:
        return max(self.tokens_before - self.tokens_after, 0)


@dataclass
class InterceptionResult:
    messages: list[dict[str, Any]]
    spans: list[TransformSpan]


INTERCEPTORS: list[ToolResultInterceptor] = []


def register(interceptor: ToolResultInterceptor) -> None:
    """Add an interceptor to the registry. Idempotent on name."""
    for existing in INTERCEPTORS:
        if existing.name == interceptor.name:
            return
    INTERCEPTORS.append(interceptor)


def _find_tool_use(
    messages: list[dict[str, Any]],
    tool_use_id: str,
) -> tuple[str | None, dict[str, Any]]:
    """Walk prior messages to find the tool_use block that produced a given id.

    Returns (tool_name, tool_input) or (None, {}) if not found.
    """
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                # Anthropic: {"type": "tool_use", "id": ..., "name": ..., "input": {...}}
                if block.get("type") == "tool_use" and block.get("id") == tool_use_id:
                    return (
                        block.get("name"),
                        block.get("input") or {},
                    )
        # OpenAI: assistant message with `tool_calls` list
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if isinstance(call, dict) and call.get("id") == tool_use_id:
                    fn = call.get("function") or {}
                    # arguments is a JSON string in OpenAI; decode best-effort
                    import json as _json

                    args: dict[str, Any] = {}
                    if isinstance(fn.get("arguments"), str):
                        try:
                            args = _json.loads(fn["arguments"])
                        except Exception:  # noqa: BLE001
                            args = {}
                    elif isinstance(fn.get("arguments"), dict):
                        args = fn["arguments"]
                    return fn.get("name"), args
    return None, {}


def _tool_use_id_for_message(msg: dict[str, Any]) -> str | None:
    """Return the tool_use_id linked to a tool_result message."""
    # Anthropic format
    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tuid = block.get("tool_use_id")
                if isinstance(tuid, str):
                    return tuid
    # OpenAI format
    if msg.get("role") == "tool":
        tcid = msg.get("tool_call_id")
        if isinstance(tcid, str):
            return tcid
    return None


def apply_to_messages(
    messages: list[dict[str, Any]],
    tokenizer: Tokenizer,
) -> InterceptionResult:
    """Run every registered interceptor against every tool_result in `messages`.

    Returns the (possibly) rewritten message list and a list of spans that
    actually saved tokens.
    """
    if not INTERCEPTORS:
        return InterceptionResult(messages=messages, spans=[])

    new_messages: list[dict[str, Any]] = []
    spans: list[TransformSpan] = []
    # Progressive disclosure: per-interceptor set of keys already rewritten
    # earlier in this message list. Prevents the second Read of the same
    # file from being outlined again — the model evidently came back for
    # more, so give it the raw content.
    fired: dict[str, set[str]] = {}

    for msg in messages:
        if not _is_tool_result_message(msg):
            new_messages.append(msg)
            continue

        original = _extract_tool_result_content(msg)
        if not isinstance(original, str) or not original:
            new_messages.append(msg)
            continue

        tuid = _tool_use_id_for_message(msg)
        tool_name: str | None = None
        tool_input: dict[str, Any] = {}
        if tuid:
            tool_name, tool_input = _find_tool_use(messages, tuid)

        current = original
        for interceptor in INTERCEPTORS:
            # Progressive disclosure: skip if already fired for this key.
            key: str | None = None
            key_fn = getattr(interceptor, "progressive_disclosure_key", None)
            if callable(key_fn):
                try:
                    key = key_fn(tool_name, tool_input)
                except Exception as e:  # noqa: BLE001
                    logger.warning("interceptor %s key() failed: %s", interceptor.name, e)
                    key = None
            if key and key in fired.get(interceptor.name, set()):
                continue

            try:
                if not interceptor.matches(tool_name, tool_input, current):
                    continue
                rewritten = interceptor.transform(tool_name, tool_input, current)
            except Exception as e:  # noqa: BLE001 — never crash a request
                logger.warning("interceptor %s failed: %s", interceptor.name, e)
                continue
            if not rewritten or rewritten == current:
                continue
            before = tokenizer.count_text(current)
            after = tokenizer.count_text(rewritten)
            if after >= before:
                continue  # refuse to enlarge
            spans.append(
                TransformSpan(
                    tool=interceptor.name,
                    tokens_before=before,
                    tokens_after=after,
                )
            )
            current = rewritten
            if key:
                fired.setdefault(interceptor.name, set()).add(key)

        new_messages.append(
            _swap_tool_result_content(msg, current) if current is not original else msg
        )

    return InterceptionResult(messages=new_messages, spans=spans)


class ToolResultInterceptorTransform(Transform):
    """Pipeline-level adapter: runs interceptors as the first compression stage.

    Placed at transforms[0] so downstream compressors operate on the already-
    shrunk content. Transform names of firing interceptors are added to
    `transforms_applied` so they appear in existing dashboards/metrics.
    """

    name = "tool_result_interceptors"

    def apply(
        self,
        messages: list[dict[str, Any]],
        tokenizer: Tokenizer,
        **kwargs: Any,
    ) -> TransformResult:
        result = apply_to_messages(messages, tokenizer)
        tokens_after = tokenizer.count_messages(result.messages)
        tokens_before = tokens_after + sum(s.tokens_saved for s in result.spans)
        transforms_applied = [f"interceptor:{s.tool}" for s in result.spans] if result.spans else []
        return TransformResult(
            messages=result.messages,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            transforms_applied=transforms_applied,
        )
