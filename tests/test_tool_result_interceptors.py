"""Tests for the tool_result interceptor framework + ast-grep Read outliner."""

from __future__ import annotations

import textwrap

import pytest

from headroom.proxy.interceptors import (
    INTERCEPTORS,
    ToolResultInterceptor,
    apply_to_messages,
    register,
)
from headroom.proxy.interceptors.astgrep import AstGrepReadOutline
from headroom.tokenizer import Tokenizer


class _FakeTokenCounter:
    """Deterministic 4-chars-per-token counter for unit tests."""

    def count_text(self, text: str) -> int:
        return max(1, len(text) // 4)

    def count_messages(self, messages) -> int:
        total = 0
        for m in messages:
            c = m.get("content")
            if isinstance(c, str):
                total += self.count_text(c)
            elif isinstance(c, list):
                for b in c:
                    if isinstance(b, dict):
                        inner = b.get("content") or b.get("text") or ""
                        if isinstance(inner, str):
                            total += self.count_text(inner)
        return total


@pytest.fixture
def tokenizer() -> Tokenizer:
    # Real Tokenizer wrapping the fake counter; mirrors production construction.
    return Tokenizer(_FakeTokenCounter())  # type: ignore[arg-type]


# -------- Framework basics ----------------------------------------------- #


def test_astgrep_interceptor_registered_by_default():
    assert any(i.name == "ast-grep" for i in INTERCEPTORS)


def test_register_is_idempotent_on_name():
    before = len(INTERCEPTORS)
    register(AstGrepReadOutline())  # same name
    assert len(INTERCEPTORS) == before


def test_custom_interceptor_plugs_in(tokenizer):
    class UpperCase:
        name = "uppercase-test"

        def matches(self, tool_name, tool_input, tool_output):
            return tool_name == "Echo"

        def transform(self, tool_name, tool_input, tool_output):
            # Must REDUCE tokens — use a single short marker.
            return "X"

    dummy: ToolResultInterceptor = UpperCase()  # type: ignore[assignment]
    register(dummy)
    try:
        messages = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "1", "name": "Echo", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "1",
                        "content": "hello " * 100,
                    }
                ],
            },
        ]
        result = apply_to_messages(messages, tokenizer)
        assert any(s.tool == "uppercase-test" for s in result.spans)
        swapped = result.messages[1]["content"][0]["content"]
        assert swapped == "X"
    finally:
        INTERCEPTORS[:] = [i for i in INTERCEPTORS if i.name != "uppercase-test"]


def test_pass_through_when_no_interceptor_matches(tokenizer):
    messages = [
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "1", "name": "Unknown", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "1", "content": "x" * 5000}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    assert result.spans == []
    assert result.messages[1] is messages[1]  # untouched identity


# -------- ast-grep interceptor ------------------------------------------- #


_PY_FIXTURE = textwrap.dedent(
    '''
    """Payments module fixture."""
    from decimal import Decimal

    def compute_subtotal(items):
        total = Decimal("0")
        for item in items:
            total += item.price * item.qty
        return total


    def apply_promo(subtotal, code):
        if not code:
            return subtotal
        if code == "SAVE10":
            return subtotal * Decimal("0.9")
        return subtotal


    def compute_tax(subtotal, rate):
        return (subtotal * rate).quantize(Decimal("0.01"))


    def process_payment(items, promo, tax_rate):
        """Main entry point."""
        subtotal = compute_subtotal(items)
        after = apply_promo(subtotal, promo)
        tax = compute_tax(after, tax_rate)
        return after + tax


    def refund(order_id, amount):
        """Issue a refund."""
        return {"order": order_id, "refund": str(amount)}


    def list_orders_for_user(user_id, limit=20):
        """Placeholder DB lookup for a user's orders."""
        return [{"user": user_id, "order": i} for i in range(limit)]


    def cancel_order(order_id, reason=None):
        """Cancel an order, logging the reason if provided."""
        return {"order": order_id, "cancelled": True, "reason": reason or "unspecified"}


    def summarize_cart(items):
        """Return a one-line summary of cart contents."""
        skus = [i.sku for i in items]
        total_qty = sum(i.qty for i in items)
        return f"{len(items)} line items ({total_qty} units): {', '.join(skus)}"


    def format_receipt(order_id, items, total):
        """Render a textual receipt."""
        lines = [f"Order {order_id}"]
        for i in items:
            lines.append(f"  {i.sku} x {i.qty} @ {i.unit_price} = {i.qty * i.unit_price}")
        lines.append(f"Total: {total}")
        return "\\n".join(lines)
    '''
).strip()


def test_astgrep_outlines_large_python_read(tokenizer):
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "abc",
                    "name": "Read",
                    "input": {"file_path": "/repo/payments.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "abc", "content": _PY_FIXTURE}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    assert len(result.spans) == 1
    span = result.spans[0]
    assert span.tool == "ast-grep"
    assert span.tokens_after < span.tokens_before
    new_content = result.messages[1]["content"][0]["content"]
    assert "outlined by ast-grep" in new_content
    assert "body elided" in new_content
    assert "def process_payment" in new_content
    assert "def apply_promo" in new_content
    # Bodies should NOT leak through unchanged.
    assert "total += item.price * item.qty" not in new_content


def test_astgrep_skips_small_files(tokenizer):
    small = "def foo(): return 1\n"
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "x",
                    "name": "Read",
                    "input": {"file_path": "/a.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "x", "content": small}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    assert result.spans == []


def test_astgrep_skips_non_code_extensions(tokenizer):
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r",
                    "name": "Read",
                    "input": {"file_path": "/notes.txt"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "r", "content": "x" * 3000}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    assert result.spans == []


# -------- OpenAI-format tool_result -------------------------------------- #


def test_astgrep_skips_when_line_range_requested(tokenizer):
    """If the tool_input specifies a line range, the model wants those lines — pass through."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "r",
                    "name": "Read",
                    "input": {
                        "file_path": "/repo/payments.py",
                        "offset": 30,
                        "limit": 20,
                    },
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "r", "content": _PY_FIXTURE}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    assert result.spans == []


def test_progressive_disclosure_second_read_passes_through(tokenizer):
    """First Read of a file gets outlined; second Read of the same path is untouched."""
    messages = [
        # Turn 1: Read foo.py → outlined
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Read",
                    "input": {"file_path": "/repo/payments.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": _PY_FIXTURE}],
        },
        # Turn 2: Read foo.py again (model came back for more) → pass through
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "Read",
                    "input": {"file_path": "/repo/payments.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t2", "content": _PY_FIXTURE}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    # Only the first Read is rewritten; the second keeps its full body.
    assert len(result.spans) == 1
    first_tr = result.messages[1]["content"][0]["content"]
    second_tr = result.messages[3]["content"][0]["content"]
    assert "outlined by ast-grep" in first_tr
    assert "outlined by ast-grep" not in second_tr
    assert "def process_payment" in second_tr
    # Second Read preserves the bodies.
    assert "subtotal = compute_subtotal(items)" in second_tr


def test_progressive_disclosure_different_file_still_outlined(tokenizer):
    """Reading a DIFFERENT file after the first outline should still outline."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "Read",
                    "input": {"file_path": "/repo/payments.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": _PY_FIXTURE}],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "t2",
                    "name": "Read",
                    "input": {"file_path": "/repo/other.py"},
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t2", "content": _PY_FIXTURE}],
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    # Both files get outlined — different keys.
    assert len(result.spans) == 2


def test_openai_format_tool_result_is_rewritten(tokenizer):
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "Read",
                        "arguments": '{"file_path": "/x/payments.py"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": _PY_FIXTURE,
        },
    ]
    result = apply_to_messages(messages, tokenizer)
    assert len(result.spans) == 1
    new_content = result.messages[1]["content"]
    assert "outlined by ast-grep" in new_content
