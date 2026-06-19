"""Tests for the reasoning/thinking-budget guard (_apply_thinking_policy).

Models with ``disable_thinking_below_max_tokens`` set (e.g. DiffusionGemma)
get ``chat_template_kwargs.enable_thinking=false`` injected on small-budget
chat-completion requests, so the answer channel isn't starved to empty content
by an unbounded thinking chain. Generous/unset budgets, other models, and
non-chat paths are left byte-for-byte untouched.
"""

import json

import pytest

from router.app import _apply_thinking_policy
from router.config import ModelSpec

CHAT = "/v1/chat/completions"


class _FakeManager:
    def __init__(self, specs):
        self.index = {s.id: s for s in specs}


@pytest.fixture
def manager():
    return _FakeManager(
        [
            ModelSpec(
                id="dg",
                engine="diffusiongemma",
                display_name="dg",
                disable_thinking_below_max_tokens=1024,
            ),
            ModelSpec(id="plain", engine="ds4", display_name="plain"),
        ]
    )


def _run(manager, model, body, path=CHAT):
    raw = json.dumps(body).encode()
    out = _apply_thinking_policy(manager, model, path, dict(body), raw)
    return json.loads(out)


def test_small_budget_disables_thinking(manager):
    out = _run(manager, "dg", {"model": "dg", "max_tokens": 100})
    assert out["chat_template_kwargs"] == {"enable_thinking": False}


def test_small_budget_via_max_completion_tokens(manager):
    out = _run(manager, "dg", {"model": "dg", "max_completion_tokens": 1023})
    assert out["chat_template_kwargs"] == {"enable_thinking": False}


def test_generous_budget_left_untouched(manager):
    body = {"model": "dg", "max_tokens": 4096}
    out = _run(manager, "dg", body)
    assert "chat_template_kwargs" not in out


def test_threshold_is_exclusive(manager):
    # Exactly at the threshold keeps thinking on (>= threshold => quality path).
    out = _run(manager, "dg", {"model": "dg", "max_tokens": 1024})
    assert "chat_template_kwargs" not in out


def test_no_budget_left_untouched(manager):
    # No max_tokens => full context available, no starvation risk.
    out = _run(manager, "dg", {"model": "dg"})
    assert "chat_template_kwargs" not in out


def test_explicit_client_choice_respected(manager):
    # Client explicitly asked for thinking on with a tiny budget — respect it.
    body = {
        "model": "dg",
        "max_tokens": 50,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    out = _run(manager, "dg", body)
    assert out["chat_template_kwargs"] == {"enable_thinking": True}


def test_explicit_other_kwargs_preserved(manager):
    # Other chat_template_kwargs are preserved when we add enable_thinking.
    body = {"model": "dg", "max_tokens": 50, "chat_template_kwargs": {"foo": "bar"}}
    out = _run(manager, "dg", body)
    assert out["chat_template_kwargs"] == {"foo": "bar", "enable_thinking": False}


def test_model_without_policy_untouched(manager):
    out = _run(manager, "plain", {"model": "plain", "max_tokens": 50})
    assert "chat_template_kwargs" not in out


def test_unknown_model_untouched(manager):
    out = _run(manager, "ghost", {"model": "ghost", "max_tokens": 50})
    assert "chat_template_kwargs" not in out


def test_non_chat_path_untouched(manager):
    out = _run(manager, "dg", {"model": "dg", "max_tokens": 50}, path="/v1/completions")
    assert "chat_template_kwargs" not in out


def test_bool_budget_is_not_treated_as_int(manager):
    # bool is an int subclass; it must not count as a real token budget.
    out = _run(manager, "dg", {"model": "dg", "max_tokens": True})
    assert "chat_template_kwargs" not in out


def test_bytes_unchanged_when_no_policy_fires(manager):
    # When the policy is a no-op, the exact original bytes are returned.
    body = {"model": "dg", "max_tokens": 4096}
    raw = json.dumps(body).encode()
    out = _apply_thinking_policy(manager, "dg", CHAT, dict(body), raw)
    assert out is raw


def test_float_budget_is_treated_as_int(manager):
    # JSON/JS clients may send max_tokens as 500.0 — the policy must still fire.
    out = _run(manager, "dg", {"model": "dg", "max_tokens": 500.0})
    assert out["chat_template_kwargs"] == {"enable_thinking": False}


def test_float_budget_above_threshold_untouched(manager):
    out = _run(manager, "dg", {"model": "dg", "max_tokens": 4096.0})
    assert "chat_template_kwargs" not in out


def test_composes_with_alias_rewrite():
    # The policy must compose with alias rewriting: the final body carries BOTH
    # the rewritten real model id AND the injected enable_thinking=false, with
    # the rest of the body (messages) preserved.
    from router.app import _resolve_alias_and_rewrite

    class _AliasManager:
        def __init__(self):
            self.index = {
                "real-dg": ModelSpec(
                    id="real-dg",
                    engine="diffusiongemma",
                    display_name="dg",
                    disable_thinking_below_max_tokens=1024,
                ),
            }
            self._aliases = {"dg": "real-dg"}

        def resolve_model_id(self, mid):
            return self._aliases.get(mid, mid)

    mgr = _AliasManager()
    body = {
        "model": "dg",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}],
    }
    raw = json.dumps(body).encode()
    model, raw = _resolve_alias_and_rewrite(mgr, "dg", body, raw)
    raw = _apply_thinking_policy(mgr, model, CHAT, body, raw)
    out = json.loads(raw)
    assert model == "real-dg"
    assert out["model"] == "real-dg"
    assert out["chat_template_kwargs"] == {"enable_thinking": False}
    assert out["messages"] == [{"role": "user", "content": "hi"}]


def test_config_rejects_nonpositive_threshold(tmp_path):
    from router.config import ConfigError, load_config

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "engines:\n"
        "  e:\n"
        "    type: ollama\n"
        "    base_url: http://127.0.0.1:11434\n"
        "models:\n"
        "  - id: m\n"
        "    engine: e\n"
        "    disable_thinking_below_max_tokens: 0\n"
    )
    with pytest.raises(ConfigError):
        load_config(str(cfg_file))


def test_config_accepts_valid_threshold(tmp_path):
    from router.config import build_model_index, load_config

    cfg_file = tmp_path / "c.yaml"
    cfg_file.write_text(
        "engines:\n"
        "  e:\n"
        "    type: ollama\n"
        "    base_url: http://127.0.0.1:11434\n"
        "models:\n"
        "  - id: m\n"
        "    engine: e\n"
        "    disable_thinking_below_max_tokens: 1024\n"
    )
    cfg = load_config(str(cfg_file))
    assert build_model_index(cfg)["m"].disable_thinking_below_max_tokens == 1024
