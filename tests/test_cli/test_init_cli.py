from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import click
from click.testing import CliRunner


def _load_init_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "headroom.cli.init", raising=False)
    monkeypatch.delitem(sys.modules, "headroom.cli.main", raising=False)
    fake_main_module = types.ModuleType("headroom.cli.main")

    @click.group()
    def fake_main() -> None:
        pass

    fake_main_module.main = fake_main
    monkeypatch.setitem(sys.modules, "headroom.cli.main", fake_main_module)
    importlib.invalidate_caches()
    init_cli = importlib.import_module("headroom.cli.init")
    monkeypatch.delitem(sys.modules, "headroom.cli.init", raising=False)
    return init_cli, fake_main


def test_init_auto_detects_targets(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()
    captured: dict[str, object] = {}

    monkeypatch.setattr(init_cli, "detect_init_targets", lambda global_scope: ["claude", "codex"])
    monkeypatch.setattr(init_cli, "_run_init_targets", lambda **kwargs: captured.update(kwargs))

    result = runner.invoke(fake_main, ["init", "-g"])

    assert result.exit_code == 0, result.output
    assert captured["targets"] == ["claude", "codex"]
    assert captured["global_scope"] is True


def test_init_fails_when_auto_detection_empty(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()
    monkeypatch.setattr(init_cli, "detect_init_targets", lambda global_scope: [])

    result = runner.invoke(fake_main, ["init"])

    assert result.exit_code != 0
    assert "auto-detected" in result.output


def test_init_copilot_requires_global(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()
    monkeypatch.setattr(init_cli, "_ensure_runtime_manifest", lambda **kwargs: "init-local-test")

    result = runner.invoke(fake_main, ["init", "copilot"])

    assert result.exit_code != 0
    assert "requires -g" in result.output


def test_init_claude_local_writes_settings_and_installs_marketplace(
    monkeypatch, tmp_path: Path
) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    marketplace_calls: list[str] = []
    monkeypatch.setattr(init_cli, "_ensure_runtime_manifest", lambda **kwargs: "init-local-demo")
    monkeypatch.setattr(
        init_cli,
        "_install_claude_marketplace",
        lambda scope: marketplace_calls.append(scope),
    )

    result = runner.invoke(fake_main, ["init", "claude"])

    assert result.exit_code == 0, result.output
    settings_path = tmp_path / ".claude" / "settings.local.json"
    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8787"
    assert marketplace_calls == ["local"]
    assert any(
        "--profile init-local-demo" in hook["command"] and "init hook ensure" in hook["command"]
        for entry in payload["hooks"]["SessionStart"]
        for hook in entry["hooks"]
    )


def test_init_codex_merges_feature_flag_into_existing_table(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / ".codex" / "config.toml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("[features]\nshell_tool = true\n", encoding="utf-8")

    init_cli._init_codex(global_scope=False, profile="init-local-demo", port=9000)

    content = config_path.read_text(encoding="utf-8")
    assert 'base_url = "http://127.0.0.1:9000/v1"' in content
    assert content.count("[features]") == 1
    assert "codex_hooks = true" in content
    hooks = json.loads((tmp_path / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    assert "--profile init-local-demo" in hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]
    assert "init hook ensure" in hooks["hooks"]["SessionStart"][0]["hooks"][0]["command"]


def test_init_claude_uses_custom_port(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(init_cli, "_install_claude_marketplace", lambda scope: None)

    init_cli._init_claude(global_scope=False, profile="init-local-demo", port=9011)

    payload = json.loads((tmp_path / ".claude" / "settings.local.json").read_text(encoding="utf-8"))
    assert payload["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:9011"


def test_init_copilot_global_writes_hooks_and_env(monkeypatch, tmp_path: Path) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    captured_env: dict[str, str] = {}
    monkeypatch.setattr(init_cli, "_copilot_config_path", lambda: tmp_path / "copilot-config.json")
    monkeypatch.setattr(init_cli, "_apply_user_env", lambda values: captured_env.update(values))
    monkeypatch.setattr(init_cli, "_install_copilot_marketplace", lambda: None)

    init_cli._init_copilot(global_scope=True, profile="init-user", port=9005, backend="openai")

    payload = json.loads((tmp_path / "copilot-config.json").read_text(encoding="utf-8"))
    assert "SessionStart" in payload["hooks"]
    assert "PreToolUse" in payload["hooks"]
    assert "--profile init-user" in payload["hooks"]["SessionStart"][0]["command"]
    assert captured_env == {
        "COPILOT_PROVIDER_TYPE": "openai",
        "COPILOT_PROVIDER_BASE_URL": "http://127.0.0.1:9005/v1",
        "COPILOT_PROVIDER_WIRE_API": "completions",
    }


def test_init_hook_ensure_prefers_local_profile(monkeypatch) -> None:
    init_cli, fake_main = _load_init_module(monkeypatch)
    ensured: list[str] = []

    def fake_load(profile: str):
        return object() if profile == "init-repo-12345678" else None

    monkeypatch.setattr(init_cli, "_local_profile", lambda cwd=None: "init-repo-12345678")
    monkeypatch.setattr(init_cli, "load_manifest", fake_load)
    monkeypatch.setattr(
        init_cli, "_ensure_profile_running", lambda profile: ensured.append(profile)
    )

    runner = CliRunner()
    result = runner.invoke(fake_main, ["init", "hook", "ensure"])

    assert result.exit_code == 0, result.output
    assert ensured == ["init-repo-12345678"]


def test_init_openclaw_requires_global(monkeypatch) -> None:
    _, fake_main = _load_init_module(monkeypatch)
    runner = CliRunner()

    result = runner.invoke(fake_main, ["init", "openclaw"])

    assert result.exit_code != 0
    assert "requires -g" in result.output


def test_init_openclaw_delegates_to_wrap(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    calls: list[list[str]] = []

    class _Result:
        returncode = 0

    monkeypatch.setattr(init_cli, "resolve_headroom_command", lambda: ["headroom"])
    monkeypatch.setattr(
        init_cli.subprocess,
        "run",
        lambda cmd: calls.append(cmd) or _Result(),
    )

    init_cli._init_openclaw(global_scope=True, port=9999)

    assert calls == [["headroom", "wrap", "openclaw", "--proxy-port", "9999"]]


def test_detect_init_targets_respects_scope(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setattr(
        init_cli.shutil,
        "which",
        lambda name: name if name in {"claude", "copilot", "codex", "openclaw"} else None,
    )

    assert init_cli.detect_init_targets(False) == ["claude", "codex"]
    assert init_cli.detect_init_targets(True) == ["claude", "copilot", "codex", "openclaw"]


def test_marketplace_source_prefers_env_override(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)
    monkeypatch.setenv("HEADROOM_MARKETPLACE_SOURCE", "custom/source")

    assert init_cli._marketplace_source() == "custom/source"


def test_run_checked_treats_existing_install_as_success(monkeypatch) -> None:
    init_cli, _ = _load_init_module(monkeypatch)

    class _Result:
        returncode = 1
        stderr = "plugin already exists"
        stdout = ""

    monkeypatch.setattr(init_cli.subprocess, "run", lambda *args, **kwargs: _Result())

    init_cli._run_checked(["claude", "plugin", "install"], action="claude plugin install")
