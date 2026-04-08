from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx

REPO_ROOT = Path("/workspace")
PLUGIN_DIR = REPO_ROOT / "plugins" / "openclaw"
SDK_DIR = REPO_ROOT / "sdk" / "typescript"
RTK_MARKER = "<!-- headroom:rtk-instructions -->"


def log(message: str) -> None:
    print(f"[wrap-e2e] {message}", flush=True)


def run(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: int = 180,
) -> subprocess.CompletedProcess[str]:
    log(f"$ {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        env=env,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    if result.stdout.strip():
        print(result.stdout.rstrip(), flush=True)
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")
    return result


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    current_mode = path.stat().st_mode
    path.chmod(current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class MockOpenAIServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, MockOpenAIHandler)
        self.requests: list[dict[str, Any]] = []


class MockOpenAIHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _write_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self, body: dict[str, Any] | None = None) -> None:
        server = self.server
        assert isinstance(server, MockOpenAIServer)
        server.requests.append(
            {
                "method": self.command,
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": body,
            }
        )

    def do_GET(self) -> None:  # noqa: N802
        self._record()
        if self.path == "/v1/models":
            self._write_json(
                200,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": "gpt-4o-mini",
                            "object": "model",
                            "owned_by": "openai",
                        }
                    ],
                },
            )
            return
        self._write_json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length else b""
        payload = json.loads(raw_body.decode("utf-8") or "{}")
        self._record(body=payload)
        if self.path == "/v1/chat/completions":
            self._write_json(
                200,
                {
                    "id": "chatcmpl-e2e",
                    "object": "chat.completion",
                    "created": 0,
                    "model": payload.get("model", "gpt-4o-mini"),
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "mock completion from upstream",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 5,
                        "total_tokens": 17,
                    },
                },
            )
            return
        self._write_json(404, {"error": {"message": "not found"}})


def wait_for_http(url: str, *, timeout: int = 30) -> httpx.Response:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code < 500:
                return response
        except Exception as exc:  # pragma: no cover - best effort retry surface
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}: {last_error}")


def wait_for_output(proc: subprocess.Popen[str], text: str, *, timeout: int = 30) -> str:
    deadline = time.time() + timeout
    chunks: list[str] = []
    while time.time() < deadline:
        if proc.stdout is None:
            break
        line = proc.stdout.readline()
        if line:
            chunks.append(line)
            if text in "".join(chunks):
                return "".join(chunks)
        elif proc.poll() is not None:
            break
    output = "".join(chunks)
    raise RuntimeError(f"Timed out waiting for process output '{text}'. Output so far:\n{output}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def create_shims(shim_dir: Path) -> None:
    generic_shim = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        from __future__ import annotations

        import json
        import os
        import sys
        from pathlib import Path

        tool = Path(sys.argv[0]).name
        log_dir = Path(os.environ["HEADROOM_E2E_LOG_DIR"])
        log_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "tool": tool,
            "argv": sys.argv[1:],
            "cwd": os.getcwd(),
            "env": {
                key: os.environ.get(key)
                for key in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "ANTHROPIC_BASE_URL")
                if os.environ.get(key) is not None
            },
        }
        with (log_dir / f"{tool}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\\n")
        print(f"{tool} shim executed")
        raise SystemExit(0)
        """
    )
    rtk_shim = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        from __future__ import annotations

        import sys

        if "--version" in sys.argv:
            print("rtk e2e-shim")
        else:
            print("rtk shim")
        raise SystemExit(0)
        """
    )
    openclaw_shim = textwrap.dedent(
        """\
        #!/usr/bin/env python3
        from __future__ import annotations

        import json
        import os
        import sys
        from pathlib import Path

        state_path = Path(os.environ["HEADROOM_E2E_OPENCLAW_STATE"])
        config_path = Path(os.environ["HEADROOM_E2E_OPENCLAW_CONFIG"])
        log_dir = Path(os.environ["HEADROOM_E2E_LOG_DIR"])
        log_dir.mkdir(parents=True, exist_ok=True)

        def default_state() -> dict:
            return {
                "plugins": {
                    "entries": {},
                    "slots": {"contextEngine": "legacy"},
                },
                "installs": [],
                "gateway_actions": [],
            }

        def load_state() -> dict:
            if state_path.exists():
                return json.loads(state_path.read_text(encoding="utf-8"))
            state = default_state()
            save_state(state)
            return state

        def save_state(state: dict) -> None:
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(state, indent=2) + "\\n", encoding="utf-8")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(state, indent=2) + "\\n", encoding="utf-8")

        def log_call(args: list[str]) -> None:
            record = {
                "tool": "openclaw",
                "argv": args,
                "cwd": os.getcwd(),
            }
            with (log_dir / "openclaw.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\\n")

        def get_nested(container: dict, dotted_path: str):
            current = container
            for key in dotted_path.split("."):
                if not isinstance(current, dict) or key not in current:
                    return None
                current = current[key]
            return current

        def set_nested(container: dict, dotted_path: str, value) -> None:
            keys = dotted_path.split(".")
            current = container
            for key in keys[:-1]:
                current = current.setdefault(key, {})
            current[keys[-1]] = value

        args = sys.argv[1:]
        if not args or args[0] in {"--help", "-h", "help"}:
            print("openclaw shim")
            raise SystemExit(0)

        log_call(args)
        state = load_state()
        extensions_dir = config_path.parent / "extensions"

        if args[:2] == ["config", "file"]:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.touch()
            print(str(config_path))
            raise SystemExit(0)

        if args[:2] == ["config", "get"]:
            print(json.dumps(get_nested(state, args[2])))
            raise SystemExit(0)

        if args[:2] == ["config", "set"]:
            set_nested(state, args[2], json.loads(args[3]))
            save_state(state)
            print("ok")
            raise SystemExit(0)

        if args[:2] == ["config", "validate"]:
            print("valid")
            raise SystemExit(0)

        if args[:2] == ["gateway", "restart"] or args[:2] == ["gateway", "start"]:
            state["gateway_actions"].append(args[1])
            save_state(state)
            print(args[1])
            raise SystemExit(0)

        if args[:2] == ["plugins", "install"]:
            state["installs"].append({"argv": args, "cwd": os.getcwd()})
            save_state(state)
            (extensions_dir / "headroom").mkdir(parents=True, exist_ok=True)
            print("installed")
            raise SystemExit(0)

        if args[:3] == ["plugins", "inspect", "headroom"]:
            print(
                json.dumps(
                    {
                        "id": "headroom",
                        "installed": True,
                        "entry": get_nested(state, "plugins.entries.headroom"),
                    }
                )
            )
            raise SystemExit(0)

        print(f"unsupported openclaw args: {args}", file=sys.stderr)
        raise SystemExit(2)
        """
    )

    write_executable(shim_dir / "codex", generic_shim)
    write_executable(shim_dir / "aider", generic_shim)
    write_executable(shim_dir / "rtk", rtk_shim)
    write_executable(shim_dir / "openclaw", openclaw_shim)


def start_mock_server(port: int) -> tuple[MockOpenAIServer, threading.Thread]:
    server = MockOpenAIServer(("127.0.0.1", port))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def start_proxy(port: int, env: dict[str, str]) -> subprocess.Popen[str]:
    log(f"Starting headroom proxy on port {port}")
    proc = subprocess.Popen(
        ["headroom", "proxy", "--host", "127.0.0.1", "--port", str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    wait_for_http(f"http://127.0.0.1:{port}/health", timeout=30)
    return proc


def stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate(timeout=5)


def verify_installs() -> None:
    log("Verifying installed packages and binaries")
    for tool in ("headroom", "codex", "aider", "openclaw"):
        assert_true(shutil.which(tool) is not None, f"Expected '{tool}' on PATH")
    run(["headroom", "--help"], timeout=30)
    run(["npm", "list", "-g", "--depth=0", "@openai/codex", "openclaw"], timeout=60)
    run(["/opt/aider-venv/bin/python", "-m", "pip", "show", "aider-chat"], timeout=60)


def prepare_local_openclaw_plugin(base_env: dict[str, str], tmp_dir: Path) -> Path:
    log("Preparing local TypeScript package for OpenClaw plugin build")
    sdk_dir = tmp_dir / "sdk-typescript"
    plugin_dir = tmp_dir / "openclaw-plugin"
    shutil.copytree(SDK_DIR, sdk_dir)
    shutil.copytree(PLUGIN_DIR, plugin_dir)

    plugin_lock = plugin_dir / "package-lock.json"
    if plugin_lock.exists():
        plugin_lock.unlink()

    run(["npm", "install"], env=base_env, cwd=sdk_dir, timeout=600)
    run(["npm", "run", "build"], env=base_env, cwd=sdk_dir, timeout=600)
    pack_result = run(["npm", "pack"], env=base_env, cwd=sdk_dir, timeout=600)
    tarball_name = pack_result.stdout.strip().splitlines()[-1].strip()
    tarball_path = sdk_dir / tarball_name
    assert_true(tarball_path.exists(), "Expected npm pack to produce a local SDK tarball")

    package_json_path = plugin_dir / "package.json"
    package_json = json.loads(package_json_path.read_text(encoding="utf-8"))
    package_json["dependencies"]["headroom-ai"] = f"file:{tarball_path.as_posix()}"
    package_json_path.write_text(f"{json.dumps(package_json, indent=2)}\n", encoding="utf-8")

    return plugin_dir


def verify_proxy_round_trip(base_env: dict[str, str], mock_server: MockOpenAIServer) -> None:
    proxy_port = 18787
    proc = start_proxy(proxy_port, base_env)
    try:
        health = wait_for_http(f"http://127.0.0.1:{proxy_port}/health")
        assert_true(health.status_code == 200, "Proxy health check should return 200")

        models = httpx.get(
            f"http://127.0.0.1:{proxy_port}/v1/models",
            headers={"Authorization": "Bearer test-key"},
            timeout=10.0,
        )
        assert_true(models.status_code == 200, "Proxy should pass through /v1/models")
        assert_true(models.json()["data"][0]["id"] == "gpt-4o-mini", "Unexpected models payload")

        chat = httpx.post(
            f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Say hello."}],
            },
            timeout=10.0,
        )
        assert_true(chat.status_code == 200, "Proxy should pass through chat completions")
        assert_true(
            chat.json()["choices"][0]["message"]["content"] == "mock completion from upstream",
            "Unexpected chat completion payload",
        )
        assert_true(
            any(item["path"] == "/v1/models" for item in mock_server.requests),
            "Mock upstream should receive /v1/models",
        )
        assert_true(
            any(item["path"] == "/v1/chat/completions" for item in mock_server.requests),
            "Mock upstream should receive /v1/chat/completions",
        )
    finally:
        stop_process(proc)


def verify_codex_wrap(base_env: dict[str, str], project_dir: Path, log_dir: Path) -> None:
    port = 18788
    run(
        ["headroom", "wrap", "codex", "--port", str(port), "--", "--help"],
        env=base_env,
        cwd=project_dir,
        timeout=120,
    )
    project_agents = project_dir / "AGENTS.md"
    global_agents = Path(base_env["HOME"]) / ".codex" / "AGENTS.md"
    assert_true(project_agents.exists(), "Codex wrap should create project AGENTS.md")
    assert_true(global_agents.exists(), "Codex wrap should create ~/.codex/AGENTS.md")
    assert_true(RTK_MARKER in project_agents.read_text(encoding="utf-8"), "Missing RTK marker")
    assert_true(
        RTK_MARKER in global_agents.read_text(encoding="utf-8"), "Missing global RTK marker"
    )

    entries = read_jsonl(log_dir / "codex.jsonl")
    assert_true(len(entries) > 0, "Codex shim should have been invoked")
    env_vars = entries[-1]["env"]
    assert_true(
        env_vars.get("OPENAI_BASE_URL") == f"http://127.0.0.1:{port}/v1",
        "Codex wrap should set OPENAI_BASE_URL",
    )


def verify_aider_wrap(base_env: dict[str, str], project_dir: Path, log_dir: Path) -> None:
    port = 18789
    run(
        ["headroom", "wrap", "aider", "--port", str(port), "--", "--help"],
        env=base_env,
        cwd=project_dir,
        timeout=120,
    )
    conventions = project_dir / "CONVENTIONS.md"
    assert_true(conventions.exists(), "Aider wrap should create CONVENTIONS.md")
    assert_true(
        RTK_MARKER in conventions.read_text(encoding="utf-8"),
        "Aider wrap should inject RTK instructions",
    )

    entries = read_jsonl(log_dir / "aider.jsonl")
    assert_true(len(entries) > 0, "Aider shim should have been invoked")
    env_vars = entries[-1]["env"]
    assert_true(
        env_vars.get("OPENAI_API_BASE") == f"http://127.0.0.1:{port}/v1",
        "Aider wrap should set OPENAI_API_BASE",
    )
    assert_true(
        env_vars.get("ANTHROPIC_BASE_URL") == f"http://127.0.0.1:{port}",
        "Aider wrap should set ANTHROPIC_BASE_URL",
    )


def verify_cursor_wrap(base_env: dict[str, str], project_dir: Path) -> None:
    port = 18790
    proc = subprocess.Popen(
        ["headroom", "wrap", "cursor", "--port", str(port)],
        env=base_env,
        cwd=str(project_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        output = wait_for_output(proc, "Press Ctrl+C to stop the proxy.", timeout=30)
        assert_true(
            f"http://127.0.0.1:{port}/v1" in output,
            "Cursor wrap should print the OpenAI base URL override",
        )
        wait_for_http(f"http://127.0.0.1:{port}/health", timeout=15)
        cursorrules = project_dir / ".cursorrules"
        assert_true(cursorrules.exists(), "Cursor wrap should create .cursorrules")
        assert_true(
            RTK_MARKER in cursorrules.read_text(encoding="utf-8"),
            "Cursor wrap should inject RTK instructions",
        )
    finally:
        stop_process(proc)


def verify_openclaw_wrap(
    base_env: dict[str, str],
    project_dir: Path,
    plugin_dir: Path,
    state_path: Path,
) -> None:
    run(
        [
            "headroom",
            "wrap",
            "openclaw",
            "--plugin-path",
            str(plugin_dir),
            "--proxy-port",
            "18791",
            "--startup-timeout-ms",
            "5000",
            "--verbose",
        ],
        env=base_env,
        cwd=project_dir,
        timeout=600,
    )
    dist_index = plugin_dir / "dist" / "index.js"
    assert_true(dist_index.exists(), "OpenClaw plugin build should produce dist/index.js")

    state = json.loads(state_path.read_text(encoding="utf-8"))
    entry = state["plugins"]["entries"]["headroom"]
    assert_true(entry["enabled"] is True, "OpenClaw wrap should enable the plugin")
    assert_true(entry["config"]["proxyPort"] == 18791, "OpenClaw wrap should set proxy port")
    assert_true(entry["config"]["autoStart"] is True, "OpenClaw wrap should enable autoStart")
    assert_true(
        state["plugins"]["slots"]["contextEngine"] == "headroom",
        "OpenClaw wrap should set the context engine slot",
    )
    assert_true(
        state["gateway_actions"] == ["restart"],
        "OpenClaw wrap should restart the gateway once",
    )


def main() -> None:
    verify_installs()
    with tempfile.TemporaryDirectory(prefix="headroom-wrap-e2e-") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        home_dir = tmp_dir / "home"
        project_dir = tmp_dir / "project"
        shim_dir = tmp_dir / "shim-bin"
        log_dir = tmp_dir / "logs"
        openclaw_state_path = tmp_dir / "openclaw" / "state.json"
        openclaw_config_path = tmp_dir / "openclaw" / "config" / "config.json"

        for path in (home_dir, project_dir, shim_dir, log_dir):
            path.mkdir(parents=True, exist_ok=True)
        create_shims(shim_dir)

        mock_server, mock_thread = start_mock_server(19001)
        base_env = os.environ.copy()
        base_env.update(
            {
                "HOME": str(home_dir),
                "PATH": f"{shim_dir}{os.pathsep}{base_env['PATH']}",
                "HEADROOM_E2E_LOG_DIR": str(log_dir),
                "HEADROOM_E2E_OPENCLAW_STATE": str(openclaw_state_path),
                "HEADROOM_E2E_OPENCLAW_CONFIG": str(openclaw_config_path),
                "OPENAI_TARGET_API_URL": "http://127.0.0.1:19001/v1",
            }
        )

        try:
            verify_proxy_round_trip(base_env, mock_server)
            verify_codex_wrap(base_env, project_dir, log_dir)
            verify_aider_wrap(base_env, project_dir, log_dir)
            verify_cursor_wrap(base_env, project_dir)
            local_plugin_dir = prepare_local_openclaw_plugin(base_env, tmp_dir)
            verify_openclaw_wrap(base_env, project_dir, local_plugin_dir, openclaw_state_path)
        finally:
            mock_server.shutdown()
            mock_thread.join(timeout=5)

    log("All Docker wrap e2e checks passed.")


if __name__ == "__main__":
    main()
