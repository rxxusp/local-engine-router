"""Interactive setup wizard for local-engine-router (``init`` subcommand).

The goal is "answer a few prompts, get a running router." The wizard probes
the well-known localhost ports of the supported backends (an independent probe;
the reserved ``discover.port_probe`` config flag is not read here), confirms
what is actually listening, fetches each engine's live model list where it can,
asks the few things it cannot infer (bind host, API key, which detected engines
to include), and scaffolds a working ``config.yaml`` from the matching presets.

Design notes
------------
* Everything that touches the outside world is injected so the wizard is fully
  hermetic in tests: ``probe`` (a TCP port check) and ``http_get`` (a JSON GET)
  both have stdlib-only defaults but are overridable.
* It NEVER auto-routes to a port it could not confirm. Unconfirmed open ports
  are surfaced and require an explicit yes; confirmed engines are suggested and,
  in non-interactive mode, auto-included.
* ``build_config_yaml`` is a pure function (selections -> YAML text). The write
  path validates the rendered config through ``load_config`` before replacing
  the target file, so the wizard can never leave an invalid config behind.

This module is imported by both console entry points:
  * ``routerctl init``           (router/cli.py)
  * ``local-engine-router init`` (router/__main__.py)
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

# Default listen settings, kept in one place so the wizard and the rest of the
# package agree.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8077


# --------------------------------------------------------------------------- #
# Port-probe target table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProbeTarget:
    """One well-known backend the wizard knows how to detect and scaffold.

    ``port`` is the conventional localhost port for the backend. ``engine_type``
    selects the config block shape (generic_process / api_swap / ollama).
    ``models_path`` + ``models_kind`` describe how to confirm the backend and
    list its live models over HTTP.
    """

    label: str
    port: int
    engine_key: str
    engine_type: str  # generic_process | api_swap | ollama
    models_path: str
    models_kind: str  # "ollama_tags" | "openai_models"
    base_url: str
    note: str = ""


# Conventional localhost ports for each supported backend. Some ports are shared
# by more than one backend (8080: llama.cpp / LocalAI / MLX / ramalama; 8000:
# vLLM / MAX); the table lists the most common occupant and the note names the
# alternatives so the wizard can explain the ambiguity.
PROBE_TARGETS: tuple[ProbeTarget, ...] = (
    ProbeTarget(
        label="Ollama",
        port=11434,
        engine_key="ollama",
        engine_type="ollama",
        models_path="/api/tags",
        models_kind="ollama_tags",
        base_url="http://127.0.0.1:11434",
    ),
    ProbeTarget(
        label="llama.cpp / LocalAI (OpenAI-compatible server)",
        port=8080,
        engine_key="llamacpp",
        engine_type="generic_process",
        models_path="/v1/models",
        models_kind="openai_models",
        base_url="http://127.0.0.1:8080",
        note="port 8080 is also used by LocalAI, MLX and ramalama",
    ),
    ProbeTarget(
        label="vLLM (OpenAI-compatible server)",
        port=8000,
        engine_key="vllm",
        engine_type="generic_process",
        models_path="/v1/models",
        models_kind="openai_models",
        base_url="http://127.0.0.1:8000",
        note="port 8000 is also used by Modular MAX",
    ),
    ProbeTarget(
        label="SGLang",
        port=30000,
        engine_key="sglang",
        engine_type="generic_process",
        models_path="/v1/models",
        models_kind="openai_models",
        base_url="http://127.0.0.1:30000",
    ),
    ProbeTarget(
        label="LM Studio",
        port=1234,
        engine_key="lmstudio",
        engine_type="api_swap",
        models_path="/api/v1/models",
        models_kind="openai_models",
        base_url="http://127.0.0.1:1234",
    ),
    ProbeTarget(
        label="TabbyAPI",
        port=5000,
        engine_key="tabbyapi",
        engine_type="api_swap",
        models_path="/v1/models",
        models_kind="openai_models",
        base_url="http://127.0.0.1:5000",
    ),
    ProbeTarget(
        label="KoboldCpp",
        port=5001,
        engine_key="koboldcpp",
        engine_type="generic_process",
        models_path="/v1/models",
        models_kind="openai_models",
        base_url="http://127.0.0.1:5001",
    ),
)


@dataclass
class Detection:
    """Result of probing one target: was the port open, did the backend confirm
    itself over HTTP, and which model ids did it advertise."""

    target: ProbeTarget
    port_open: bool = False
    confirmed: bool = False
    models: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Default (real) probe implementations - stdlib only, no extra deps
# --------------------------------------------------------------------------- #
def default_probe(host: str, port: int, timeout: float = 0.35) -> bool:
    """Return True if a TCP connection to host:port succeeds within *timeout*."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def default_http_get_json(url: str, timeout: float = 1.5) -> Any | None:
    """GET *url* and return parsed JSON, or None on any error.

    Deliberately forgiving: detection is best-effort and a failed fetch simply
    means "could not confirm / no models", never a crash.
    """
    import json

    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (urllib.error.URLError, OSError, ValueError):
        return None
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, TypeError):
        return None


def _extract_models(kind: str, payload: Any) -> list[str]:
    """Pull model ids out of a backend's model-list payload.

    Returns [] when the payload is missing or not the expected shape (so an
    engine that answered but advertised nothing is still 'confirmed').
    """
    ids: list[str] = []
    if not isinstance(payload, dict):
        return ids
    if kind == "ollama_tags":
        for m in payload.get("models") or []:
            if isinstance(m, dict):
                name = m.get("name") or m.get("model")
                if isinstance(name, str) and name:
                    ids.append(name)
    else:  # openai_models: {"data": [{"id": ...}, ...]}
        for m in payload.get("data") or []:
            if isinstance(m, dict):
                mid = m.get("id")
                if isinstance(mid, str) and mid:
                    ids.append(mid)
    # De-duplicate while preserving order, dropping ids that carry control
    # characters. Model ids come from a process the wizard does not trust; a raw
    # newline/tab/NUL in an id would corrupt the scaffolded YAML, so one bad id
    # is dropped rather than allowed to poison the whole config.
    seen: set[str] = set()
    out: list[str] = []
    for mid in ids:
        if any(ord(c) < 0x20 for c in mid):
            continue
        if mid not in seen:
            seen.add(mid)
            out.append(mid)
    return out


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def detect_engines(
    targets: Sequence[ProbeTarget] = PROBE_TARGETS,
    *,
    host: str = "127.0.0.1",
    probe: Callable[[str, int, float], bool] = default_probe,
    http_get: Callable[[str, float], Any | None] = default_http_get_json,
    connect_timeout: float = 0.35,
    http_timeout: float = 1.5,
) -> list[Detection]:
    """Probe every target on *host* and return a Detection per target.

    A target is ``confirmed`` when its model-list endpoint answers with
    parseable JSON of the expected shape (even if the model list is empty).
    """
    results: list[Detection] = []
    for t in targets:
        det = Detection(target=t)
        det.port_open = bool(probe(host, t.port, connect_timeout))
        if det.port_open:
            url = f"http://{host}:{t.port}{t.models_path}"
            payload = http_get(url, http_timeout)
            if payload is not None:
                # An expected-shape response confirms the backend.
                if isinstance(payload, dict) and (
                    "models" in payload or "data" in payload
                ):
                    det.confirmed = True
                    det.models = _extract_models(t.models_kind, payload)
        results.append(det)
    return results


# --------------------------------------------------------------------------- #
# Config generation (pure)
# --------------------------------------------------------------------------- #
@dataclass
class EngineSelection:
    """A chosen engine plus the model ids to register for it."""

    target: ProbeTarget
    models: list[str] = field(default_factory=list)


# A scalar is emitted bare only when it matches this simple-token shape;
# everything else (whitespace, YAML specials, control chars, reserved words) is
# double-quoted with full escaping.
_SAFE_PLAIN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./+-]*$")
# Tokens YAML 1.1 would resolve to a number if emitted bare ("1.5", "123",
# "1_000", "0x1F", "1e3", ".5", "1.2.3" is fine) — these must be quoted so a
# numeric-looking model id round-trips as a string.
_YAML_NUMERIC_RE = re.compile(
    r"^[+-]?("
    r"[0-9_]+"                      # int (with YAML 1.1 _ separators)
    r"|0x[0-9a-fA-F_]+"             # hex
    r"|0o?[0-7_]+"                  # octal
    r"|[0-9_]*\.[0-9_]*([eE][+-]?[0-9]+)?"  # float
    r"|[0-9_]+([eE][+-]?[0-9]+)"    # scientific without dot
    r")$"
)
_YAML_RESERVED = frozenset(
    {
        "true", "false", "null", "yes", "no", "on", "off",
        "True", "False", "Null", "Yes", "No", "On", "Off", "~",
    }
)


def _yaml_str(value: str) -> str:
    """Render *value* as a safe YAML scalar.

    Emits a bare scalar only for simple tokens; anything else, including control
    characters, whitespace, YAML specials, or a reserved word, is double-quoted
    with full escaping. The result always parses back to the original string, so
    even a hostile id advertised by a local engine cannot inject YAML structure.
    """
    if (
        _SAFE_PLAIN_RE.match(value)
        and value not in _YAML_RESERVED
        and not _YAML_NUMERIC_RE.match(value)
    ):
        return value
    out: list[str] = []
    for ch in value:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\x00":
            out.append("\\0")
        elif ord(ch) < 0x20:
            out.append("\\x%02X" % ord(ch))
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _engine_block(sel: EngineSelection) -> list[str]:
    """Render the ``engines:`` entry (indented under ``engines:``) for one
    selection, as a list of lines. Uses the same shapes as presets/."""
    t = sel.target
    L: list[str] = [f"  {t.engine_key}:"]
    if t.engine_type == "ollama":
        L += [
            "    type: ollama",
            "    enabled: true",
            f"    base_url: {t.base_url}",
            "    health_path: /api/tags",
            "    unload_timeout_s: 60",
            "    tags_cache_ttl_s: 30",
            "    # systemd_unit: ollama.service   # best-effort start if unreachable",
        ]
    elif t.engine_type == "api_swap":
        L += [
            "    type: api_swap",
            "    enabled: true",
            f"    base_url: {t.base_url}",
        ]
        if t.engine_key == "lmstudio":
            L += [
                "    health_path: /api/v1/models",
                "    load_path: /api/v1/models/load",
                "    load_body: { model: \"{model}\" }",
                "    unload_path: /api/v1/models/unload",
                "    unload_body: { instance_id: \"{model}\" }",
                "    loaded_path: /api/v1/models",
                "    loaded_models_key: data",
                "    loaded_name_key: id",
                "    unload_timeout_s: 60",
                "    tags_cache_ttl_s: 30",
            ]
        elif t.engine_key == "tabbyapi":
            L += [
                "    health_path: /health",
                "    load_path: /v1/model/load",
                "    load_body: { model_name: \"{model}\" }",
                "    unload_path: /v1/model/unload",
                "    unload_body: {}",
                "    loaded_path: /v1/model",
                "    loaded_models_key: \"\"",
                "    loaded_name_key: id",
                "    unload_timeout_s: 60",
                "    tags_cache_ttl_s: 30",
                "    # control_headers: { x-admin-key: \"<TABBYAPI_ADMIN_KEY>\" }",
            ]
        else:  # generic api_swap fallback
            L += [
                "    health_path: /v1/models",
                "    unload_path: \"\"   # set the engine's unload endpoint to free VRAM on swap",
                "    tags_cache_ttl_s: 30",
            ]
    else:  # generic_process
        port = str(t.port)
        ready_path = "/health" if t.engine_key == "llamacpp" else "/v1/models"
        L += [
            "    type: generic_process",
            "    enabled: true",
            f"    base_url: {t.base_url}",
            "    # EDIT: the exact command that launches this server, so the router",
            "    #       can restart it after a swap. A server is already running on",
            f"    #       port {port}; until start_cmd is correct, swapping away and",
            "    #       back will not relaunch it.",
            "    start_cmd:",
            f"      - <PATH_TO_{t.engine_key.upper()}_SERVER>",
            "      - --port",
            f"      - \"{port}\"",
            f"    ready_path: {ready_path}",
            "    start_timeout_s: 300",
            "    stop_signal: SIGTERM",
            "    stop_timeout_s: 30",
            f"    # process_pattern: {t.engine_key}   # pgrep -f match to stop a server the router did not launch",
        ]
    return L


def _model_lines(sel: EngineSelection) -> list[str]:
    """Render the ``models:`` entries (indented under ``models:``) for one
    selection."""
    L: list[str] = []
    for mid in sel.models:
        L.append(f"  - id: {_yaml_str(mid)}")
        L.append(f"    engine: {sel.target.engine_key}")
    return L


_HEADER = """\
# local-engine-router config - generated by `local-engine-router init`.
# Routing is by the request's `model` field; only one engine holds the GPU at a
# time, so the router swaps engines on demand. Edit freely and restart the
# router (`routerctl restart`) to apply changes.
#
# Validate without starting:  python3 -m router --check-config --config config.yaml
"""


def build_config_yaml(
    selections: Sequence[EngineSelection],
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    api_keys: Sequence[str] = (),
    bound_offlocalhost_note: bool = True,
) -> str:
    """Render a complete config.yaml as text from chosen engines + models.

    Pure function: no I/O. The result is guaranteed to be parseable YAML and, by
    construction, valid for ``load_config`` (the write path re-validates it).
    """
    lines: list[str] = [_HEADER.rstrip("\n"), ""]
    lines.append(f"host: {host}")
    lines.append(f"port: {port}")
    if api_keys:
        lines.append("api_keys:")
        for k in api_keys:
            lines.append(f"  - {_yaml_str(k)}")
    else:
        lines.append("api_keys: []   # set a key here if you bind off-localhost")
    lines.append("")

    if selections:
        lines.append("engines:")
        for sel in selections:
            lines.extend(_engine_block(sel))
            lines.append("")
        # Models (only real, detected ids are emitted).
        model_lines: list[str] = []
        for sel in selections:
            model_lines.extend(_model_lines(sel))
        lines.append("models:")
        if model_lines:
            lines.extend(model_lines)
        else:
            lines.append(
                "  # No live models were detected. Add entries as:"
            )
            lines.append("  #   - id: <model-id-clients-send>")
            lines.append(
                f"  #     engine: {selections[0].target.engine_key}"
            )
        lines.append("")
    else:
        # Nothing selected: emit a minimal but valid starter the user can edit.
        return STARTER_CONFIG

    return "\n".join(lines).rstrip("\n") + "\n"


# A self-contained, fully-commented starter written when nothing is detected or
# when `init --example` is used. It defines one active engine (Ollama, the most
# common zero-config backend) so the router is useful as soon as Ollama is
# running, and shows a commented generic_process block to copy for other
# backends. An explicit `engines:` block is important: if `engines:` is omitted
# entirely the loader falls back to the legacy ds4 + ollama defaults, which is
# more surprising than declaring exactly one engine here.
STARTER_CONFIG = """\
# local-engine-router config - starter scaffold.
# Routing is by the request's `model` field; only one engine holds the GPU at a
# time, so the router swaps engines on demand.
#
# Next steps:
#   - Have Ollama running? This works as-is; the ollama engine below picks up
#     whatever `ollama list` shows. Just restart the router: routerctl restart
#   - Another backend? Uncomment a block below (or see presets/ for every
#     backend), then add a models: entry pointing at it.
#   - Or let the wizard detect everything for you:  local-engine-router init
#
# Validate without starting: python3 -m router --check-config --config config.yaml

host: 127.0.0.1
port: 8077
api_keys: []          # set a long random key if you bind off-localhost (host: 0.0.0.0)

engines:
  # Ollama: the router unloads its models on swap-away; it does not launch the
  # Ollama daemon itself (start that separately). Live Ollama tags route even
  # without a static models: entry below.
  ollama:
    type: ollama
    enabled: true
    base_url: http://127.0.0.1:11434
    health_path: /api/tags
    unload_timeout_s: 60

  # A server the router launches + supervises (llama.cpp, vLLM, SGLang, ...).
  # Uncomment and edit, then add a matching models: entry.
  # llamacpp:
  #   type: generic_process
  #   enabled: true
  #   base_url: http://127.0.0.1:8080
  #   start_cmd: ["/usr/local/bin/llama-server", "-m", "/models/my-model.gguf", "--port", "8080"]
  #   ready_path: /health
  #   start_timeout_s: 300

models: []            # e.g. - { id: llama3.1:8b, engine: ollama }
"""


# --------------------------------------------------------------------------- #
# Interactive front-end
# --------------------------------------------------------------------------- #
def _isatty(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _ask(
    prompt: str,
    default: str,
    *,
    in_: Any,
    out: Any,
    interactive: bool,
) -> str:
    """Prompt for a line; return *default* unchanged in non-interactive mode."""
    if not interactive:
        return default
    suffix = f" [{default}]" if default != "" else ""
    out.write(f"{prompt}{suffix}: ")
    out.flush()
    line = in_.readline()
    if not line:
        return default
    line = line.strip()
    return line if line else default


def _ask_yes_no(
    prompt: str,
    default: bool,
    *,
    in_: Any,
    out: Any,
    interactive: bool,
) -> bool:
    if not interactive:
        return default
    d = "Y/n" if default else "y/N"
    out.write(f"{prompt} [{d}]: ")
    out.flush()
    line = in_.readline()
    if not line:
        return default
    line = line.strip().lower()
    if not line:
        return default
    return line[0] == "y"


def _validate_config_text(text: str) -> None:
    """Raise if *text* is not a valid router config.

    Writes to a temp file and runs the real loader, so the wizard never emits a
    config the router would reject at startup.
    """
    from .config import load_config

    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="ler-init-")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        load_config(tmp)
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _write_config(path: str, text: str) -> None:
    """Validate then atomically write *text* to *path*."""
    _validate_config_text(text)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".tmp", prefix="config-", dir=parent)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _parse_init_args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="local-engine-router init",
        description=(
            "Detect running local engines, scaffold a working config.yaml, and "
            "optionally start the router."
        ),
    )
    p.add_argument(
        "--config",
        default=os.environ.get("ROUTER_CONFIG", "config.yaml"),
        help="path to write the config (default: $ROUTER_CONFIG or ./config.yaml)",
    )
    p.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"bind address to write into the config (default: {DEFAULT_HOST})",
    )
    p.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"listen port to write into the config (default: {DEFAULT_PORT})",
    )
    p.add_argument(
        "--probe-host",
        default="127.0.0.1",
        help="host to probe for running engines (default: 127.0.0.1)",
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="non-interactive: accept defaults, include all confirmed engines",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing config without asking",
    )
    p.add_argument(
        "--example",
        action="store_true",
        help="write a commented starter config without probing any ports",
    )
    p.add_argument(
        "--detect-only",
        action="store_true",
        help="probe and print what is running; write nothing",
    )
    return p.parse_args(list(argv) if argv is not None else None)


def run_init(
    argv: Sequence[str] | None = None,
    *,
    stdin: Any = None,
    stdout: Any = None,
    probe: Callable[[str, int, float], bool] | None = None,
    http_get: Callable[[str, float], Any | None] | None = None,
) -> int:
    """Entry point for ``init``. Returns a process exit code (0 == success)."""
    args = _parse_init_args(argv)
    out = stdout if stdout is not None else sys.stdout
    in_ = stdin if stdin is not None else sys.stdin
    probe = probe or default_probe
    http_get = http_get or default_http_get_json
    interactive = (not args.yes) and _isatty(in_)

    # --example: write the starter and stop (used by install.sh).
    if args.example:
        if os.path.exists(args.config) and not args.force:
            out.write(
                f"config already exists at {args.config}; not overwriting "
                "(use --force).\n"
            )
            return 0
        _write_config(args.config, STARTER_CONFIG)
        out.write(f"wrote starter config to {args.config}\n")
        return 0

    out.write("Probing well-known localhost ports for running engines...\n")
    detections = detect_engines(
        host=args.probe_host, probe=probe, http_get=http_get
    )
    confirmed = [d for d in detections if d.confirmed]
    open_unconfirmed = [
        d for d in detections if d.port_open and not d.confirmed
    ]

    if confirmed:
        out.write("\nDetected engines:\n")
        for d in confirmed:
            extra = f"  ({len(d.models)} model(s))" if d.models else "  (no models advertised)"
            out.write(f"  [x] {d.target.label} on :{d.target.port}{extra}\n")
            for mid in d.models[:12]:
                out.write(f"        - {mid}\n")
            if len(d.models) > 12:
                out.write(f"        ... and {len(d.models) - 12} more\n")
    else:
        out.write("\nNo engines confirmed on the well-known ports.\n")

    if open_unconfirmed:
        out.write("\nOpen ports that did not confirm as a known backend:\n")
        for d in open_unconfirmed:
            out.write(f"  [?] :{d.target.port} (expected {d.target.label})\n")

    # --detect-only: report and stop.
    if args.detect_only:
        return 0

    # Choose which engines to include.
    selections: list[EngineSelection] = []
    for d in confirmed:
        include = _ask_yes_no(
            f"Include {d.target.label} (:{d.target.port})?",
            True,
            in_=in_,
            out=out,
            interactive=interactive,
        )
        if include:
            selections.append(EngineSelection(target=d.target, models=list(d.models)))
    # Offer unconfirmed open ports only in interactive mode (never auto-route).
    if interactive:
        for d in open_unconfirmed:
            include = _ask_yes_no(
                f"Port :{d.target.port} is open but unconfirmed. Add it as "
                f"{d.target.label} anyway?",
                False,
                in_=in_,
                out=out,
                interactive=interactive,
            )
            if include:
                selections.append(EngineSelection(target=d.target, models=[]))

    # Questions the probe cannot answer.
    host = _ask(
        "Bind host for the router",
        args.host,
        in_=in_,
        out=out,
        interactive=interactive,
    )
    api_key = _ask(
        "API key to require (blank for none)",
        "",
        in_=in_,
        out=out,
        interactive=interactive,
    )
    api_keys = [api_key] if api_key else []

    text = build_config_yaml(
        selections, host=host, port=args.port, api_keys=api_keys
    )

    if os.path.exists(args.config) and not args.force:
        overwrite = _ask_yes_no(
            f"{args.config} already exists. Overwrite it?",
            False,
            in_=in_,
            out=out,
            interactive=interactive,
        )
        if not overwrite:
            out.write("Aborted; existing config left unchanged.\n")
            return 1

    try:
        _write_config(args.config, text)
    except Exception as exc:  # noqa: BLE001 - surface a clear message
        out.write(f"ERROR: could not write a valid config: {exc}\n")
        return 1

    out.write(f"\nWrote {args.config}\n")
    if not selections:
        out.write(
            "No engines were added. Edit the file (see presets/) or re-run "
            "`local-engine-router init` once an engine is running.\n"
        )
    else:
        needs_cmd = [
            s.target.label
            for s in selections
            if s.target.engine_type == "generic_process"
        ]
        if needs_cmd:
            out.write(
                "\nNOTE: set start_cmd for these process-managed engines so the "
                "router can restart them after a swap:\n"
            )
            for label in needs_cmd:
                out.write(f"  - {label}\n")
    probe_host = host if host != "0.0.0.0" else "127.0.0.1"
    out.write(
        "\nNext: start the router with this config:\n"
        f"  local-engine-router --config {args.config}\n"
        f"If your systemd service reads {args.config}, restart it instead:\n"
        "  routerctl restart\n"
        "Then check it:\n"
        f"  curl http://{probe_host}:{args.port}/health\n"
    )
    return 0
