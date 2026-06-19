#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# install.sh - one-command bootstrap for local-engine-router.
#
#   curl -fsSL https://raw.githubusercontent.com/rxxusp/local-engine-router/main/install.sh | bash
#
# What it does:
#   1. checks for python3 >= 3.10
#   2. creates an isolated virtualenv (no system-site-packages pollution)
#   3. installs the local-engine-router package + its dependencies into it
#   4. puts `local-engine-router` and `routerctl` on your PATH (~/.local/bin)
#   5. writes a starter config if you do not have one yet
#   6. offers to install + enable the systemd --user service (Linux)
#
# It is idempotent and safe to re-run. Nothing here needs a GPU.
#
# Pass flags through the pipe with `-s --`:
#   curl -fsSL .../install.sh | bash -s -- --yes
#   curl -fsSL .../install.sh | bash -s -- --no-service
#
# Everything is overridable by environment variable (see DEFAULTS below).
# ===========================================================================

# --- Defaults (all overridable via environment) ----------------------------
APP="local-engine-router"
: "${LER_VENV:=${XDG_DATA_HOME:-$HOME/.local/share}/${APP}/venv}"
: "${LER_CONFIG:=${XDG_CONFIG_HOME:-$HOME/.config}/${APP}/config.yaml}"
: "${LER_BIN:=$HOME/.local/bin}"
: "${LER_UNIT_DIR:=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user}"
: "${LER_GIT_URL:=https://github.com/rxxusp/local-engine-router.git}"
: "${LER_REF:=main}"
# LER_SOURCE: pip install spec to use. Empty => auto (local checkout, else PyPI,
# else the GitHub repo). Set e.g. LER_SOURCE=local-engine-router==0.5.0 to pin.
: "${LER_SOURCE:=}"

YES=0
NO_SERVICE=0
DRY_RUN=0
PRINT_UNIT=0
UNINSTALL=0

# --- Argument parsing ------------------------------------------------------
usage() {
    cat <<EOF
local-engine-router installer

Usage: install.sh [options]

Options:
  -y, --yes          non-interactive; accept defaults and enable the service
      --no-service   do not install or enable the systemd --user service
      --dry-run      print what would happen and exit (no changes)
      --print-unit   print the generated systemd unit and exit
      --uninstall    remove the venv, PATH shims, and service (keeps your config)
  -h, --help         show this help and exit

Environment overrides:
  LER_VENV       venv location      (default: \$XDG_DATA_HOME/${APP}/venv)
  LER_CONFIG     config file path   (default: \$XDG_CONFIG_HOME/${APP}/config.yaml)
  LER_BIN        PATH bin dir       (default: ~/.local/bin)
  LER_UNIT_DIR   systemd user dir   (default: \$XDG_CONFIG_HOME/systemd/user)
  LER_SOURCE     pip install spec   (default: auto-detect checkout, else PyPI/git)
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -y|--yes) YES=1 ;;
        --no-service) NO_SERVICE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --print-unit) PRINT_UNIT=1 ;;
        --uninstall) UNINSTALL=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

say()  { printf '==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
err()  { printf 'ERROR: %s\n' "$*" >&2; }

CONFIG_DIR="$(dirname "$LER_CONFIG")"
VENV_BIN="$LER_VENV/bin"

# --- Generated systemd unit ------------------------------------------------
emit_unit() {
    cat <<EOF
[Unit]
Description=local-engine-router (single-port switchboard for local LLM engines)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${CONFIG_DIR}
Environment=PYTHONUNBUFFERED=1
Environment=ROUTER_CONFIG=${LER_CONFIG}
ExecStart=${VENV_BIN}/local-engine-router --config ${LER_CONFIG}
Restart=on-failure
RestartSec=3
TimeoutStopSec=15

[Install]
WantedBy=default.target
EOF
}

if [ "$PRINT_UNIT" -eq 1 ]; then
    emit_unit
    exit 0
fi

# --- Uninstall -------------------------------------------------------------
if [ "$UNINSTALL" -eq 1 ]; then
    say "Uninstalling ${APP} (your config at ${LER_CONFIG} is kept)"
    if command -v systemctl >/dev/null 2>&1; then
        systemctl --user disable --now "${APP}.service" 2>/dev/null || true
    fi
    rm -f "$LER_UNIT_DIR/${APP}.service"
    [ -d "$LER_UNIT_DIR" ] && systemctl --user daemon-reload 2>/dev/null || true
    rm -f "$LER_BIN/local-engine-router" "$LER_BIN/routerctl"
    rm -rf "$LER_VENV"
    say "Done. Removed venv, PATH shims, and service unit."
    say "Kept config: ${LER_CONFIG}"
    exit 0
fi

# --- Resolve interactivity -------------------------------------------------
INTERACTIVE=0
if [ "$YES" -eq 0 ] && [ -t 0 ]; then
    INTERACTIVE=1
fi

# --- Resolve the pip source ------------------------------------------------
script_dir() {
    local src="${BASH_SOURCE[0]:-}"
    [ -n "$src" ] || return 0
    [ -f "$src" ] || return 0
    (cd "$(dirname "$src")" 2>/dev/null && pwd) || true
}

resolve_source() {
    if [ -n "$LER_SOURCE" ]; then
        printf '%s' "$LER_SOURCE"
        return 0
    fi
    local here
    here="$(script_dir)"
    if [ -n "$here" ] && [ -f "$here/pyproject.toml" ] \
        && grep -q 'name = "local-engine-router"' "$here/pyproject.toml" 2>/dev/null; then
        printf '%s' "$here"
        return 0
    fi
    printf 'PYPI'
}

SOURCE="$(resolve_source)"

# --- Dry run ---------------------------------------------------------------
if [ "$DRY_RUN" -eq 1 ]; then
    say "Dry run - no changes will be made."
    echo "  python      : $(command -v python3 || echo '(not found)')"
    echo "  venv        : $LER_VENV"
    echo "  config      : $LER_CONFIG"
    echo "  bin dir     : $LER_BIN"
    echo "  unit dir    : $LER_UNIT_DIR"
    if [ "$SOURCE" = "PYPI" ]; then
        echo "  pip source  : PyPI ($APP), fallback git+$LER_GIT_URL@$LER_REF"
    else
        echo "  pip source  : $SOURCE"
    fi
    echo "  service     : $([ "$NO_SERVICE" -eq 1 ] && echo skip || echo install)"
    echo "  interactive : $INTERACTIVE"
    exit 0
fi

# --- 1. Python check -------------------------------------------------------
say "Checking Python (need >= 3.10)..."
if ! command -v python3 >/dev/null 2>&1; then
    err "python3 not found on PATH. Install Python 3.10+ and re-run."
    exit 1
fi
if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)'; then
    err "python3 is older than 3.10 ($(python3 -V 2>&1)). Upgrade and re-run."
    exit 1
fi
say "    $(python3 -V 2>&1)"

# --- 2. Virtualenv ---------------------------------------------------------
if [ -x "$VENV_BIN/python" ]; then
    say "Reusing existing virtualenv at $LER_VENV"
else
    say "Creating virtualenv at $LER_VENV ..."
    if ! python3 -m venv "$LER_VENV"; then
        err "could not create a virtualenv. On Debian/Ubuntu: apt install python3-venv"
        exit 1
    fi
fi
"$VENV_BIN/python" -m pip install --upgrade pip >/dev/null 2>&1 || \
    warn "could not upgrade pip inside the venv; continuing"

# --- 3. Install the package ------------------------------------------------
say "Installing the ${APP} package..."
if [ "$SOURCE" = "PYPI" ]; then
    if "$VENV_BIN/python" -m pip install --upgrade "$APP"; then
        :
    else
        say "PyPI install unavailable; installing from GitHub ($LER_REF)..."
        "$VENV_BIN/python" -m pip install --upgrade "git+${LER_GIT_URL}@${LER_REF}"
    fi
else
    "$VENV_BIN/python" -m pip install --upgrade "$SOURCE"
fi

# Sanity-check the install.
if ! "$VENV_BIN/python" -c 'import router' 2>/dev/null; then
    err "package import failed after install; aborting."
    exit 1
fi
say "    installed $("$VENV_BIN/python" -c 'import router; print("v"+router.__version__)' 2>/dev/null || echo ok)"

# --- 4. PATH shims ---------------------------------------------------------
say "Linking CLIs into $LER_BIN ..."
mkdir -p "$LER_BIN"
ln -sf "$VENV_BIN/local-engine-router" "$LER_BIN/local-engine-router"
ln -sf "$VENV_BIN/routerctl" "$LER_BIN/routerctl"
case ":$PATH:" in
    *":$LER_BIN:"*) : ;;
    *) warn "$LER_BIN is not on your PATH. Add it, e.g.:
         echo 'export PATH=\"$LER_BIN:\$PATH\"' >> ~/.bashrc && exec \$SHELL" ;;
esac

# --- 5. Starter config -----------------------------------------------------
if [ -f "$LER_CONFIG" ]; then
    say "Keeping existing config at $LER_CONFIG"
else
    say "Writing a starter config to $LER_CONFIG ..."
    mkdir -p "$CONFIG_DIR"
    "$VENV_BIN/local-engine-router" init --example --config "$LER_CONFIG"
fi

# --- 6. systemd --user service --------------------------------------------
SERVICE_STARTED=0
if [ "$NO_SERVICE" -eq 1 ]; then
    say "Skipping systemd service (--no-service)."
elif ! command -v systemctl >/dev/null 2>&1; then
    say "systemctl not found (not a systemd host); skipping the service."
    say "Run the router directly:  $LER_BIN/local-engine-router --config $LER_CONFIG"
else
    WANT=0
    if [ "$YES" -eq 1 ]; then
        WANT=1
    elif [ "$INTERACTIVE" -eq 1 ]; then
        printf 'Install and enable the systemd --user service now? [Y/n]: '
        read -r reply || reply=""
        case "${reply:-y}" in [nN]*) WANT=0 ;; *) WANT=1 ;; esac
    else
        # Piped, no --yes: install the unit but do not start it unprompted.
        WANT=2
    fi

    mkdir -p "$LER_UNIT_DIR"
    emit_unit > "$LER_UNIT_DIR/${APP}.service"
    systemctl --user daemon-reload 2>/dev/null || \
        warn "systemctl --user daemon-reload failed (no user bus in this session?)"

    if [ "$WANT" -eq 1 ]; then
        # $USER is not always exported (piped curl from a minimal shell, sudo
        # without -l, cron); resolve it robustly so `set -u` does not abort here.
        _user="${USER:-$(id -un 2>/dev/null || echo "")}"
        if command -v loginctl >/dev/null 2>&1 && [ -n "$_user" ]; then
            if [ "$(loginctl show-user "$_user" -p Linger --value 2>/dev/null || echo no)" != "yes" ]; then
                # `sudo -n` so an uncached credential fails fast instead of
                # silently blocking on a hidden tty password prompt.
                sudo -n loginctl enable-linger "$_user" 2>/dev/null || \
                    warn "could not enable lingering (needs passwordless sudo); the router will not auto-start at boot until you log in. Enable later with: sudo loginctl enable-linger $_user"
            fi
        fi
        if systemctl --user enable --now "${APP}.service" 2>/dev/null; then
            SERVICE_STARTED=1
            say "Service enabled and started."
        else
            warn "could not start the service; run it directly:
         $LER_BIN/local-engine-router --config $LER_CONFIG"
        fi
    else
        say "Installed the unit at $LER_UNIT_DIR/${APP}.service (not started)."
        say "Start it with:  systemctl --user enable --now ${APP}.service"
    fi
fi

# --- 7. Health wait (only if we started the service) -----------------------
if [ "$SERVICE_STARTED" -eq 1 ]; then
    HOST="$("$VENV_BIN/python" -c "from router.config import load_config; print(load_config('$LER_CONFIG').host)" 2>/dev/null || echo 127.0.0.1)"
    PORT="$("$VENV_BIN/python" -c "from router.config import load_config; print(load_config('$LER_CONFIG').port)" 2>/dev/null || echo 8077)"
    [ "$HOST" = "0.0.0.0" ] && HOST=127.0.0.1
    say "Waiting for the router at http://$HOST:$PORT/health (up to 20s)..."
    HEALTHY=0
    for _ in $(seq 1 20); do
        if "$VENV_BIN/python" - "$HOST" "$PORT" <<'PY' 2>/dev/null
import sys, urllib.request
host, port = sys.argv[1], sys.argv[2]
try:
    urllib.request.urlopen(f"http://{host}:{port}/health", timeout=2)
except Exception:
    raise SystemExit(1)
PY
        then HEALTHY=1; break; fi
        sleep 1
    done
    if [ "$HEALTHY" -eq 1 ]; then
        say "Router is up."
    else
        warn "router did not answer /health yet. Check: journalctl --user -u ${APP} -n 50"
    fi
fi

# --- Next steps ------------------------------------------------------------
cat <<EOF

=== local-engine-router installed ======================================

  CLIs : $LER_BIN/local-engine-router , $LER_BIN/routerctl
  venv : $LER_VENV
  conf : $LER_CONFIG

Next:
  1. Detect your running engines and write them into this config:
       local-engine-router init --config $LER_CONFIG
  2. Start (or restart) the router so it picks up that config:
       routerctl restart            # if you installed the service
       # or run it directly:
       local-engine-router --config $LER_CONFIG
  3. Send a request:
       curl http://127.0.0.1:8077/v1/models

Re-run this installer any time to upgrade. Remove everything with:
  install.sh --uninstall
=======================================================================
EOF
