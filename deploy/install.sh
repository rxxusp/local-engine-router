#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# install.sh  — idempotent installer for llm-router.
# Safe to re-run at any time.
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> llm-router install (repo: $REPO_ROOT)"

# 1. Verify Python dependencies -------------------------------------------
echo "==> checking Python dependencies..."
if ! python3 -c "import fastapi, uvicorn, httpx, yaml, pydantic" 2>/dev/null; then
    echo "    some deps missing; attempting: python3 -m pip install --user -r requirements.txt"
    if ! python3 -m pip install --user -r requirements.txt; then
        echo "    WARNING: pip install did not fully succeed; proceeding anyway (deps may be present system-wide)"
    fi
    # Re-check after install attempt.
    if ! python3 -c "import fastapi, uvicorn, httpx, yaml, pydantic" 2>/dev/null; then
        echo "    WARNING: could not import all required packages after install attempt"
        echo "             The service may fail to start until dependencies are installed."
    fi
else
    echo "    dependencies OK"
fi

# 2. Ensure logs directory exists -----------------------------------------
echo "==> creating logs/ directory..."
mkdir -p "$REPO_ROOT/logs"

# 3. Install systemd *user* unit -------------------------------------------
# The router runs as a user unit so it shares the same `systemctl --user`
# manager as ds4.service and can start/stop it. (A system unit running as
# User=grahamfm cannot reliably control the user manager.)
UNIT_DIR="$HOME/.config/systemd/user"
echo "==> installing user systemd unit to $UNIT_DIR ..."
mkdir -p "$UNIT_DIR"
cp "$REPO_ROOT/deploy/llm-router.service" "$UNIT_DIR/llm-router.service"

# 4. Enable lingering so the user manager (and the router) start at boot ----
echo "==> ensuring lingering is enabled for $USER (boot start without login)..."
if command -v loginctl >/dev/null 2>&1; then
    if [ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" != "yes" ]; then
        sudo loginctl enable-linger "$USER" || \
            echo "    WARNING: could not enable lingering; router won't auto-start at boot until you log in"
    else
        echo "    lingering already enabled"
    fi
fi

# 5. Reload + enable + start -----------------------------------------------
echo "==> enabling and starting llm-router.service (user)..."
systemctl --user daemon-reload
systemctl --user enable --now llm-router.service

# 6. Install routerctl into PATH ------------------------------------------
echo "==> installing routerctl to ~/.local/bin ..."
mkdir -p "$HOME/.local/bin"
chmod +x "$REPO_ROOT/routerctl"
ln -sf "$REPO_ROOT/routerctl" "$HOME/.local/bin/routerctl"
case ":$PATH:" in
    *":$HOME/.local/bin:"*) : ;;
    *) echo "    NOTE: add ~/.local/bin to your PATH to use 'routerctl' directly" ;;
esac

# 7. Wait for /health (up to 30 s) ----------------------------------------
ROUTER_URL="http://127.0.0.1:8077"
echo "==> waiting for router at $ROUTER_URL/health (up to 30 s)..."
HEALTHY=0
for i in $(seq 1 30); do
    if curl -fsS "$ROUTER_URL/health" >/dev/null 2>&1; then
        HEALTHY=1
        break
    fi
    sleep 1
done

if [ "$HEALTHY" -eq 0 ]; then
    echo ""
    echo "ERROR: router did not become healthy within 30 s."
    echo "       Check logs with: journalctl --user -u llm-router -n 50"
    exit 1
fi

echo "==> router is up!"
echo ""
echo "--- /status ---"
curl -fsS "$ROUTER_URL/status" | python3 -m json.tool || curl -fsS "$ROUTER_URL/status"
echo ""

# 8. Next steps -----------------------------------------------------------
cat <<'EOF'

=== Next steps =========================================================

Point your clients at http://127.0.0.1:8077  (or http://172.17.0.1:8077
from inside Docker containers, e.g. Open WebUI).

  OpenAI-compat:  http://127.0.0.1:8077/v1
  Ollama-native:  http://127.0.0.1:8077  (same base, /api/* is forwarded)

routerctl usage:
  routerctl status           — show active engine, in-flight counts
  routerctl models           — list all known models
  routerctl ds4              — swap to ds4 engine now
  routerctl ollama           — swap to ollama engine now
  routerctl use deepseek-v4-flash   — swap to whatever engine owns that model
  routerctl logs             — tail the service log (journalctl --user)
  routerctl restart          — restart the router service

Service management (user unit):
  systemctl --user status llm-router
  systemctl --user restart llm-router
  journalctl --user -u llm-router -f

=======================================================================
EOF
