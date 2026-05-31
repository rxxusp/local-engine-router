#!/usr/bin/env bash
# smoke_test.sh — End-to-end smoke test for llm-router.
#
# WARNING: This script performs REAL engine swaps. Steps 3, 5, and 7 send
# actual requests to GPU-backed models and will trigger ds4<->ollama swaps.
# Each swap can take 1-4 minutes. Total runtime is typically 5-15 minutes.
# Do NOT run this while other inference workloads are active.
#
# Usage:
#   ROUTER_URL=http://127.0.0.1:8077 bash deploy/smoke_test.sh
#
# Requires: curl, python3 (no jq needed).

set -uo pipefail

ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:8077}"

PASS=0
FAIL=0
TOTAL=0

pass() {
    PASS=$(( PASS + 1 ))
    TOTAL=$(( TOTAL + 1 ))
    printf '[PASS] %s\n' "$1"
}

fail() {
    FAIL=$(( FAIL + 1 ))
    TOTAL=$(( TOTAL + 1 ))
    printf '[FAIL] %s\n' "$1"
    if [ -n "${2:-}" ]; then
        printf '       detail: %s\n' "$2"
    fi
}

# ---------------------------------------------------------------------------
# Step 1: GET /health == ok
# ---------------------------------------------------------------------------
printf '\n--- Step 1: GET /health ---\n'
HEALTH_BODY=$(curl --silent --max-time 10 "${ROUTER_URL}/health")
HEALTH_STATUS=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('status',''))" "$HEALTH_BODY" 2>/dev/null || echo "")
if [ "$HEALTH_STATUS" = "ok" ]; then
    pass "GET /health returned status=ok"
else
    fail "GET /health did not return status=ok" "body=${HEALTH_BODY}"
fi

# ---------------------------------------------------------------------------
# Step 2: GET /v1/models contains deepseek-v4-flash AND qwen3.6-uncensored:27b
# ---------------------------------------------------------------------------
printf '\n--- Step 2: GET /v1/models ---\n'
MODELS_BODY=$(curl --silent --max-time 15 "${ROUTER_URL}/v1/models")
HAS_DS4=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    ids = [m.get('id','') for m in d.get('data', [])]
    print('yes' if 'deepseek-v4-flash' in ids else 'no')
except Exception as e:
    print('error: ' + str(e))
" "$MODELS_BODY" 2>/dev/null || echo "error")
HAS_QWEN=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    ids = [m.get('id','') for m in d.get('data', [])]
    print('yes' if 'qwen3.6-uncensored:27b' in ids else 'no')
except Exception as e:
    print('error: ' + str(e))
" "$MODELS_BODY" 2>/dev/null || echo "error")
if [ "$HAS_DS4" = "yes" ] && [ "$HAS_QWEN" = "yes" ]; then
    pass "GET /v1/models contains deepseek-v4-flash and qwen3.6-uncensored:27b"
else
    fail "GET /v1/models missing expected models" "has_deepseek=${HAS_DS4} has_qwen=${HAS_QWEN}"
fi

# ---------------------------------------------------------------------------
# Step 3: POST /v1/chat/completions with deepseek-v4-flash (non-streaming)
#         This may trigger a swap to ds4 — allow up to 300s.
# ---------------------------------------------------------------------------
printf '\n--- Step 3: POST /v1/chat/completions (deepseek-v4-flash, non-streaming) ---\n'
printf '    (may trigger ds4 swap, allow up to 300s)\n'
CHAT_DS4_BODY=$(curl --silent --max-time 300 \
    -H "Content-Type: application/json" \
    -d '{"model":"deepseek-v4-flash","stream":false,"max_tokens":64,"messages":[{"role":"user","content":"say hi in 3 words"}]}' \
    "${ROUTER_URL}/v1/chat/completions")
CHAT_DS4_HTTP=$(curl --silent --max-time 300 --output /dev/null --write-out '%{http_code}' \
    -H "Content-Type: application/json" \
    -d '{"model":"deepseek-v4-flash","stream":false,"max_tokens":64,"messages":[{"role":"user","content":"say hi in 3 words"}]}' \
    "${ROUTER_URL}/v1/chat/completions")
CHAT_DS4_CONTENT=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    msg = d['choices'][0]['message']
    c = msg.get('content')
    r = msg.get('reasoning_content') or msg.get('reasoning')
    text = c if c else r
    print('ok:' + repr((text or '')[:80]))
except Exception as e:
    print('error: ' + str(e))
" "$CHAT_DS4_BODY" 2>/dev/null || echo "error: parse failed")
if [ "$CHAT_DS4_HTTP" = "200" ] && [[ "$CHAT_DS4_CONTENT" == ok:* ]]; then
    pass "deepseek-v4-flash chat: HTTP 200 with choices[0].message.content present (${CHAT_DS4_CONTENT})"
else
    fail "deepseek-v4-flash chat failed" "http=${CHAT_DS4_HTTP} content=${CHAT_DS4_CONTENT}"
fi

# ---------------------------------------------------------------------------
# Step 4: GET /status -> active_engine == "ds4"
# ---------------------------------------------------------------------------
printf '\n--- Step 4: GET /status (expect active_engine=ds4) ---\n'
STATUS_BODY=$(curl --silent --max-time 15 "${ROUTER_URL}/status")
ACTIVE=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    print(d.get('active_engine', 'null'))
except Exception as e:
    print('error: ' + str(e))
" "$STATUS_BODY" 2>/dev/null || echo "error")
if [ "$ACTIVE" = "ds4" ]; then
    pass "GET /status: active_engine=ds4"
else
    fail "GET /status: expected active_engine=ds4" "got=${ACTIVE}"
fi

# ---------------------------------------------------------------------------
# Step 5: POST /v1/chat/completions with qwen3.6-uncensored:27b (non-streaming)
#         This FORCES a ds4->ollama swap — allow up to 300s.
# ---------------------------------------------------------------------------
printf '\n--- Step 5: POST /v1/chat/completions (qwen3.6-uncensored:27b, non-streaming) ---\n'
printf '    (forces ds4->ollama swap, allow up to 300s)\n'
CHAT_QWEN_HTTP=$(curl --silent --max-time 300 --output /tmp/smoke_qwen_body.json --write-out '%{http_code}' \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3.6-uncensored:27b","stream":false,"max_tokens":64,"messages":[{"role":"user","content":"say hi in 3 words"}]}' \
    "${ROUTER_URL}/v1/chat/completions")
CHAT_QWEN_BODY=$(cat /tmp/smoke_qwen_body.json 2>/dev/null || echo "{}")
CHAT_QWEN_CONTENT=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    msg = d['choices'][0]['message']
    c = msg.get('content')
    r = msg.get('reasoning_content') or msg.get('reasoning')
    text = c if c else r
    print('ok:' + repr((text or '')[:80]))
except Exception as e:
    print('error: ' + str(e))
" "$CHAT_QWEN_BODY" 2>/dev/null || echo "error: parse failed")
if [ "$CHAT_QWEN_HTTP" = "200" ] && [[ "$CHAT_QWEN_CONTENT" == ok:* ]]; then
    pass "qwen3.6-uncensored:27b chat: HTTP 200 with choices[0].message.content (${CHAT_QWEN_CONTENT})"
else
    fail "qwen3.6-uncensored:27b chat failed" "http=${CHAT_QWEN_HTTP} content=${CHAT_QWEN_CONTENT}"
fi
rm -f /tmp/smoke_qwen_body.json

# ---------------------------------------------------------------------------
# Step 6: GET /status -> active_engine == "ollama" AND ds4.process_running == false
# ---------------------------------------------------------------------------
printf '\n--- Step 6: GET /status (expect active_engine=ollama, ds4.process_running=false) ---\n'
STATUS2_BODY=$(curl --silent --max-time 15 "${ROUTER_URL}/status")
ACTIVE2=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    print(d.get('active_engine', 'null'))
except Exception as e:
    print('error: ' + str(e))
" "$STATUS2_BODY" 2>/dev/null || echo "error")
DS4_RUNNING=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    pr = d['engines']['ds4']['process_running']
    print('true' if pr else 'false')
except Exception as e:
    print('error: ' + str(e))
" "$STATUS2_BODY" 2>/dev/null || echo "error")
if [ "$ACTIVE2" = "ollama" ] && [ "$DS4_RUNNING" = "false" ]; then
    pass "GET /status: active_engine=ollama and ds4.process_running=false"
else
    fail "GET /status after ollama swap" "active_engine=${ACTIVE2} ds4.process_running=${DS4_RUNNING}"
fi

# ---------------------------------------------------------------------------
# Step 7: Streaming check — POST /v1/chat/completions with deepseek-v4-flash,
#         stream:true. This forces an ollama->ds4 swap. First bytes must arrive.
#         Check for "data:" lines and log any ": keepalive" comments seen.
# ---------------------------------------------------------------------------
printf '\n--- Step 7: Streaming (deepseek-v4-flash, stream:true) ---\n'
printf '    (forces ollama->ds4 swap, allow up to 300s)\n'

STREAM_TMP=$(mktemp)
STREAM_HTTP=$(curl --silent --max-time 300 --no-buffer \
    --output "$STREAM_TMP" \
    --write-out '%{http_code}' \
    -H "Content-Type: application/json" \
    -d '{"model":"deepseek-v4-flash","stream":true,"max_tokens":64,"messages":[{"role":"user","content":"say hi in 3 words"}]}' \
    "${ROUTER_URL}/v1/chat/completions")

STREAM_RESULT=$(python3 - "$STREAM_TMP" <<'PYEOF'
import sys

with open(sys.argv[1], 'r', errors='replace') as f:
    lines = f.readlines()

data_lines = [l for l in lines if l.startswith('data:')]
keepalive_lines = [l for l in lines if l.startswith(': keepalive') or l.strip() == ':']

if data_lines:
    print('ok')
    print('data_count=' + str(len(data_lines)))
    print('keepalive_count=' + str(len(keepalive_lines)))
else:
    print('no_data')
    print('data_count=0')
    print('keepalive_count=' + str(len(keepalive_lines)))
    print('first100lines=' + repr(lines[:5]))
PYEOF
)

# Extract values from python output
STREAM_STATUS=$(echo "$STREAM_RESULT" | head -1)
DATA_COUNT=$(echo "$STREAM_RESULT" | grep '^data_count=' | cut -d= -f2)
KEEPALIVE_COUNT=$(echo "$STREAM_RESULT" | grep '^keepalive_count=' | cut -d= -f2)

if [ "$STREAM_HTTP" = "200" ] && [ "$STREAM_STATUS" = "ok" ]; then
    pass "Streaming chat (deepseek-v4-flash): HTTP 200, got ${DATA_COUNT} data: line(s)"
    if [ "${KEEPALIVE_COUNT:-0}" -gt 0 ]; then
        printf '    note: saw %s keepalive comment(s) during swap\n' "$KEEPALIVE_COUNT"
    else
        printf '    note: no keepalive comments seen (swap may have been fast or keepalive disabled)\n'
    fi
else
    fail "Streaming chat failed" "http=${STREAM_HTTP} status=${STREAM_STATUS} data_lines=${DATA_COUNT:-0}"
fi
rm -f "$STREAM_TMP"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
printf '\n============================================================\n'
printf 'SUMMARY: %d passed / %d total\n' "$PASS" "$TOTAL"
printf '============================================================\n'

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
