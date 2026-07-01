#!/usr/bin/env bash
# smoke_test.sh — End-to-end smoke test for local-engine-router.
#
# WARNING: This script performs REAL engine swaps. Steps 3, 5, and 7 send
# actual requests to GPU-backed models and will trigger swaps between engines.
# Each swap can take 1-4 minutes. Total runtime is typically 5-15 minutes.
# Do NOT run this while other inference workloads are active.
#
# EDIT BEFORE RUNNING: set MODEL_A and MODEL_B to two real model ids from your
# own config that live on DIFFERENT engines (so a request to each forces a
# swap), and ENGINE_A to the engine key that owns MODEL_A. The defaults match
# config.example.yaml (llama.cpp + Ollama) and will not work unchanged unless
# your config actually serves those ids.
#
# Usage:
#   ROUTER_URL=http://127.0.0.1:8077 \
#   MODEL_A=qwen2.5-7b-instruct ENGINE_A=llamacpp MODEL_B=llama3.1:8b \
#   bash deploy/smoke_test.sh
#
# Requires: curl, python3 (no jq needed).

set -uo pipefail

ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:8077}"
# Two models on two different engines, and the engine that owns MODEL_A.
MODEL_A="${MODEL_A:-qwen2.5-7b-instruct}"
ENGINE_A="${ENGINE_A:-llamacpp}"
MODEL_B="${MODEL_B:-llama3.1:8b}"

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

printf 'Config: ROUTER_URL=%s  MODEL_A=%s (engine %s)  MODEL_B=%s\n' \
    "$ROUTER_URL" "$MODEL_A" "$ENGINE_A" "$MODEL_B"

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
# Step 2: GET /v1/models contains MODEL_A AND MODEL_B
# ---------------------------------------------------------------------------
printf '\n--- Step 2: GET /v1/models ---\n'
MODELS_BODY=$(curl --silent --max-time 15 "${ROUTER_URL}/v1/models")
HAS_A=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    ids = [m.get('id','') for m in d.get('data', [])]
    print('yes' if sys.argv[2] in ids else 'no')
except Exception as e:
    print('error: ' + str(e))
" "$MODELS_BODY" "$MODEL_A" 2>/dev/null || echo "error")
HAS_B=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    ids = [m.get('id','') for m in d.get('data', [])]
    print('yes' if sys.argv[2] in ids else 'no')
except Exception as e:
    print('error: ' + str(e))
" "$MODELS_BODY" "$MODEL_B" 2>/dev/null || echo "error")
if [ "$HAS_A" = "yes" ] && [ "$HAS_B" = "yes" ]; then
    pass "GET /v1/models contains ${MODEL_A} and ${MODEL_B}"
else
    fail "GET /v1/models missing expected models" "has_${MODEL_A}=${HAS_A} has_${MODEL_B}=${HAS_B}"
fi

# ---------------------------------------------------------------------------
# Step 3: POST /v1/chat/completions with MODEL_A (non-streaming)
#         This may trigger a swap to ENGINE_A — allow up to 300s.
# ---------------------------------------------------------------------------
printf '\n--- Step 3: POST /v1/chat/completions (%s, non-streaming) ---\n' "$MODEL_A"
printf '    (may trigger a swap to %s, allow up to 300s)\n' "$ENGINE_A"
CHAT_A_TMP=$(mktemp)
CHAT_A_HTTP=$(curl --silent --max-time 300 --output "$CHAT_A_TMP" --write-out '%{http_code}' \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_A}\",\"stream\":false,\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"say hi in 3 words\"}]}" \
    "${ROUTER_URL}/v1/chat/completions")
CHAT_A_BODY=$(cat "$CHAT_A_TMP" 2>/dev/null || echo "{}")
rm -f "$CHAT_A_TMP"
CHAT_A_CONTENT=$(python3 -c "
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
" "$CHAT_A_BODY" 2>/dev/null || echo "error: parse failed")
if [ "$CHAT_A_HTTP" = "200" ] && [[ "$CHAT_A_CONTENT" == ok:* ]]; then
    pass "${MODEL_A} chat: HTTP 200 with choices[0].message.content present (${CHAT_A_CONTENT})"
else
    fail "${MODEL_A} chat failed" "http=${CHAT_A_HTTP} content=${CHAT_A_CONTENT}"
fi

# ---------------------------------------------------------------------------
# Step 4: GET /status -> active_engine == ENGINE_A
# ---------------------------------------------------------------------------
printf '\n--- Step 4: GET /status (expect active_engine=%s) ---\n' "$ENGINE_A"
STATUS_BODY=$(curl --silent --max-time 15 "${ROUTER_URL}/status")
ACTIVE=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    print(d.get('active_engine', 'null'))
except Exception as e:
    print('error: ' + str(e))
" "$STATUS_BODY" 2>/dev/null || echo "error")
if [ "$ACTIVE" = "$ENGINE_A" ]; then
    pass "GET /status: active_engine=${ENGINE_A}"
else
    fail "GET /status: expected active_engine=${ENGINE_A}" "got=${ACTIVE}"
fi

# ---------------------------------------------------------------------------
# Step 5: POST /v1/chat/completions with MODEL_B (non-streaming)
#         This FORCES a swap away from ENGINE_A — allow up to 300s.
# ---------------------------------------------------------------------------
printf '\n--- Step 5: POST /v1/chat/completions (%s, non-streaming) ---\n' "$MODEL_B"
printf '    (forces a swap away from %s, allow up to 300s)\n' "$ENGINE_A"
CHAT_B_TMP=$(mktemp)
CHAT_B_HTTP=$(curl --silent --max-time 300 --output "$CHAT_B_TMP" --write-out '%{http_code}' \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_B}\",\"stream\":false,\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"say hi in 3 words\"}]}" \
    "${ROUTER_URL}/v1/chat/completions")
CHAT_B_BODY=$(cat "$CHAT_B_TMP" 2>/dev/null || echo "{}")
CHAT_B_CONTENT=$(python3 -c "
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
" "$CHAT_B_BODY" 2>/dev/null || echo "error: parse failed")
if [ "$CHAT_B_HTTP" = "200" ] && [[ "$CHAT_B_CONTENT" == ok:* ]]; then
    pass "${MODEL_B} chat: HTTP 200 with choices[0].message.content (${CHAT_B_CONTENT})"
else
    fail "${MODEL_B} chat failed" "http=${CHAT_B_HTTP} content=${CHAT_B_CONTENT}"
fi
rm -f "$CHAT_B_TMP"

# ---------------------------------------------------------------------------
# Step 6: GET /status -> active_engine swapped away from ENGINE_A
# ---------------------------------------------------------------------------
printf '\n--- Step 6: GET /status (expect active_engine != %s) ---\n' "$ENGINE_A"
STATUS2_BODY=$(curl --silent --max-time 15 "${ROUTER_URL}/status")
ACTIVE2=$(python3 -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
    print(d.get('active_engine', 'null'))
except Exception as e:
    print('error: ' + str(e))
" "$STATUS2_BODY" 2>/dev/null || echo "error")
if [ "$ACTIVE2" != "$ENGINE_A" ] && [ "$ACTIVE2" != "error" ] && [ "$ACTIVE2" != "null" ]; then
    pass "GET /status: active_engine=${ACTIVE2} (swapped away from ${ENGINE_A})"
else
    fail "GET /status after second swap" "active_engine=${ACTIVE2} (expected != ${ENGINE_A})"
fi

# ---------------------------------------------------------------------------
# Step 7: Streaming check — POST /v1/chat/completions with MODEL_A, stream:true.
#         This forces a swap back to ENGINE_A. First bytes must arrive.
#         Check for "data:" lines and log any ": keepalive" comments seen.
# ---------------------------------------------------------------------------
printf '\n--- Step 7: Streaming (%s, stream:true) ---\n' "$MODEL_A"
printf '    (forces a swap back to %s, allow up to 300s)\n' "$ENGINE_A"

STREAM_TMP=$(mktemp)
STREAM_HTTP=$(curl --silent --max-time 300 --no-buffer \
    --output "$STREAM_TMP" \
    --write-out '%{http_code}' \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_A}\",\"stream\":true,\"max_tokens\":64,\"messages\":[{\"role\":\"user\",\"content\":\"say hi in 3 words\"}]}" \
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
    pass "Streaming chat (${MODEL_A}): HTTP 200, got ${DATA_COUNT} data: line(s)"
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
