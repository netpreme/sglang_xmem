#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Launch Claude Code pointed at the local SGLang server.
#
#  Usage:
#    ./run_claude.sh                          # interactive session
#    ./run_claude.sh --start                  # start server first, then interactive
#    ./run_claude.sh --start --hybrid-cpu     # start hybrid-cpu server, then interactive
#    ./run_claude.sh --start --hybrid-mtier   # start hybrid-mtier server, then interactive
#    ./run_claude.sh --auto "fix the bug"     # autonomous: run task
#    ./run_claude.sh -- --resume              # pass flags to claude
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3.12}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3 2>/dev/null || true)"
fi

# ── Load .env ────────────────────────────────────────────────────
if [[ -f "$ENV_FILE" ]]; then
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        [[ -v "$key" ]] || export "$key=$value"
    done < "$ENV_FILE"
fi

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
PORT="${PORT:-8002}"

# ── Parse flags ──────────────────────────────────────────────────
START_SERVER=0
AUTO_TASK=""
CLAUDE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --start)        START_SERVER=1 ;;
        --auto)
            AUTO_TASK="$2"; shift ;;
        --port|--port=*)
            [[ "$1" == *=* ]] && PORT="${1#*=}" || { shift; PORT="$1"; } ;;
        --)
            shift; CLAUDE_ARGS+=("$@"); break ;;
        *)
            CLAUDE_ARGS+=("$1") ;;
    esac
    shift
done

BASE_URL="http://localhost:${PORT}"

# ── Optionally start the server ──────────────────────────────────
if [[ "$START_SERVER" == "1" ]]; then
    echo "Starting SGLang server in background..."
    bash "${SCRIPT_DIR}/start_server.sh" &
    echo "  server pid: $!"
fi

# ── Wait for health ───────────────────────────────────────────────
echo "Waiting for SGLang at ${BASE_URL}/health ..."
until curl -sf "$BASE_URL/health" > /dev/null 2>&1; do
    printf "."
    sleep 2
done
echo " ready!"
echo ""

# ── Launch ───────────────────────────────────────────────────────
COMMON_ENV=(
    ANTHROPIC_BASE_URL="$BASE_URL"
    ANTHROPIC_API_KEY="dummy"
    ANTHROPIC_AUTH_TOKEN="dummy"
    ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
    ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
    ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
)

if [[ -n "$AUTO_TASK" ]]; then
    echo "Running autonomous Claude Code"
    echo "  ANTHROPIC_BASE_URL = $BASE_URL"
    echo "  model              = $MODEL"
    echo "  task               = $AUTO_TASK"
    echo ""
    env "${COMMON_ENV[@]}" \
        claude --model "$MODEL" \
               --dangerously-skip-permissions \
               --verbose \
               --output-format stream-json \
               -p "$AUTO_TASK" \
      | python3 "${SCRIPT_DIR}/monitoring/parse_stream.py"
else
    echo "Launching Claude Code"
    echo "  ANTHROPIC_BASE_URL = $BASE_URL"
    echo "  model              = $MODEL"
    echo ""
    exec env "${COMMON_ENV[@]}" \
        claude --model "$MODEL" "${CLAUDE_ARGS[@]}"
fi
