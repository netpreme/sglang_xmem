#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Launch SGLang for Claude Code — hierarchical KV cache support.
#
#  Architecture:
#
#    Claude Code
#      │  POST /v1/messages   (native Anthropic API format)
#      │  ANTHROPIC_BASE_URL=http://localhost:8000
#      ▼
#    SGLang Server (:8000)
#      │  python -m sglang.launch_server
#      │  OpenAI-compatible /v1/chat/completions + /v1/messages
#      ▼
#    SGLang Engine (owns GPU memory)
#      Hierarchical KV cache: GPU HBM → CPU DRAM or MTier chip
#
# ───────────────────────────────────────────────────────────────
#  KV Cache Modes
# ───────────────────────────────────────────────────────────────
#
#  (default) --hbm-only
#    All KV stays in GPU HBM via SGLang's radix tree prefix cache.
#    No host offloading.
#
#  --hybrid-cpu
#    GPU HBM prefix cache + CPU DRAM overflow (hierarchical cache).
#    Hot blocks in GPU; evicted blocks spill to CPU DRAM pinned memory.
#    Flags: --enable-hierarchical-cache --hicache-size 60
#
#  --hybrid-mtier
#    GPU HBM prefix cache + MTier chip overflow (hierarchical cache).
#    Hot blocks in GPU; evicted blocks spill to MTier DRAM.
#    Flags: --enable-hierarchical-cache --hicache-size 60 --hicache-use-xmem
#
#  Note: cpu-only and mtier-only modes (GPU prefix cache disabled)
#  are not natively supported by SGLang's hierarchical cache.
#  Use vllm_xmem for those modes.
#
# ───────────────────────────────────────────────────────────────
#  Usage:
#    ./start_server.sh                        # hbm-only (default)
#    ./start_server.sh --hbm-only             # GPU HBM only
#    ./start_server.sh --hybrid-cpu           # GPU HBM + CPU DRAM
#    ./start_server.sh --hybrid-mtier         # GPU HBM + MTier chip
#    ./start_server.sh --port 8001            # custom port
#    ./start_server.sh --gpu-util 0.60        # override GPU mem fraction
#    ./start_server.sh --hybrid-cpu --port 8001
#
#  Connect Claude Code (separate terminal after server is ready):
#    ./run_claude.sh
#
#  Stop:   Ctrl-C
#  Log:    tail -f /tmp/sglang_server_<PORT>.log
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/../.env"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python3.12}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3 2>/dev/null || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
    echo "ERROR: No Python interpreter found." >&2
    exit 1
fi

# ── Load .env ────────────────────────────────────────────────────
if [[ -f "$ENV_FILE" ]]; then
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        [[ -v "$key" ]] || export "$key=$value"
    done < "$ENV_FILE"
else
    echo "WARNING: .env not found at $ENV_FILE, using built-in defaults." >&2
fi

# ── Parse args ───────────────────────────────────────────────────
MODE="hybrid-mtier"
GPU_UTIL_OVERRIDE=""
PORT_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hbm-only)     MODE="hbm-only" ;;
        --hybrid-cpu)   MODE="hybrid-cpu" ;;
        --hybrid-mtier) MODE="hybrid-mtier" ;;
        --gpu-util|--gpu-util=*)
            [[ "$1" == *=* ]] && GPU_UTIL_OVERRIDE="${1#*=}" || { shift; GPU_UTIL_OVERRIDE="$1"; }
            ;;
        --port|--port=*)
            [[ "$1" == *=* ]] && PORT_OVERRIDE="${1#*=}" || { shift; PORT_OVERRIDE="$1"; }
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            echo "       Valid modes: --hbm-only | --hybrid-cpu | --hybrid-mtier" >&2
            echo "       Options:     --gpu-util <0.0-1.0>  --port <number>" >&2
            exit 1 ;;
    esac
    shift
done

# ── Core settings ────────────────────────────────────────────────
MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"
HOST="${HOST:-0.0.0.0}"

if [[ -z "${PORT_OVERRIDE:-}" && -z "${PORT:-}" ]]; then
    case "$MODE" in
        hbm-only)     _DEFAULT_PORT=8000 ;;
        hybrid-cpu)   _DEFAULT_PORT=8001 ;;
        hybrid-mtier) _DEFAULT_PORT=8002 ;;
        *)            _DEFAULT_PORT=8000 ;;
    esac
else
    _DEFAULT_PORT=8000
fi
PORT="${PORT_OVERRIDE:-${PORT:-$_DEFAULT_PORT}}"

MEM_FRACTION="${GPU_UTIL_OVERRIDE:-${GPU_MEMORY_UTILIZATION:-0.90}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
TP="${TENSOR_PARALLEL_SIZE:-1}"
PAGE_SIZE="${PAGE_SIZE:-64}"
DTYPE="${DTYPE:-auto}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-auto}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-1}"

# Auto-assign GPU: mtier → GPU 0, cpu → GPU 1, hbm-only → GPU 0
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    case "$MODE" in
        *mtier*) CVD="0" ;;
        hybrid-cpu) CVD="1" ;;
        *) CVD="0" ;;
    esac
else
    CVD="$CUDA_VISIBLE_DEVICES"
fi

SGLANG_LOG="/tmp/sglang_server_${PORT}.log"

# ── Hierarchical cache flags by mode ─────────────────────────────
HICACHE_FLAGS=""
case "$MODE" in
    hbm-only)
        ;;
    hybrid-cpu)
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-size 60 --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout layer_first"
        ;;
    hybrid-mtier)
        HICACHE_FLAGS="--enable-hierarchical-cache --hicache-size 68 --hicache-use-xmem --hicache-write-policy write_through --hicache-io-backend kernel --hicache-mem-layout layer_first"
        ;;
esac

# ── Print config ─────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Backend : SGLang"
echo "  Model   : $MODEL"
echo "  Host    : $HOST"
echo "  Port    : $PORT  (/v1/messages + /v1/chat/completions)"
echo "  GPUs    : $CVD"
echo "  TP      : $TP"
echo "  Max len : $MAX_MODEL_LEN"
echo "  GPU mem : $MEM_FRACTION"
echo "  Mode    : $MODE"
echo "  Log     : $SGLANG_LOG"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── Kill any stale process on our port ──────────────────────────
# Use curl without -f so we catch both 200 (healthy) and 503 (shutting down)
if curl -s --max-time 3 "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "WARNING: port $PORT already in use — killing stale SGLang processes..."
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 5
fi

# ── Cleanup ──────────────────────────────────────────────────────
SGLANG_PID=""
TAIL_PID=""

cleanup() {
    echo ""
    echo "Stopping SGLang..."
    [[ -n "$TAIL_PID"   ]] && kill "$TAIL_PID"   2>/dev/null || true
    [[ -n "$SGLANG_PID" ]] && kill "$SGLANG_PID" 2>/dev/null || true
    wait 2>/dev/null || true

    echo "Freeing GPU memory on device(s): $CVD ..."
    for gpu_id in ${CVD//,/ }; do
        nvidia-smi --query-compute-apps=pid --format=csv,noheader --id="$gpu_id" 2>/dev/null \
            | xargs -r kill -9 2>/dev/null || true
    done

    if [[ "$MODE" == *mtier* ]]; then
        echo "Resetting MTier memory..."
        echo "yes" | mtier_service reset 2>/dev/null || true
    fi

    echo "Done."
    exit 0
}
trap cleanup INT TERM

# ── Start SGLang server ──────────────────────────────────────────
echo "[1/2] Starting SGLang server on GPU(s) $CVD ..."
CUDA_VISIBLE_DEVICES="$CVD" \
HF_HUB_OFFLINE=1 \
    "$PYTHON_BIN" -m sglang.launch_server \
        --model-path "$MODEL" \
        --served-model-name "$MODEL" \
        --host "$HOST" \
        --port "$PORT" \
        --tp "$TP" \
        --dtype "$DTYPE" \
        --context-length "$MAX_MODEL_LEN" \
        --kv-cache-dtype "$KV_CACHE_DTYPE" \
        --mem-fraction-static "$MEM_FRACTION" \
        --page-size "$PAGE_SIZE" \
        --max-running-requests "$MAX_NUM_SEQS" \
        --max-prefill-tokens 65536 \
        --enable-metrics \
        --enable-cache-report \
        --tool-call-parser qwen25 \
        --sampling-defaults openai \
        ${DISABLE_CUDA_GRAPH:+--disable-cuda-graph} \
        $HICACHE_FLAGS \
        > "$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!
echo "      PID $SGLANG_PID  (log: $SGLANG_LOG)"

# ── Wait for health ───────────────────────────────────────────────
echo ""
echo "[2/2] Waiting for SGLang to be ready — streaming log:"
echo ""
tail -f "$SGLANG_LOG" &
TAIL_PID=$!

until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
    if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
        kill "$TAIL_PID" 2>/dev/null || true
        echo ""
        echo "ERROR: SGLang died. Check: tail $SGLANG_LOG"
        cleanup
    fi
    sleep 2
done
kill "$TAIL_PID" 2>/dev/null || true
TAIL_PID=""
echo ""
echo "  SGLang ready!"
echo ""
echo "  Claude Code:  ANTHROPIC_BASE_URL=http://localhost:$PORT"
echo "  Launch:       ./run_claude.sh"
echo "  Log:          tail -f $SGLANG_LOG"
echo "  PID:          sglang=$SGLANG_PID"
echo "  Ctrl-C to stop."
echo ""

# ── Tail log ─────────────────────────────────────────────────────
tail -f "$SGLANG_LOG" &
TAIL_PID=$!

while true; do
    if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
        echo ""
        echo "SGLang process ($SGLANG_PID) exited. Cleaning up..."
        break
    fi
    sleep 5
done
cleanup
