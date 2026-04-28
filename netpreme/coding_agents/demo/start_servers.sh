#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  Demo: start hybrid-cpu (GPU 1, port 8001) and hybrid-mtier
#  (GPU 0, port 8002) side-by-side using SGLang.
#  MTier MUST be on GPU 0 (cuMemCreate topology — GPU 1 causes slowdown).
#
#  By default runs with full HBM cache (0.90 GPU mem, ~446K tokens).
#  Use --gpu-frac to cap it and force faster evictions.
#
#  SGLang formula: rest_kv = avail_after_load - total_gpu × (1 - frac)
#  With 78.5 GB GPU and ~29 GB model: minimum viable frac ≈ 0.39.
#  0.50 gives ~86K tokens; 0.90 gives ~446K tokens.
#
#  Usage:
#    ./start_servers.sh               # full HBM cache (default)
#    ./start_servers.sh --gpu-frac 0.50  # small cache, forces evictions faster
#    ./start_servers.sh --gpu-frac 0.60
#
#  Then in another terminal:
#    python3 demo/swe_spill_recall.py
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_SCRIPT="${SCRIPT_DIR}/../start_server.sh"
MONITORING_SCRIPT="${SCRIPT_DIR}/../monitoring/start_monitoring.sh"

# ── Kill any stale SGLang processes ─────────────────────────────────
echo "Killing any stale SGLang processes..."
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
sleep 2

# ── Start monitoring (Prometheus) ────────────────────────────────────
echo "Starting Prometheus monitoring..."
bash "$MONITORING_SCRIPT" &
MONITORING_PID=$!
echo "  PID=$MONITORING_PID  log: /tmp/prometheus_sglang.log"
echo ""

# ── Parse flags ──────────────────────────────────────────────────────
GPU_FRAC="0.90"

for arg in "$@"; do
    case "$arg" in
        --gpu-frac=*)   GPU_FRAC="${arg#--gpu-frac=}" ;;
        *) echo "Unknown arg: $arg" >&2
           echo "Usage: $0 [--gpu-frac 0.50]" >&2
           exit 1 ;;
    esac
done

echo ""
echo "Starting demo servers (GPU_MEMORY_UTILIZATION=$GPU_FRAC)..."
echo ""

# ── Start CPU server (GPU 1, port 8001) ──────────────────────────────
echo "[1/2] hybrid-cpu  → GPU 1, port 8001"
CUDA_VISIBLE_DEVICES=1 PORT=8001 GPU_MEMORY_UTILIZATION="$GPU_FRAC" \
    bash "$START_SCRIPT" --hybrid-cpu &
CPU_PID=$!
echo "      PID=$CPU_PID  log: /tmp/sglang_server_8001.log"

# ── Start MTier server (GPU 0, port 8002) ────────────────────────────
# MTier MUST run on GPU 0: cuMemCreate allocates on physical GPU 0.
echo "[2/2] hybrid-mtier → GPU 0, port 8002"
CUDA_VISIBLE_DEVICES=0 PORT=8002 GPU_MEMORY_UTILIZATION="$GPU_FRAC" \
    bash "$START_SCRIPT" --hybrid-mtier &
MTIER_PID=$!
echo "      PID=$MTIER_PID  log: /tmp/sglang_server_8002.log"

echo ""
echo "Waiting for both servers to be healthy..."

wait_healthy() {
    local port=$1 label=$2
    local t0; t0=$(date +%s)
    while true; do
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
            local elapsed=$(( $(date +%s) - t0 ))
            echo "  ✓ $label ready (${elapsed}s)"
            return 0
        fi
        sleep 3
    done
}

wait_healthy 8001 "hybrid-cpu  (port 8001)" &
wait_healthy 8002 "hybrid-mtier(port 8002)" &
wait

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Both servers ready."
echo "  CPU   : http://localhost:8001   GPU 1"
echo "  MTier : http://localhost:8002   GPU 0"
echo "  Metrics: :8001/metrics  :8002/metrics"
echo "  GPU frac: $GPU_FRAC"
echo ""
echo "  Run the demo:  python3 demo/swe_spill_recall.py"
echo "  Prometheus:    http://localhost:9090"
echo "  Ctrl-C to stop both servers."
echo "═══════════════════════════════════════════════════════════════"

cleanup() {
    echo ""
    echo "Stopping demo servers and monitoring..."
    kill "$CPU_PID"        2>/dev/null || true
    kill "$MTIER_PID"      2>/dev/null || true
    kill "$MONITORING_PID" 2>/dev/null || true
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid --format=csv,noheader --id=0 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    nvidia-smi --query-compute-apps=pid --format=csv,noheader --id=1 2>/dev/null | xargs -r kill -9 2>/dev/null || true
    echo "yes" | mtier_service reset 2>/dev/null || true
    echo "Done."
}
trap cleanup INT TERM

wait "$CPU_PID" "$MTIER_PID" "$MONITORING_PID"
