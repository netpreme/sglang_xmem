#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════
#  Start Prometheus for SGLang monitoring.
#  No Grafana / kv_exporter needed for basic bench monitoring.
#
#  Usage:
#    ./start_monitoring.sh            # fresh start (wipes data)
#    ./start_monitoring.sh --keep     # keep existing data
# ═══════════════════════════════════════════════════════════
set -euo pipefail

MONITORING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROM_DATA_DIR="/tmp/prometheus_sglang_data"
KEEP_DATA="${1:-}"

# ── 1. Kill any existing Prometheus on default port ─────────
pkill -f "prometheus.*sglang" 2>/dev/null || true
sleep 1

# ── 2. Reset data ────────────────────────────────────────────
if [[ "$KEEP_DATA" == "--keep" ]]; then
    echo "Keeping existing data at $PROM_DATA_DIR"
else
    echo "Wiping Prometheus data at $PROM_DATA_DIR ..."
    rm -rf "$PROM_DATA_DIR"
fi
mkdir -p "$PROM_DATA_DIR"

# ── 3. Start Prometheus ──────────────────────────────────────
prometheus \
    --config.file="$MONITORING_DIR/prometheus.yml" \
    --storage.tsdb.path="$PROM_DATA_DIR" \
    --storage.tsdb.retention.time=1d \
    --web.listen-address=":9090" \
    > /tmp/prometheus_sglang.log 2>&1 &
PROM_PID=$!
echo "Prometheus started (pid $PROM_PID) → http://localhost:9090"
echo "  log: /tmp/prometheus_sglang.log"
echo "  scraping: port 8000 (hbm-only), 8001 (hybrid-cpu), 8002 (hybrid-mtier)"
echo ""
echo "Metrics: http://localhost:9090/api/v1/query?query=sglang:time_to_first_token_seconds_count"
echo ""
echo "Press Ctrl+C to stop."

cleanup() {
    echo ""
    echo "Stopping Prometheus..."
    kill "$PROM_PID" 2>/dev/null || true
    wait "$PROM_PID" 2>/dev/null || true
    echo "Done."
}
trap cleanup INT TERM

wait "$PROM_PID"
