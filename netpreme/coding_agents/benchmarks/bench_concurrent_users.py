#!/usr/bin/env python3
"""
Concurrent SWE-bench benchmark with per-turn metrics — SGLang backend.

Runs N concurrent claude sessions on SWE-bench tasks and captures
per-turn ISL/OSL/cached-tokens/cache-hit-rate/E2E from stream-json output.
Prometheus is used for TTFT/ITL histograms.

Supported setups (SGLang hierarchical KV cache):
  hbm-only      — GPU HBM only (no offload)
  hybrid-cpu    — GPU HBM + CPU DRAM hierarchical cache
  hybrid-mtier  — GPU HBM + MTier chip hierarchical cache

Note: cpu-only and mtier-only are not natively supported by SGLang's
hierarchical cache. Use vllm_xmem for those modes.

All modes default to CUDA_VISIBLE_DEVICES=0 (override with SGLANG_GPU_IDX env var).

Usage:
    python3 bench_concurrent_users.py --concurrency 1 2 4 8 16

    python3 bench_concurrent_users.py --setup hybrid-cpu --concurrency 4 8 16

    python3 bench_concurrent_users.py --concurrency 1 2 4 8 16 --sustained

    python3 bench_concurrent_users.py --setup hbm-only hybrid-cpu hybrid-mtier --concurrency 4 8

Results saved to results_benchmarks/bench_<setup>_<timestamp>/
    per_turn.csv        — one row per assistant turn (ISL/OSL/cache/E2E)
    summary.csv         — one row per concurrency level
    c<N>.json           — full detail per level
"""

import argparse
import atexit
import concurrent.futures
import csv
import json
import os
import queue as _queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _PYNVML_OK = True
except Exception:
    _PYNVML_OK = False

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = Path(__file__).resolve().parent
ENV_FILE        = SCRIPT_DIR.parent / ".env"
WORKSPACE_ROOT  = Path("/tmp/swe_workspaces")
RESULTS_DIR     = SCRIPT_DIR / "results_benchmarks"
REPO_ROOT       = SCRIPT_DIR.parent.parent.parent

DEFAULT_MODEL        = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
DEFAULT_PORT         = "8000"
DEFAULT_PROM         = "http://localhost:9090"
PROM_SCRAPE_BUFFER_S = 5
CLONE_WORKERS        = 16
SUSTAINED_MIN_S      = 600
SERVER_STARTUP_TIMEOUT_S = 480  # sglang torch.compile adds ~2 min

# GPU index for all sglang modes (CUDA_VISIBLE_DEVICES)
SGLANG_GPU_IDX = os.environ.get("SGLANG_GPU_IDX", "0")

# Hierarchical cache flags per mode
_HICACHE_FLAGS: dict[str, list[str]] = {
    "hbm-only":     [],
    # write_through: backup happens on first hit (cache_finished_req at request
    # completion). Combined with CLAUDE_CODE_ATTRIBUTION_HEADER=0 in the env
    # below, this gets us cache hits across multi-turn agent traffic.
    # NOTE: write_back triggers an SGLang assertion bug in evict_host() under
    # heavy concurrent load (load_back → evict → write_backup → evict_host →
    # AssertionError "parent does not have child key"). write_through avoids
    # that cascade.
    "hybrid-cpu":   [
        "--enable-hierarchical-cache",
        "--hicache-size", "70",
        "--hicache-write-policy", "write_through",
        "--hicache-io-backend", "kernel",
        "--hicache-mem-layout", "layer_first",
    ],
    "hybrid-mtier": [
        "--enable-hierarchical-cache",
        "--hicache-size", "70",
        "--hicache-use-xmem",
        "--hicache-write-policy", "write_through",
        "--hicache-io-backend", "kernel",
        "--hicache-mem-layout", "layer_first",
    ],
}

VALID_SETUPS = set(_HICACHE_FLAGS.keys())

# ── live progress counter ─────────────────────────────────────────────────────
_live: dict    = {}
_live_lock     = threading.Lock()

# ── active server info (set in main, used by cleanup) ─────────────────────────
_active_port:     "int | None" = None
_active_setup:    "str | None" = None
_server_stopping: bool         = False


def _redraw_live() -> None:
    with _live_lock:
        c        = _live.get("c", "?")
        cfg      = _live.get("cfg", "")
        elap     = _live.get("elapsed", 0.0)
        done     = _live.get("done", 0)
        ok       = _live.get("ok", 0)
        fail     = _live.get("fail", 0)
        active   = _live.get("active", 0)
        cloned   = _live.get("cloned", 0)
        ttft_p50 = _live.get("ttft_p50")
        ttft_p95 = _live.get("ttft_p95")
    ttft_str = ""
    if ttft_p50 is not None:
        ttft_str = ("  ttft_p50=%sms" % int(ttft_p50))
        if ttft_p95 is not None:
            ttft_str += ("  p95=%sms" % int(ttft_p95))
    print(
        f"\r  C={c} [{cfg}]  {elap:>5.0f}s  "
        f"done={done}  ok={ok}  fail={fail}  active={active}  cloned={cloned}"
        + ttft_str + "   ",
        end="", flush=True,
    )


# ── server management ─────────────────────────────────────────────────────────

def _find_python() -> str:
    """Return the Python interpreter to use for launching SGLang."""
    candidates = [
        str(REPO_ROOT / ".venv" / "bin" / "python3.12"),
        str(REPO_ROOT / ".venv" / "bin" / "python3"),
        "python3",
    ]
    for c in candidates:
        if Path(c).exists() and os.access(c, os.X_OK):
            return c
    return "python3"


def start_server(setup: str, port: int, tp: int | None = None, gpu_util: float | None = None,
                 gpus: str | None = None, max_running_reqs: int | None = None) -> subprocess.Popen:
    """
    Start SGLang server directly via python -m sglang.launch_server.
    Blocks until /health responds, then returns the Popen handle.
    """
    if setup not in VALID_SETUPS:
        raise ValueError(f"Invalid setup {setup!r}. Valid: {VALID_SETUPS}")

    log = Path(f"/tmp/sglang_server_{port}.log")
    cvd = gpus or SGLANG_GPU_IDX

    print(f"\n  [server] Starting SGLang  setup={setup}  tp={tp or 1}  "
          f"gpus={cvd}  gpu_util={gpu_util or 0.75}  "
          f"max_running_reqs={max_running_reqs or 'default'}  log={log}", flush=True)

    load_env(ENV_FILE)
    model        = os.environ.get("MODEL", DEFAULT_MODEL)
    mem_fraction = str(gpu_util or float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.75")))
    max_len      = os.environ.get("MAX_MODEL_LEN", "131072")
    page_size    = os.environ.get("PAGE_SIZE", "64")
    tp_size      = str(tp or int(os.environ.get("TENSOR_PARALLEL_SIZE", "1")))

    cmd = [
        _find_python(), "-m", "sglang.launch_server",
        "--model-path", model,
        "--served-model-name", model,
        "--host", "0.0.0.0",
        "--port", str(port),
        "--tp", tp_size,
        "--context-length", max_len,
        "--mem-fraction-static", mem_fraction,
        "--page-size", page_size,
        "--dtype", "auto",
        "--kv-cache-dtype", "auto",
        "--max-prefill-tokens", "131072",
        "--tool-call-parser", "qwen",
        "--sampling-defaults", "openai",
        "--disable-cuda-graph",
        "--enable-metrics",
        "--enable-cache-report",
    ]
    if max_running_reqs is not None:
        cmd += ["--max-running-requests", str(max_running_reqs)]

    cmd += _HICACHE_FLAGS[setup]

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": cvd, "HF_HUB_OFFLINE": "1"}

    log.parent.mkdir(parents=True, exist_ok=True)
    log_fp_write = open(log, "w")

    proc = subprocess.Popen(
        cmd,
        stdout=log_fp_write,
        stderr=log_fp_write,
        start_new_session=True,
        env=env,
    )
    log_fp_write.close()

    t0 = time.monotonic()
    _log_fp = None
    while True:
        if _log_fp is None:
            try:
                _log_fp = open(log, "r", errors="replace")
            except OSError:
                pass

        if _log_fp is not None:
            while True:
                line = _log_fp.readline()
                if not line:
                    break
                stripped = line.rstrip("\n")
                if stripped:
                    print(f"  {stripped}", flush=True)

        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                if _log_fp is not None:
                    _log_fp.close()
                print(f"  [server] Ready  ({time.monotonic()-t0:.0f}s)  PID={proc.pid}",
                      flush=True)
                return proc
        except Exception:
            pass

        if proc.poll() is not None:
            if _log_fp is not None:
                _log_fp.close()
            raise RuntimeError(f"SGLang server process died during startup (see {log})")

        if time.monotonic() - t0 > SERVER_STARTUP_TIMEOUT_S:
            proc.terminate()
            if _log_fp is not None:
                _log_fp.close()
            raise RuntimeError(f"SGLang server startup timed out after {SERVER_STARTUP_TIMEOUT_S}s")

        time.sleep(3)


def stop_server(proc: subprocess.Popen, setup: str, port: int = 8000) -> None:
    global _server_stopping
    _server_stopping = True
    print(f"  [server] Stopping  setup={setup}  port={port} ...", flush=True)

    # SGLang's graceful shutdown hangs on in-flight requests; skip it.
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass

    for _ in range(10):
        time.sleep(0.5)
        try:
            os.killpg(os.getpgid(proc.pid), 0)
        except ProcessLookupError:
            break

    # Belt-and-suspenders: kill anything left on the port, via pkill pattern
    # match on command line (ss/fuser/lsof may not be installed).
    subprocess.run(
        ["pkill", "-9", "-f", f"sglang.launch_server.*--port {port}"],
        capture_output=True,
    )
    subprocess.run(
        ["pkill", "-9", "-f", f"sglang.*--port {port}"],
        capture_output=True,
    )
    time.sleep(2)

    if "mtier" in setup:
        subprocess.run(["sh", "-c", "echo yes | mtier_service reset 2>/dev/null || true"],
                       capture_output=True)
        time.sleep(2)

    print(f"  [server] Stopped.", flush=True)


# ── per-turn record ────────────────────────────────────────────────────────────

@dataclass
class TurnRecord:
    config: str
    concurrency: int
    instance_id: str
    turn_idx: int
    isl: int
    osl: int
    turn_e2e_ms: float
    ttft_ms: Optional[float]
    itl_ms: Optional[float]
    queue_ms: Optional[float]
    gpu_hit_pct: Optional[float]
    cpu_hit_pct: Optional[float]
    response_type: str


def _classify(has_text: bool, has_tools: bool) -> str:
    if has_text and has_tools:
        return "mixed"
    if has_tools:
        return "tool_call"
    return "text"


# ── environment ───────────────────────────────────────────────────────────────

def load_env(path: Path) -> None:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() and key.strip() not in os.environ:
                os.environ[key.strip()] = value.strip()


# ── SGLang /metrics scraper ───────────────────────────────────────────────────

_METRICS_LINE_RE  = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(?:\{([^}]*)\})?'
    r'\s+([+-]?(?:[0-9]*\.)?[0-9]+(?:[eE][+-]?[0-9]+)?|NaN|[+-]?Inf)'
)
_METRICS_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')
Scraped = dict[str, list[tuple[dict[str, str], float]]]


def scrape_server(url: str) -> Scraped:
    result: Scraped = {}
    try:
        r = requests.get(f"{url}/metrics", timeout=2)
        for line in r.text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            m = _METRICS_LINE_RE.match(line)
            if not m:
                continue
            name, labels_str, value_str = m.groups()
            if value_str in ("NaN", "Inf", "+Inf", "-Inf"):
                continue
            try:
                value = float(value_str)
            except ValueError:
                continue
            labels: dict[str, str] = {}
            if labels_str:
                for kv in _METRICS_LABEL_RE.finditer(labels_str):
                    labels[kv.group(1)] = kv.group(2)
            result.setdefault(name, []).append((labels, value))
    except Exception:
        pass
    return result


def _sum_scraped(s: Scraped, name: str) -> float:
    return sum(v for _, v in s.get(name, []))


def _scraped_by_label(s: Scraped, name: str, label_key: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for labels, value in s.get(name, []):
        k = labels.get(label_key, "unknown")
        result[k] = result.get(k, 0.0) + value
    return result


def _snapshot_offload(url: str) -> dict:
    """Snapshot SGLang KV offload counters (token counts, not bytes)."""
    s = scrape_server(url)
    # evicted_tokens_total: device→host transfers; load_back_tokens_total: host→device
    return {
        "evicted":   _sum_scraped(s, "sglang:evicted_tokens_total"),
        "load_back": _sum_scraped(s, "sglang:load_back_tokens_total"),
    }


def _snap_turn_metrics(url: str) -> dict:
    """Snapshot per-turn latency + cache counters from SGLang /metrics (c=1 only).

    Cache hit rate uses token-level counts: cached_tokens / prompt_tokens.
    sglang does not expose per-request prefill time as a separate metric.
    """
    s = scrape_server(url)
    # cached_tokens_total by cache_source label
    cached_by_source = _scraped_by_label(s, "sglang:cached_tokens_total", "cache_source")
    return {
        "ttft_s":   _sum_scraped(s, "sglang:time_to_first_token_seconds_sum"),
        "ttft_n":   _sum_scraped(s, "sglang:time_to_first_token_seconds_count"),
        "itl_s":    _sum_scraped(s, "sglang:inter_token_latency_seconds_sum"),
        "itl_n":    _sum_scraped(s, "sglang:inter_token_latency_seconds_count"),
        "q_s":      _sum_scraped(s, "sglang:queue_time_seconds_sum"),
        "q_n":      _sum_scraped(s, "sglang:queue_time_seconds_count"),
        # GPU (device) cache hits in tokens
        "gpu_hits": cached_by_source.get("device", 0.0),
        # Host (CPU/MTier) cache hits in tokens
        "cpu_hits": cached_by_source.get("host", 0.0),
        # Total prompt tokens = denominator for hit rates
        "prompt_tokens": _sum_scraped(s, "sglang:prompt_tokens_total"),
    }


def _delta_turn_metrics(b: dict, a: dict) -> dict:
    def _avg_ms(sk, ck):
        ds, dc = a[sk] - b[sk], a[ck] - b[ck]
        return round(ds / dc * 1000, 1) if dc > 0 else None
    d_prompt = a["prompt_tokens"] - b["prompt_tokens"]
    return {
        "ttft_ms":    _avg_ms("ttft_s", "ttft_n"),
        "itl_ms":     _avg_ms("itl_s",  "itl_n"),
        "queue_ms":   _avg_ms("q_s",    "q_n"),
        # Token-level hit rates (fraction of prompt tokens served from cache)
        "gpu_hit_pct": round((a["gpu_hits"] - b["gpu_hits"]) / d_prompt * 100, 1)
                       if d_prompt > 0 else None,
        "cpu_hit_pct": round((a["cpu_hits"] - b["cpu_hits"]) / d_prompt * 100, 1)
                       if d_prompt > 0 else None,
    }


def _delta_offload(before: dict, after: dict) -> dict:
    return {
        "evicted_tokens":   round(after["evicted"]   - before["evicted"],   0),
        "load_back_tokens": round(after["load_back"] - before["load_back"], 0),
    }


# ── Prometheus helpers ────────────────────────────────────────────────────────

def prom_scalar(prom: str, q: str, time_unix: Optional[float] = None) -> Optional[float]:
    params: dict = {"query": q}
    if time_unix is not None:
        params["time"] = time_unix
    try:
        r = requests.get(f"{prom}/api/v1/query", params=params, timeout=5)
        results = r.json()["data"]["result"]
        if results:
            v = results[0]["value"][1]
            return None if v in ("NaN", "Inf", "-Inf") else float(v)
    except Exception:
        pass
    return None


def query_percentiles(prom: str, bucket_metric: str, duration_s: float,
                      multiplier: float = 1000.0,
                      time_unix: Optional[float] = None,
                      prom_instance: Optional[str] = None) -> dict[str, Optional[float]]:
    d = max(15, int(duration_s))
    inst_sel = ('instance="%s"' % prom_instance) if prom_instance else ""
    sel = "{%s}" % inst_sel
    m = f"{bucket_metric}{sel}" if prom_instance else bucket_metric
    result: dict[str, Optional[float]] = {}
    for q, label in [(0.5, "p50"), (0.9, "p90"), (0.95, "p95"), (0.99, "p99")]:
        v = prom_scalar(prom, f"histogram_quantile({q}, sum(rate({m}[{d}s])) by (le))",
                        time_unix=time_unix)
        result[label] = round(v * multiplier, 1) if v is not None else None
    return result


def query_rate(prom: str, metric: str, duration_s: float,
               time_unix: Optional[float] = None,
               prom_instance: Optional[str] = None) -> Optional[float]:
    d = max(15, int(duration_s))
    m = f"{metric}{{instance=\"{prom_instance}\"}}" if prom_instance else metric
    v = prom_scalar(prom, f"sum(rate({m}[{d}s]))", time_unix=time_unix)
    return round(v, 3) if v is not None else None


def query_hit_rate_pct(prom: str, hits: str, total: str, duration_s: float,
                       time_unix: Optional[float] = None,
                       prom_instance: Optional[str] = None) -> Optional[float]:
    """Token-level cache hit rate: increase(hits[Xs]) / increase(total[Xs]) * 100."""
    d = max(15, int(duration_s))
    h_m = f"{hits}{{instance=\"{prom_instance}\"}}"  if prom_instance else hits
    q_m = f"{total}{{instance=\"{prom_instance}\"}}" if prom_instance else total
    h = prom_scalar(prom, f"increase({h_m}[{d}s])", time_unix=time_unix)
    q = prom_scalar(prom, f"increase({q_m}[{d}s])", time_unix=time_unix)
    if h is None or q is None or q == 0:
        return None
    return round(h / q * 100, 1)


# ── per-task runner ────────────────────────────────────────────────────────────

def run_task(
    instance: dict,
    workdir: Path,
    model: str,
    base_url: str,
    config: str,
    concurrency: int,
    osl_queue: "_queue.Queue | None" = None,
    warmup_done: "threading.Event | None" = None,
) -> tuple[dict, list[TurnRecord]]:
    """
    Run one SWE-bench task via claude CLI with stream-json output.

    OSL per turn comes from the shared osl_queue (polled from SGLang /metrics).
    Multiple assistant events with the same ISL are merged into one TurnRecord.
    """
    instance_id = instance["instance_id"]
    problem     = instance["problem_statement"]

    env = {
        **os.environ,
        "ANTHROPIC_BASE_URL":             base_url,
        "ANTHROPIC_API_KEY":              "dummy",
        "ANTHROPIC_AUTH_TOKEN":           "dummy",
        "ANTHROPIC_DEFAULT_OPUS_MODEL":   model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL":  model,
        # Disable per-request attribution hash (cch=<5-hex>) injected near
        # byte 79 of every system prompt. Without this, the radix prefix cache
        # diverges at byte 79 of every request and we get 0% offload hits.
        # With it set to 0, multi-turn agent traffic hits the cache normally
        # (~99% hit rate on turn 2+).
        "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    }

    t_task_start = time.monotonic()
    t_turn_start = t_task_start
    turn_records: list[TurnRecord] = []
    final_usage:  dict = {}
    status        = "ok"
    stderr_lines: list[str] = []

    do_snap   = (concurrency == 1)
    snap_prev: dict = _snap_turn_metrics(base_url) if do_snap else {}

    pending_isl:       int | None = None
    pending_t_end:     float      = t_task_start
    pending_has_text:  bool       = False
    pending_has_tools: bool       = False

    def flush_pending() -> None:
        nonlocal pending_isl, pending_t_end, pending_has_text, pending_has_tools, snap_prev
        if pending_isl is None:
            return
        try:
            osl = osl_queue.get(timeout=2.0) if osl_queue is not None else 0
        except _queue.Empty:
            osl = 0

        tm: dict = {}
        if do_snap:
            snap_now = _snap_turn_metrics(base_url)
            tm       = _delta_turn_metrics(snap_prev, snap_now)
            snap_prev = snap_now

        if warmup_done is None or warmup_done.is_set():
            turn_records.append(TurnRecord(
                config        = config,
                concurrency   = concurrency,
                instance_id   = instance_id,
                turn_idx      = len(turn_records) + 1,
                isl           = pending_isl,
                osl           = osl,
                turn_e2e_ms   = (pending_t_end - t_turn_start) * 1000,
                ttft_ms       = tm.get("ttft_ms"),
                itl_ms        = tm.get("itl_ms"),
                queue_ms      = tm.get("queue_ms"),
                gpu_hit_pct   = tm.get("gpu_hit_pct"),
                cpu_hit_pct   = tm.get("cpu_hit_pct"),
                response_type = _classify(pending_has_text, pending_has_tools),
            ))
        pending_isl       = None
        pending_has_text  = False
        pending_has_tools = False

    try:
        proc = subprocess.Popen(
            ["claude", "--model", model,
             "--dangerously-skip-permissions",
             "--output-format", "stream-json",
             "--verbose",
             "-p", problem],
            cwd=str(workdir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        def _drain_stderr():
            for line in proc.stderr:
                stderr_lines.append(line.rstrip())

        _stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        _stderr_thread.start()

        for raw in proc.stdout:
            t_now = time.monotonic()
            line  = raw.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type")

            if etype == "assistant":
                msg     = event.get("message", {})
                usage   = msg.get("usage", {})
                content = msg.get("content", [])
                isl     = usage.get("input_tokens", 0) or 0

                has_text  = any(
                    isinstance(b, dict) and b.get("type") == "text" for b in content
                )
                has_tools = any(
                    isinstance(b, dict) and b.get("type") == "tool_use" for b in content
                )

                if isl != pending_isl:
                    flush_pending()
                    pending_isl = isl

                pending_t_end     = t_now
                pending_has_text  |= has_text
                pending_has_tools |= has_tools

            elif etype == "user":
                flush_pending()
                t_turn_start = time.monotonic()
                if do_snap:
                    snap_prev = _snap_turn_metrics(base_url)

            elif etype == "result":
                flush_pending()
                final_usage = event.get("usage", {})
                if event.get("subtype") != "success":
                    status = event.get("subtype", "unknown")

        rc = proc.wait(timeout=10)
        _stderr_thread.join(timeout=5)
        if rc != 0 and status == "ok":
            err_snippet = " | ".join(stderr_lines[-3:]) if stderr_lines else "(no stderr)"
            status = f"exit:{rc} [{err_snippet}]"

    except subprocess.TimeoutExpired:
        proc.kill()
        status = "timeout"
    except Exception as e:
        status = f"error:{e}"

    wall_time_s = round(time.monotonic() - t_task_start, 2)

    if ("exit:" in status or "error:" in status) and not _server_stopping:
        print(f"  [task] FAILED {instance_id}  status={status}", flush=True)
        for ln in stderr_lines[-10:]:
            print(f"    stderr> {ln}", flush=True)

    summary = {
        "instance_id":           instance_id,
        "status":                status,
        "wall_time_s":           wall_time_s,
        "n_turns":               len(turn_records),
        "input_tokens":          final_usage.get("input_tokens"),
        "output_tokens":         final_usage.get("output_tokens"),
        "cache_read_tokens":     final_usage.get("cache_read_input_tokens"),
        "cache_creation_tokens": final_usage.get("cache_creation_input_tokens"),
    }
    return summary, turn_records


# ── workspace setup ───────────────────────────────────────────────────────────

def setup_workspace(instance: dict) -> Path:
    instance_id = instance["instance_id"]
    repo        = instance["repo"]
    base_commit = instance["base_commit"]
    workspace   = WORKSPACE_ROOT / instance_id

    if workspace.exists():
        subprocess.run(["git", "checkout", "-f", base_commit],
                       cwd=workspace, check=True, capture_output=True)
        subprocess.run(["git", "clean", "-fd"],
                       cwd=workspace, check=True, capture_output=True)
    else:
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth=50",
                        f"https://github.com/{repo}.git", str(workspace)], check=True)
        subprocess.run(["git", "fetch", "--depth=50", "origin", base_commit],
                       cwd=workspace, check=True, capture_output=True)
        subprocess.run(["git", "checkout", base_commit],
                       cwd=workspace, check=True, capture_output=True)
    return workspace


# ── percentile helper ─────────────────────────────────────────────────────────

def _pct(data: list[float]) -> dict[str, Optional[float]]:
    if not data:
        return {"p50": None, "p90": None, "p95": None, "p99": None}
    s = sorted(data)
    def _q(p):
        idx = (len(s) - 1) * p
        lo = int(idx); hi = min(lo + 1, len(s) - 1)
        return round(s[lo] + (s[hi] - s[lo]) * (idx - lo), 2)
    return {"p50": _q(0.50), "p90": _q(0.90), "p95": _q(0.95), "p99": _q(0.99)}


# ── concurrency level runner ──────────────────────────────────────────────────

def run_level(
    concurrency: int,
    initial_specs: list[dict],
    extra_queue: "_queue.Queue[dict | None]",
    n_total: int,
    model: str,
    base_url: str,
    prom: str,
    config: str,
    sustained: bool,
    warmup_cpu_hit_pct: float = 0.0,
    gpu_idx: "int | None" = None,
) -> tuple[dict, list[TurnRecord]]:
    mode = "sustained" if sustained else "batch"
    print(f"\n  ── Concurrency {concurrency:>3}  ({n_total} tasks, {mode}) ──")

    snap_before = _snapshot_offload(base_url)

    # OSL background poller — polls SGLang /metrics at 100ms intervals.
    # Uses sglang:generation_tokens_histogram_sum/count
    _osl_queue = _queue.Queue()
    _osl_stop  = threading.Event()

    def _poll_osl():
        s0       = scrape_server(base_url)
        prev_sum = _sum_scraped(s0, "sglang:generation_tokens_histogram_sum")
        prev_cnt = _sum_scraped(s0, "sglang:generation_tokens_histogram_count")
        while not _osl_stop.is_set():
            time.sleep(0.1)
            s         = scrape_server(base_url)
            new_sum   = _sum_scraped(s, "sglang:generation_tokens_histogram_sum")
            new_cnt   = _sum_scraped(s, "sglang:generation_tokens_histogram_count")
            delta_cnt = int(round(new_cnt - prev_cnt))
            if delta_cnt > 0:
                avg_osl = round((new_sum - prev_sum) / delta_cnt)
                for _ in range(delta_cnt):
                    _osl_queue.put(avg_osl)
                prev_sum = new_sum
                prev_cnt = new_cnt

    _osl_thread = threading.Thread(target=_poll_osl, daemon=True)
    _osl_thread.start()

    # GPU power sampler
    _power_samples: list[float] = []
    _power_stop = threading.Event()

    def _poll_power(idx=gpu_idx):
        if not _PYNVML_OK or idx is None:
            return
        try:
            handle = _pynvml.nvmlDeviceGetHandleByIndex(idx)
            while not _power_stop.is_set():
                try:
                    _power_samples.append(_pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
                except Exception:
                    pass
                _power_stop.wait(1.0)
        except Exception:
            pass

    threading.Thread(target=_poll_power, daemon=True).start()

    # Live TTFT poller
    _ttft_stop = threading.Event()
    from urllib.parse import urlparse as _up
    _prom_inst_live = "localhost:%s" % _up(base_url).port

    def _poll_ttft_live():
        while not _ttft_stop.is_set():
            try:
                v = query_percentiles(prom, "sglang:time_to_first_token_seconds_bucket",
                                      30.0, prom_instance=_prom_inst_live)
                with _live_lock:
                    _live["ttft_p50"] = v.get("p50")
                    _live["ttft_p95"] = v.get("p95")
            except Exception:
                pass
            _ttft_stop.wait(10)

    threading.Thread(target=_poll_ttft_live, daemon=True).start()

    # Warmup gate — suppresses TurnRecords until the host (offload) tier is
    # actively warm. SGLang in this build does NOT emit
    # cached_tokens_total{cache_source="host"}, so we cannot compute a true
    # external hit rate the way vLLM does. As a proxy that captures the same
    # intent ("offload tier has been written and has been queried enough to be
    # serving real traffic"), we gate on:
    #   1. evicted_tokens_total has shown 2 consecutive positive deltas
    #      (HBM has overflowed and is actively spilling to host), AND
    #   2. at least `min_post_evict_prompt_toks` prompt tokens have been
    #      processed AFTER the first positive eviction delta (gives the
    #      offload tier a chance to serve hits on subsequent requests).
    # The threshold parameter is reused as a sanity floor on processed tokens
    # post-eviction (10% of the post-evict window's prompt tokens must be
    # cached, falling back via `cache_source="device"` since host is missing).
    warmup_done: threading.Event | None = None
    t_warmup_wall: list[Optional[float]] = [None]
    t_warmup_mono: list[Optional[float]] = [None]
    if warmup_cpu_hit_pct > 0:
        warmup_done  = threading.Event()
        _warmup_stop = threading.Event()
        _warmup_timeout_s = min(300.0, SUSTAINED_MIN_S * 0.5)

        def _poll_cpu_hit_warmup(threshold=warmup_cpu_hit_pct, done=warmup_done,
                                 stop=_warmup_stop, wall_ref=t_warmup_wall,
                                 mono_ref=t_warmup_mono,
                                 timeout_s=_warmup_timeout_s,
                                 t0=time.monotonic()):
            prev_evict: float = 0.0
            prev_total: float = 0.0
            consec_evict_deltas = 0
            evict_started_total: float = -1.0   # prompt_tokens at first positive eviction delta
            min_post_evict_prompt_toks = 5_000  # need this many prompt tokens after eviction starts
            while not stop.is_set():
                try:
                    s = scrape_server(base_url)
                    evict_total = _sum_scraped(s, "sglang:evicted_tokens_total")
                    total_toks  = _sum_scraped(s, "sglang:prompt_tokens_total")
                    d_evict = evict_total - prev_evict
                    prev_evict = evict_total
                    if d_evict > 0:
                        consec_evict_deltas += 1
                        if evict_started_total < 0:
                            evict_started_total = total_toks
                    else:
                        consec_evict_deltas = 0
                        evict_started_total = -1.0
                    post_evict_prompt = (total_toks - evict_started_total) if evict_started_total > 0 else 0.0
                    if (consec_evict_deltas >= 2
                            and post_evict_prompt >= min_post_evict_prompt_toks):
                        wall_ref[0] = time.time()
                        mono_ref[0] = time.monotonic()
                        done.set()
                        print(f"\n  [warmup] Offload tier warm: HBM evicting + "
                              f"{int(post_evict_prompt):,} prompt tokens since first eviction "
                              f"(target ≥ {threshold}% host hit rate proxy) "
                              f"— metric collection starting", flush=True)
                        return
                    prev_total = total_toks
                except Exception:
                    pass
                elapsed = time.monotonic() - t0
                if elapsed >= timeout_s and not done.is_set():
                    wall_ref[0] = time.time()
                    mono_ref[0] = time.monotonic()
                    done.set()
                    print(f"\n  [warmup] Timeout after {elapsed:.0f}s — "
                          f"offload tier did not warm (HBM not overflowing at this concurrency). "
                          f"Starting metric collection now.", flush=True)
                    return
                time.sleep(2)

        threading.Thread(target=_poll_cpu_hit_warmup, daemon=True).start()
    else:
        _warmup_stop = threading.Event()

    t_start      = time.monotonic()
    task_results: list[dict]       = []
    all_turns:    list[TurnRecord] = []
    lock          = threading.Lock()
    n_done = n_ok = n_fail = 0
    with _live_lock:
        _live.update({"c": concurrency, "cfg": config, "elapsed": 0.0,
                      "done": 0, "ok": 0, "fail": 0, "active": 0})

    spec_iter  = iter(initial_specs)
    queue_done = False

    effective_sustained = sustained or (warmup_cpu_hit_pct > 0)

    def _next_spec() -> "dict | None":
        nonlocal queue_done
        s = next(spec_iter, None)
        if s is not None:
            return s
        if queue_done:
            return None
        if effective_sustained:
            try:
                s = extra_queue.get(timeout=2.0)
            except _queue.Empty:
                return None
        else:
            try:
                s = extra_queue.get_nowait()
            except _queue.Empty:
                return None
        if s is None:
            queue_done = True
            return None
        return s

    ex     = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
    active: dict = {}

    def _submit(spec: dict):
        f = ex.submit(
            run_task,
            spec["instance"], spec["workdir"], model, base_url, config, concurrency,
            _osl_queue, warmup_done,
        )
        active[f] = spec

    def _fill():
        while len(active) < concurrency:
            spec = _next_spec()
            if spec is None:
                break
            _submit(spec)

    _fill()

    def _measure_elapsed() -> float:
        if warmup_cpu_hit_pct > 0 and t_warmup_mono[0] is not None:
            return time.monotonic() - t_warmup_mono[0]
        return time.monotonic() - t_start

    try:
        stop = False
        while not stop:
            if not active:
                if not effective_sustained:
                    break
                if _measure_elapsed() >= SUSTAINED_MIN_S or queue_done:
                    stop = True
                    break
                _fill()
                if not active:
                    if queue_done:
                        stop = True
                        break
                    time.sleep(0.5)
                continue

            done, _ = concurrent.futures.wait(
                list(active.keys()), timeout=1.0,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                summary, turns = future.result()
                del active[future]
                with lock:
                    task_results.append(summary)
                    all_turns.extend(turns)

                elapsed = time.monotonic() - t_start
                n_done += 1
                if summary["status"] == "ok":
                    n_ok += 1
                else:
                    n_fail += 1
                with _live_lock:
                    _live.update({"elapsed": elapsed, "done": n_done, "ok": n_ok,
                                  "fail": n_fail, "active": len(active)})
                _redraw_live()

                if effective_sustained and _measure_elapsed() >= SUSTAINED_MIN_S:
                    print(f"\n    Stop: {_measure_elapsed():.0f}s since warmup/start. "
                          f"Cancelling queued tasks.")
                    for f in list(active.keys()):
                        f.cancel()
                    active.clear()
                    stop = True
                    break

            if not stop:
                _fill()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    t_end      = time.monotonic()
    t_end_unix = time.time()
    duration_s = t_end - t_start

    _osl_stop.set()
    _ttft_stop.set()
    _warmup_stop.set()
    _power_stop.set()
    _osl_thread.join(timeout=5)
    snap_after = _snapshot_offload(base_url)
    kv_offload = _delta_offload(snap_before, snap_after)

    avg_power_w = round(sum(_power_samples) / len(_power_samples), 1) if _power_samples else None

    print(f"\n    Waiting {PROM_SCRAPE_BUFFER_S}s for Prometheus ...", flush=True)
    time.sleep(PROM_SCRAPE_BUFFER_S)

    if t_warmup_wall[0] is not None:
        prom_duration_s = t_end_unix - t_warmup_wall[0]
        prom_time_unix  = t_end_unix
        print(f"    [warmup] Prometheus queries scoped to post-warmup window "
              f"({prom_duration_s:.0f}s)", flush=True)
    else:
        prom_duration_s = duration_s
        prom_time_unix  = None

    from urllib.parse import urlparse as _urlparse
    prom_inst = f"localhost:{_urlparse(base_url).port}"

    # ── Prometheus latency histograms (SGLang metric names) ──────────────────
    ttft_ms    = query_percentiles(prom, "sglang:time_to_first_token_seconds_bucket",
                                   prom_duration_s, time_unix=prom_time_unix, prom_instance=prom_inst)
    e2e_ms     = query_percentiles(prom, "sglang:e2e_request_latency_seconds_bucket",
                                   prom_duration_s, time_unix=prom_time_unix, prom_instance=prom_inst)
    itl_ms     = query_percentiles(prom, "sglang:inter_token_latency_seconds_bucket",
                                   prom_duration_s, time_unix=prom_time_unix, prom_instance=prom_inst)
    queue_ms   = query_percentiles(prom, "sglang:queue_time_seconds_bucket",
                                   prom_duration_s, time_unix=prom_time_unix, prom_instance=prom_inst)
    prompt_tok = query_percentiles(prom, "sglang:prompt_tokens_histogram_bucket",
                                   prom_duration_s, multiplier=1.0,
                                   time_unix=prom_time_unix, prom_instance=prom_inst)
    gen_tok    = query_percentiles(prom, "sglang:generation_tokens_histogram_bucket",
                                   prom_duration_s, multiplier=1.0,
                                   time_unix=prom_time_unix, prom_instance=prom_inst)

    req_rps = query_rate(prom, "sglang:e2e_request_latency_seconds_count",
                         prom_duration_s, time_unix=prom_time_unix, prom_instance=prom_inst)
    out_tps = query_rate(prom, "sglang:generation_tokens_total",
                         prom_duration_s, time_unix=prom_time_unix, prom_instance=prom_inst)

    # Cache hit rates: token-level (cached tokens / total prompt tokens)
    # sglang:cached_tokens_total{cache_source="device"} / sglang:prompt_tokens_total
    gpu_hit = query_hit_rate_pct(
        prom,
        f'sglang:cached_tokens_total{{cache_source="device"}}',
        "sglang:prompt_tokens_total",
        prom_duration_s, time_unix=prom_time_unix, prom_instance=prom_inst,
    )
    cpu_hit = query_hit_rate_pct(
        prom,
        f'sglang:cached_tokens_total{{cache_source="host"}}',
        "sglang:prompt_tokens_total",
        prom_duration_s, time_unix=prom_time_unix, prom_instance=prom_inst,
    )

    kv_used = prom_scalar(
        prom,
        f'avg_over_time(sglang:token_usage{{instance="{prom_inst}"}}[{max(15,int(prom_duration_s))}s])',
        time_unix=prom_time_unix,
    )

    # ── Client-side per-turn percentiles ─────────────────────────────────────
    ok_turns = all_turns
    ok_tasks = [r for r in task_results if r["status"] == "ok"]

    turns_by_task: dict[str, list[TurnRecord]] = {}
    for t in ok_turns:
        turns_by_task.setdefault(t.instance_id, []).append(t)

    def _turn_dict(t: TurnRecord) -> dict:
        ttft = t.ttft_ms
        return {
            "turn_idx":    t.turn_idx,
            "isl":         t.isl,
            "osl":         t.osl,
            "ttft_ms":     ttft,
            "itl_ms":      t.itl_ms,
            "decode_ms":   round(t.turn_e2e_ms - ttft, 1) if ttft is not None else None,
            "e2e_ms":      round(t.turn_e2e_ms, 1),
            "queue_ms":    t.queue_ms,
            "gpu_hit_pct": t.gpu_hit_pct,
            "cpu_hit_pct": t.cpu_hit_pct,
            "response_type": t.response_type,
        }

    for r in task_results:
        task_turns = sorted(turns_by_task.get(r["instance_id"], []), key=lambda x: x.turn_idx)
        r["turns"] = [_turn_dict(t) for t in task_turns]

    task_tps = [
        r["output_tokens"] / r["wall_time_s"]
        for r in ok_tasks
        if r.get("output_tokens") and r.get("wall_time_s")
    ]
    turn_tps = [
        t.osl / (t.turn_e2e_ms / 1000)
        for t in ok_turns
        if t.osl and t.turn_e2e_ms > 0
    ]

    level = {
        "concurrency": concurrency,
        "config":      config,
        "n_tasks":     n_total,
        "n_ok":        len(ok_tasks),
        "wall_time_s": round(duration_s, 1),
        "n_turns_total": len(ok_turns),

        "throughput": {
            "tasks_per_sec":         round(len(ok_tasks) / duration_s, 4) if ok_tasks else 0,
            "requests_per_sec":      req_rps,
            "output_tokens_per_sec": out_tps,
            "task_tps": _pct(task_tps),
            "turn_tps": _pct(turn_tps),
        },

        "power": {
            "gpu_idx":         gpu_idx,
            "avg_power_w":     avg_power_w,
            "n_samples":       len(_power_samples),
            "output_tps_per_kw": round(out_tps / (avg_power_w / 1000), 1)
                                  if avg_power_w and out_tps else None,
        },

        # Prometheus latency (per SGLang request = per claude turn)
        "ttft_ms":    ttft_ms,
        "e2e_ms":     e2e_ms,
        "itl_ms":     itl_ms,
        "queue_ms":   queue_ms,
        "prefill_ms": {"p50": None, "p90": None, "p95": None, "p99": None},  # not in sglang
        "prompt_tok_len": prompt_tok,
        "gen_tok_len":    gen_tok,

        # Client-side per-turn distributions
        "turn_e2e_ms":    _pct([t.turn_e2e_ms    for t in ok_turns]),
        "turn_isl":       _pct([float(t.isl)      for t in ok_turns]),
        "turn_osl":       _pct([float(t.osl)      for t in ok_turns]),
        "turn_ttft_ms":   _pct([t.ttft_ms  for t in ok_turns if t.ttft_ms  is not None]),
        "turn_itl_ms":    _pct([t.itl_ms   for t in ok_turns if t.itl_ms   is not None]),
        "turn_queue_ms":  _pct([t.queue_ms for t in ok_turns if t.queue_ms is not None]),
        "turn_gpu_hit_pct": _pct([t.gpu_hit_pct for t in ok_turns if t.gpu_hit_pct is not None]),
        "turn_cpu_hit_pct": _pct([t.cpu_hit_pct for t in ok_turns if t.cpu_hit_pct is not None]),

        "response_types": {
            rtype: sum(1 for t in ok_turns if t.response_type == rtype)
            for rtype in ("text", "tool_call", "mixed")
        },

        "cache": {
            "gpu_hit_rate_pct": gpu_hit,
            "cpu_hit_rate_pct": cpu_hit,
            "kv_usage_pct":     round(kv_used * 100, 1) if kv_used is not None else None,
        },

        # KV offload token deltas (evicted: GPU→host, load_back: host→GPU)
        "kv_offload": kv_offload,

        "tasks": task_results,
    }

    _print_level_summary(level)
    return level, all_turns


def _f(v, u=""):
    return f"{v}{u}" if v is not None else "-"

def _fmt_pct(d: dict, suffix: str = "") -> str:
    return (f"p50={_f(d['p50'], suffix)}  p90={_f(d['p90'], suffix)}"
            f"  p95={_f(d['p95'], suffix)}  p99={_f(d['p99'], suffix)}")


def _print_level_summary(level: dict) -> None:
    c  = level["concurrency"]
    tp = level["throughput"]
    ca = level.get("cache", {})
    pw = level.get("power", {})
    W  = 65

    print(f"\n    ┌─ C={c} ({level['config']}) {'─' * (W - 15)}")
    print(f"    │  tasks    : {level['n_ok']}/{level['n_tasks']} ok"
          f"  wall={level['wall_time_s']:.0f}s"
          f"  turns={level['n_turns_total']}")
    print(f"    │  throughput: {tp['tasks_per_sec']:.3f} tasks/s"
          f"  {_f(tp['requests_per_sec'])} req/s"
          f"  {_f(tp['output_tokens_per_sec'])} out-tok/s  (level avg)")
    if pw.get("avg_power_w") is not None:
        print(f"    │  power    : {pw['avg_power_w']:.0f}W (GPU {pw['gpu_idx']})"
              f"  efficiency={_f(pw.get('output_tps_per_kw'))} tok/s/kW")
    print(f"    │  task tok/s : {_fmt_pct(tp['task_tps'])}  (per-task)")
    print(f"    │  turn tok/s : {_fmt_pct(tp['turn_tps'])}  (per-turn)")
    print(f"    │  resp types: {level['response_types']}")
    print(f"    │")
    print(f"    │  ── SGLang latency (Prometheus, per turn) {'─' * 19}")
    print(f"    │  TTFT    (ms): {_fmt_pct(level['ttft_ms'],  'ms')}")
    print(f"    │  Queue   (ms): {_fmt_pct(level['queue_ms'], 'ms')}")
    print(f"    │  ITL     (ms): {_fmt_pct(level['itl_ms'],   'ms')}")
    print(f"    │  E2E     (ms): {_fmt_pct(level['e2e_ms'],   'ms')}")
    print(f"    │")
    print(f"    │  ── Per-turn client-side (stream-json timestamps) {'─'*11}")
    print(f"    │  Turn E2E(ms): {_fmt_pct(level['turn_e2e_ms'], 'ms')}  ← model+tool exec")
    print(f"    │")
    print(f"    │  ── KV cache (token-level hit rates) ────────────")
    print(f"    │  GPU hit %  : {_f(ca.get('gpu_hit_rate_pct'), '%')}")
    print(f"    │  Host hit % : {_f(ca.get('cpu_hit_rate_pct'), '%')}")
    print(f"    │  KV used %  : {_f(ca.get('kv_usage_pct'), '%')}")
    offload = level.get("kv_offload", {})
    if any(v > 0 for v in offload.values()):
        print(f"    │")
        print(f"    │  ── KV offload tokens (this level) ─────────────")
        for k, v in offload.items():
            if v > 0:
                print(f"    │  {k:<25}: {v:.0f} tokens")
    print(f"    └{'─' * W}")


# ── CSV writers ───────────────────────────────────────────────────────────────

def write_per_turn_csv(all_turns: list[TurnRecord], path: Path) -> None:
    if not all_turns:
        return
    fieldnames = list(asdict(all_turns[0]).keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(asdict(r) for r in all_turns)


def write_summary_csv(levels: list[dict], path: Path) -> None:
    rows = []
    for lv in levels:
        tp = lv["throughput"]
        ca = lv.get("cache", {})
        row = {
            "config":            lv["config"],
            "concurrency":       lv["concurrency"],
            "n_ok":              lv["n_ok"],
            "wall_time_s":       lv["wall_time_s"],
            "n_turns_total":     lv["n_turns_total"],
            "tasks_per_sec":     tp["tasks_per_sec"],
            "requests_per_sec":  tp["requests_per_sec"],
            "output_tok_per_sec": tp["output_tokens_per_sec"],
            # Prometheus TTFT
            "ttft_p50": lv["ttft_ms"]["p50"],
            "ttft_p90": lv["ttft_ms"]["p90"],
            "ttft_p95": lv["ttft_ms"]["p95"],
            "ttft_p99": lv["ttft_ms"]["p99"],
            # Prometheus E2E
            "e2e_p50": lv["e2e_ms"]["p50"],
            "e2e_p90": lv["e2e_ms"]["p90"],
            "e2e_p95": lv["e2e_ms"]["p95"],
            "e2e_p99": lv["e2e_ms"]["p99"],
            # Prometheus ITL
            "itl_p50": lv["itl_ms"]["p50"],
            "itl_p90": lv["itl_ms"]["p90"],
            "itl_p95": lv["itl_ms"]["p95"],
            "itl_p99": lv["itl_ms"]["p99"],
            # Prometheus queue
            "queue_p50": lv["queue_ms"]["p50"],
            "queue_p99": lv["queue_ms"]["p99"],
            # Throughput percentiles
            "task_tps_p50": tp.get("task_tps", {}).get("p50"),
            "task_tps_p90": tp.get("task_tps", {}).get("p90"),
            "task_tps_p95": tp.get("task_tps", {}).get("p95"),
            "task_tps_p99": tp.get("task_tps", {}).get("p99"),
            "turn_tps_p50": tp.get("turn_tps", {}).get("p50"),
            "turn_tps_p90": tp.get("turn_tps", {}).get("p90"),
            "turn_tps_p95": tp.get("turn_tps", {}).get("p95"),
            "turn_tps_p99": tp.get("turn_tps", {}).get("p99"),
            # Per-turn ISL / OSL
            "turn_isl_p50": lv["turn_isl"]["p50"],
            "turn_isl_p90": lv["turn_isl"]["p90"],
            "turn_isl_p95": lv["turn_isl"]["p95"],
            "turn_isl_p99": lv["turn_isl"]["p99"],
            "turn_osl_p50": lv["turn_osl"]["p50"],
            "turn_osl_p90": lv["turn_osl"]["p90"],
            "turn_osl_p95": lv["turn_osl"]["p95"],
            "turn_osl_p99": lv["turn_osl"]["p99"],
            # Per-turn latency (exact at c=1)
            "turn_ttft_p50":  lv["turn_ttft_ms"]["p50"],
            "turn_ttft_p99":  lv["turn_ttft_ms"]["p99"],
            "turn_itl_p50":   lv["turn_itl_ms"]["p50"],
            "turn_itl_p99":   lv["turn_itl_ms"]["p99"],
            "turn_queue_p50": lv["turn_queue_ms"]["p50"],
            "turn_queue_p99": lv["turn_queue_ms"]["p99"],
            "turn_gpu_hit_p50": lv["turn_gpu_hit_pct"]["p50"],
            "turn_cpu_hit_p50": lv["turn_cpu_hit_pct"]["p50"],
            # Client-side turn E2E
            "turn_e2e_p50": lv["turn_e2e_ms"]["p50"],
            "turn_e2e_p90": lv["turn_e2e_ms"]["p90"],
            "turn_e2e_p95": lv["turn_e2e_ms"]["p95"],
            "turn_e2e_p99": lv["turn_e2e_ms"]["p99"],
            # Cache
            "gpu_hit_pct": ca.get("gpu_hit_rate_pct"),
            "cpu_hit_pct": ca.get("cpu_hit_rate_pct"),
            "kv_used_pct": ca.get("kv_usage_pct"),
            # Power / efficiency
            "avg_power_w":       lv.get("power", {}).get("avg_power_w"),
            "output_tps_per_kw": lv.get("power", {}).get("output_tps_per_kw"),
            # KV offload token counts
            **lv.get("kv_offload", {}),
        }
        rows.append(row)

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


# ── cleanup on exit / Ctrl-C ─────────────────────────────────────────────────

def _cleanup() -> None:
    if _active_port is not None:
        print(f"\n  [cleanup] Killing SGLang on port {_active_port} ...", flush=True)
        subprocess.run(["pkill", "-9", "-f", f"sglang.launch_server.*--port {_active_port}"],
                       capture_output=True)
    if _active_setup and "mtier" in _active_setup:
        print("  [cleanup] Resetting MTier ...", flush=True)
        subprocess.run(["sh", "-c", "echo yes | mtier_service reset 2>/dev/null || true"],
                       capture_output=True)
    print("  [cleanup] Done.", flush=True)

def _signal_handler(sig, frame):
    _cleanup()
    sys.exit(0)

atexit.register(_cleanup)
signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                    help="Concurrency levels to sweep")
    ap.add_argument("--tasks-per-level", type=int, default=None,
                    help="Tasks per level (default: C in batch, 5×C in sustained)")
    ap.add_argument("--run-all", action="store_true",
                    help="Sustain N concurrent users until every task is done (no time cap). "
                         "Implies --sustained.")
    ap.add_argument("--sustained", action="store_true",
                    help="Keep N tasks running; replace on completion; stop after cap")
    ap.add_argument("--start",   type=int, default=0,    help="SWE-bench dataset start index")
    ap.add_argument("--end",     type=int, default=None, help="SWE-bench dataset end index")
    ap.add_argument("--difficulty", nargs="+", default=None,
                    help="Filter by SWE-bench difficulty. "
                         "Shorthand: easy=<15min medium=15min-1h hard=1-4h vhard=>4h")
    ap.add_argument("--prom",    default=DEFAULT_PROM)
    ap.add_argument("--port",    default=None)
    ap.add_argument("--model",   default=None)
    ap.add_argument("--setup",   nargs="+", default=["hybrid-cpu", "hybrid-mtier"],
                    help=f"SGLang setup(s) to benchmark. Valid: {sorted(VALID_SETUPS)}. "
                         "Multiple values run sequentially with a server restart between them.")
    ap.add_argument("--sustained-mins", type=float, default=None,
                    help="Wall-clock cap per level in sustained mode (default: 10 min)")
    ap.add_argument("--tp", type=int, default=None,
                    help="Tensor parallel size (overrides TENSOR_PARALLEL_SIZE in .env)")
    ap.add_argument("--gpu-util", type=float, default=None,
                    help="GPU memory utilization 0.0-1.0 (overrides GPU_MEMORY_UTILIZATION in .env)")
    ap.add_argument("--gpu-idx", type=str, default=None,
                    help="CUDA_VISIBLE_DEVICES for SGLang (default: SGLANG_GPU_IDX env, else '0')")
    ap.add_argument("--max-running-reqs", type=int, default=None,
                    help="SGLang --max-running-requests; defaults to current concurrency level")
    ap.add_argument("--warmup-cpu-hit-pct", type=float, default=1.0,
                    help="Don't record metrics until host cache token hit rate exceeds "
                         "this %% (default: 1). 0 = disabled.")
    ap.add_argument("--workspace-root", type=str, default=None,
                    help="Base dir for cloned task repos (default: /tmp/swe_workspaces_p<port>)")
    ap.add_argument("--no-clone", action="store_true", help="Skip repo setup, run in cwd")
    args = ap.parse_args()

    for s in args.setup:
        if s not in VALID_SETUPS:
            ap.error(f"Invalid setup {s!r}. Valid: {sorted(VALID_SETUPS)}")

    global SUSTAINED_MIN_S
    if args.run_all:
        args.sustained = True
        SUSTAINED_MIN_S = float("inf")
    elif args.sustained_mins is not None:
        SUSTAINED_MIN_S = args.sustained_mins * 60

    load_env(ENV_FILE)
    model = args.model or os.environ.get("MODEL", DEFAULT_MODEL)

    # All sglang modes run on the same GPU (default 0)
    global SGLANG_GPU_IDX
    if args.gpu_idx:
        SGLANG_GPU_IDX = args.gpu_idx

    _setup_ports = {
        "hbm-only":     8000,
        "hybrid-cpu":   8001,
        "hybrid-mtier": 8002,
    }
    _single_setup = args.setup[0] if len(args.setup) == 1 else None
    _auto_port    = _setup_ports.get(_single_setup, 8000)
    port          = int(args.port or os.environ.get("PORT", str(_auto_port)))
    base_url      = f"http://localhost:{port}"

    # Physical GPU index for power monitoring
    _gpu_idx: "int | None" = None
    if _PYNVML_OK:
        try:
            _gpu_idx = int(SGLANG_GPU_IDX.split(",")[0])
        except (ValueError, IndexError):
            _gpu_idx = 0

    global _active_port, _active_setup
    _active_port  = port
    _active_setup = args.setup[0] if len(args.setup) == 1 else None

    global WORKSPACE_ROOT
    WORKSPACE_ROOT = Path(args.workspace_root or f"/tmp/swe_workspaces_p{port}")

    from datasets import load_dataset
    print("Loading SWE-bench Verified ...", flush=True)
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    _DIFF_ALIASES = {
        "easy":   "<15 min fix",
        "medium": "15 min - 1 hour",
        "hard":   "1-4 hours",
        "vhard":  ">4 hours",
    }
    _DIFF_RANK = {
        ">4 hours":        0,
        "1-4 hours":       1,
        "15 min - 1 hour": 2,
        "<15 min fix":     3,
    }

    if args.difficulty:
        wanted = {_DIFF_ALIASES.get(d, d) for d in args.difficulty}
        ds = ds.filter(lambda row: row["difficulty"] in wanted)
        print(f"  Difficulty filter: {wanted}  → {len(ds)} tasks", flush=True)

    rows = sorted(ds, key=lambda r: (_DIFF_RANK.get(r["difficulty"], 9),
                                     -len(r["problem_statement"])))

    total = len(rows)
    end   = args.end if args.end is not None else total

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_levels: list[dict]       = []
    all_turns:  list[TurnRecord] = []

    for setup_idx, setup in enumerate(args.setup):
        if "mtier" in setup:
            print(f"\n  [mtier] Resetting MTier memory before {setup} ...", flush=True)
            subprocess.run(["sh", "-c", "echo yes | mtier_service reset 2>/dev/null || true"],
                           capture_output=True)
            time.sleep(2)

        run_name = f"bench_{setup}_p{port}_{ts}"
        out_dir  = RESULTS_DIR / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'═'*70}")
        print(f"  Backend    : SGLang")
        print(f"  Setup      : {setup}  ({setup_idx + 1}/{len(args.setup)})")
        print(f"  Model      : {model}")
        print(f"  GPU(s)     : {SGLANG_GPU_IDX}")
        print(f"  Mode       : {'sustained' if args.sustained else 'batch'}")
        print(f"  Difficulty : {args.difficulty or 'all'}")
        print(f"  Tasks pool : {total}")
        print(f"  Levels     : {args.concurrency}")
        print(f"  Results    : {out_dir}/")
        print(f"{'═'*70}")

        setup_levels: list[dict]       = []
        setup_turns:  list[TurnRecord] = []

        for concurrency in args.concurrency:
            all_indices = list(range(args.start, min(end, total)))
            if args.run_all:
                indices  = all_indices
                n_needed = len(indices)
            else:
                n_needed = args.tasks_per_level or ((5 * concurrency) if args.sustained else concurrency)
                indices  = all_indices[:n_needed]
            if len(indices) < n_needed and not args.sustained:
                print(f"WARNING: need {n_needed} tasks but only {len(indices)} available.")
            instances   = [rows[i] for i in indices]
            first_batch = instances[:concurrency]
            rest_batch  = instances[concurrency:]

            stop_clone: threading.Event = threading.Event()
            with _live_lock:
                _live.update({"c": concurrency, "cfg": setup, "cloned": 0,
                              "elapsed": 0.0, "done": 0, "ok": 0, "fail": 0, "active": 0})

            max_reqs    = args.max_running_reqs if args.max_running_reqs is not None else concurrency
            server_proc = start_server(setup, port, tp=args.tp, gpu_util=args.gpu_util,
                                       gpus=SGLANG_GPU_IDX,
                                       max_running_reqs=max_reqs)

            extra_queue: _queue.Queue = _queue.Queue(maxsize=concurrency * 5)

            if args.no_clone:
                initial_specs = [{"instance": inst, "workdir": Path.cwd()} for inst in first_batch]
                for inst in rest_batch:
                    extra_queue.put({"instance": inst, "workdir": Path.cwd()})
                if not args.sustained and not args.warmup_cpu_hit_pct:
                    extra_queue.put(None)
            else:
                print(f"\nSetting up first {len(first_batch)} workspaces ...")
                initial_specs: list[dict] = []
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(CLONE_WORKERS, max(1, len(first_batch)))
                ) as cex:
                    fmap = {cex.submit(setup_workspace, inst): inst for inst in first_batch}
                    for f in concurrent.futures.as_completed(fmap):
                        inst = fmap[f]
                        try:
                            wd = f.result()
                        except Exception as e:
                            print(f"\r  ✗ {inst['instance_id']}: {e}", flush=True)
                            wd = Path.cwd()
                        with _live_lock:
                            _live["cloned"] = _live.get("cloned", 0) + 1
                        _redraw_live()
                        initial_specs.append({"instance": inst, "workdir": wd})
                print(flush=True)

                use_sustained_clone = args.sustained or (args.warmup_cpu_hit_pct > 0)

                if use_sustained_clone:
                    print(f"  [bg] Sustained clone loop over {len(instances)} instances ...",
                          flush=True)

                    def _bg_clone_sustained(all_inst=instances, q=extra_queue, stop=stop_clone,
                                            _run_all=args.run_all):
                        idx = len(first_batch)
                        with concurrent.futures.ThreadPoolExecutor(max_workers=CLONE_WORKERS) as cex:
                            while not stop.is_set():
                                if _run_all and idx >= len(all_inst):
                                    break
                                inst = all_inst[idx % len(all_inst)]
                                idx += 1
                                if stop.is_set():
                                    break
                                try:
                                    wd = cex.submit(setup_workspace, inst).result()
                                except Exception:
                                    wd = Path.cwd()
                                with _live_lock:
                                    _live["cloned"] = _live.get("cloned", 0) + 1
                                _redraw_live()
                                if not stop.is_set():
                                    q.put({"instance": inst, "workdir": wd})
                        q.put(None)

                    threading.Thread(target=_bg_clone_sustained, daemon=True).start()
                elif rest_batch:
                    print(f"  Cloning {len(rest_batch)} remaining workspaces in background ...",
                          flush=True)

                    def _bg_clone(batch=rest_batch, q=extra_queue):
                        with concurrent.futures.ThreadPoolExecutor(max_workers=CLONE_WORKERS) as cex:
                            fmap = {cex.submit(setup_workspace, inst): inst for inst in batch}
                            for f in concurrent.futures.as_completed(fmap):
                                inst = fmap[f]
                                try:
                                    wd = f.result()
                                except Exception:
                                    wd = Path.cwd()
                                with _live_lock:
                                    _live["cloned"] = _live.get("cloned", 0) + 1
                                _redraw_live()
                                q.put({"instance": inst, "workdir": wd})
                        q.put(None)

                    threading.Thread(target=_bg_clone, daemon=True).start()
                else:
                    extra_queue.put(None)

            try:
                level, turns = run_level(
                    concurrency, initial_specs, extra_queue, n_needed,
                    model, base_url, args.prom, setup, args.sustained,
                    warmup_cpu_hit_pct=args.warmup_cpu_hit_pct,
                    gpu_idx=_gpu_idx,
                )
            finally:
                stop_clone.set()
                stop_server(server_proc, setup, port)

            setup_levels.append(level)
            setup_turns.extend(turns)
            all_levels.append(level)
            all_turns.extend(turns)

            (out_dir / f"c{concurrency:03d}.json").write_text(json.dumps(level, indent=2))

        write_summary_csv(setup_levels, out_dir / "summary.csv")
        write_per_turn_csv(setup_turns, out_dir / "per_turn.csv")
        (out_dir / "summary.json").write_text(json.dumps(setup_levels, indent=2))
        print(f"\n  → {out_dir}/  ({len(setup_turns)} turn rows)")

    if len(args.setup) > 1:
        combined_dir = RESULTS_DIR / f"bench_combined_{ts}"
        combined_dir.mkdir(parents=True, exist_ok=True)
        write_summary_csv(all_levels, combined_dir / "summary.csv")
        write_per_turn_csv(all_turns, combined_dir / "per_turn.csv")
        (combined_dir / "summary.json").write_text(json.dumps(all_levels, indent=2))
        print(f"\n{'═'*70}")
        print(f"  Combined results → {combined_dir}/")
        print(f"    summary.csv   — all setups × levels ({len(all_levels)} rows)")
        print(f"    per_turn.csv  — all turns ({len(all_turns)} rows)")
        print(f"{'═'*70}")


if __name__ == "__main__":
    main()
