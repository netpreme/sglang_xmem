#!/usr/bin/env python3
"""
Per-request metrics watcher for SGLang.

Polls /metrics directly at 0.5s intervals.  When a new request completes,
prints one row with exact per-turn deltas.

Columns:
  ISL     — avg input tokens this turn (prompt_tokens_histogram delta)
  OSL     — avg output tokens this turn (generation_tokens_histogram delta)
  TTFT    — time to first token (ms), exact per-turn delta
  ITL     — avg inter-token latency (ms), exact per-turn delta
  E2E     — end-to-end latency (ms), exact per-turn delta
  Q       — queue wait time (ms), exact per-turn delta
  GPU$    — GPU (HBM) token cache hit % this turn
  Host$   — host (CPU/MTier) token cache hit % this turn
  Evict   — tokens evicted from GPU to host this turn
  LoadBk  — tokens loaded back from host to GPU this turn

On Ctrl+C saves a chart to ./session_metrics_sglang.png.

Usage:
    python3 watch_requests.py [--url http://localhost:8000] [--interval 0.5]
"""

import argparse
import signal
import sys
import time
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

_METRICS_LINE_RE  = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(?:\{([^}]*)\})?'
    r'\s+([+-]?(?:[0-9]*\.)?[0-9]+(?:[eE][+-]?[0-9]+)?|NaN|[+-]?Inf)'
)
_METRICS_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')


def scrape(url: str) -> dict:
    result: dict[str, list[tuple[dict, float]]] = {}
    try:
        r = requests.get(f"{url}/metrics", timeout=3)
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


def _sum(s: dict, name: str) -> Optional[float]:
    entries = s.get(name)
    if not entries:
        return None
    return sum(v for _, v in entries)


def _by_label(s: dict, name: str, label_key: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for labels, value in s.get(name, []):
        k = labels.get(label_key, "unknown")
        result[k] = result.get(k, 0.0) + value
    return result


def _delta_ms(new_s, new_c, old_s, old_c) -> Optional[float]:
    if None in (new_s, new_c, old_s, old_c):
        return None
    d_c = new_c - old_c
    if d_c <= 0:
        return None
    return (new_s - old_s) / d_c * 1000


@dataclass
class Turn:
    index: int
    ts: str
    isl: Optional[float]
    osl: Optional[float]
    ttft_ms: Optional[float]
    itl_ms: Optional[float]
    e2e_ms: Optional[float]
    queue_ms: Optional[float]
    gpu_hit_pct: Optional[float]
    host_hit_pct: Optional[float]
    evict_toks: Optional[float]
    loadbk_toks: Optional[float]


def fmt_f(v, decimals=1, suffix="") -> str:
    return f"{v:.{decimals}f}{suffix}" if v is not None else "-"

def fmt_tok(v) -> str:
    if v is None or v == 0:
        return "0"
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    if v >= 1e3:
        return f"{v/1e3:.0f}K"
    return f"{v:.0f}"


def save_chart(turns: list[Turn], path: str) -> None:
    if not HAS_MPL or not turns:
        if not turns:
            print("No data to chart")
        return

    idx      = [t.index for t in turns]
    isl      = [t.isl       or 0 for t in turns]
    osl      = [t.osl       or 0 for t in turns]
    ttft     = [t.ttft_ms   or 0 for t in turns]
    e2e      = [t.e2e_ms    or 0 for t in turns]
    gpu_hit  = [t.gpu_hit_pct  or 0 for t in turns]
    host_hit = [t.host_hit_pct or 0 for t in turns]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle("Session Metrics — SGLang", fontsize=14)

    def bar(ax, y, label, color, unit):
        ax.bar(idx, y, color=color, alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel("Request #")
        ax.set_ylabel(unit)
        ax.set_xticks(idx)
        for i, v in zip(idx, y):
            if v:
                ax.text(i, v * 1.02, f"{v:.0f}", ha="center", fontsize=8)

    bar(axes[0][0], isl, "Input Sequence Length",  "#4e79a7", "tokens")
    bar(axes[0][1], osl, "Output Sequence Length", "#f28e2b", "tokens")

    ax = axes[0][2]
    ax.plot(idx, ttft, marker="o", color="#e15759", linewidth=2, label="TTFT")
    ax.plot(idx, e2e,  marker="s", color="#b07aa1", linewidth=2, label="E2E")
    ax.set_title("Latency per Turn (exact per-turn delta)")
    ax.set_xlabel("Request #"); ax.set_ylabel("ms"); ax.set_xticks(idx); ax.legend(fontsize=8)

    ax = axes[1][0]
    ax.plot(idx, gpu_hit,  marker="o", color="#4e79a7", linewidth=2, label="GPU%")
    ax.plot(idx, host_hit, marker="s", color="#f28e2b", linewidth=2, label="Host%")
    ax.set_ylim(0, 105)
    ax.set_title("Token Cache Hit Rate (GPU vs Host)")
    ax.set_xlabel("Request #"); ax.set_ylabel("%"); ax.set_xticks(idx); ax.legend(fontsize=8)

    evict  = [t.evict_toks  or 0 for t in turns]
    loadbk = [t.loadbk_toks or 0 for t in turns]
    bar(axes[1][1], evict,  "Tokens Evicted GPU→Host",  "#59a14f", "tokens")
    bar(axes[1][2], loadbk, "Tokens Loaded Back Host→GPU", "#76b7b2", "tokens")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nChart saved → {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url",      default="http://localhost:8000",
                    help="SGLang server URL (default: http://localhost:8000)")
    ap.add_argument("--interval", type=float, default=0.5,
                    help="Poll interval in seconds (default: 0.5)")
    ap.add_argument("--chart",    default="session_metrics_sglang.png")
    args = ap.parse_args()

    url    = args.url
    turns: list[Turn] = []
    turn_idx = 0

    # State for delta computation
    prev_e2e_count = None
    prev = {}

    hdr = (f"{'#':>4}  {'time':>8}  {'ISL':>7}  {'OSL':>7}"
           f"  {'TTFT':>8}  {'ITL':>8}  {'E2E':>8}  {'Q':>7}"
           f"  {'GPU$':>7}  {'Host$':>7}"
           f"  {'Evict':>8}  {'LoadBk':>8}")
    sep = "─" * len(hdr)
    print(sep); print(hdr); print(sep)

    def _snap(s):
        cached = _by_label(s, "sglang:cached_tokens_total", "cache_source")
        return {
            "e2e_count": _sum(s, "sglang:e2e_request_latency_seconds_count"),
            "ttft_s":    _sum(s, "sglang:time_to_first_token_seconds_sum"),
            "ttft_n":    _sum(s, "sglang:time_to_first_token_seconds_count"),
            "itl_s":     _sum(s, "sglang:inter_token_latency_seconds_sum"),
            "itl_n":     _sum(s, "sglang:inter_token_latency_seconds_count"),
            "e2e_s":     _sum(s, "sglang:e2e_request_latency_seconds_sum"),
            "e2e_n":     _sum(s, "sglang:e2e_request_latency_seconds_count"),
            "q_s":       _sum(s, "sglang:queue_time_seconds_sum"),
            "q_n":       _sum(s, "sglang:queue_time_seconds_count"),
            "isl_s":     _sum(s, "sglang:prompt_tokens_histogram_sum"),
            "isl_n":     _sum(s, "sglang:prompt_tokens_histogram_count"),
            "osl_s":     _sum(s, "sglang:generation_tokens_histogram_sum"),
            "osl_n":     _sum(s, "sglang:generation_tokens_histogram_count"),
            "gpu_hits":  cached.get("device", 0.0),
            "host_hits": cached.get("host", 0.0),
            "prompt_total": _sum(s, "sglang:prompt_tokens_total") or 0.0,
            "evict":     _sum(s, "sglang:evicted_tokens_total") or 0.0,
            "loadbk":    _sum(s, "sglang:load_back_tokens_total") or 0.0,
        }

    def on_exit(*_):
        print(f"\n{sep}\n  {len(turns)} requests recorded")
        save_chart(turns, args.chart)
        sys.exit(0)

    signal.signal(signal.SIGINT,  on_exit)
    signal.signal(signal.SIGTERM, on_exit)

    print("Watching SGLang metrics… (Ctrl+C to stop & save chart)\n")

    while True:
        s    = scrape(url)
        curr = _snap(s)

        if prev:
            d_e2e_n = (curr["e2e_n"] or 0) - (prev["e2e_n"] or 0)
            if d_e2e_n >= 1:
                turn_idx += int(d_e2e_n)
                ts = datetime.now().strftime("%H:%M:%S")

                isl = None
                d_isl_n = (curr["isl_n"] or 0) - (prev["isl_n"] or 0)
                if d_isl_n > 0:
                    isl = ((curr["isl_s"] or 0) - (prev["isl_s"] or 0)) / d_isl_n

                osl = None
                d_osl_n = (curr["osl_n"] or 0) - (prev["osl_n"] or 0)
                if d_osl_n > 0:
                    osl = ((curr["osl_s"] or 0) - (prev["osl_s"] or 0)) / d_osl_n

                ttft_ms  = _delta_ms(curr["ttft_s"], curr["ttft_n"], prev["ttft_s"], prev["ttft_n"])
                itl_ms   = _delta_ms(curr["itl_s"],  curr["itl_n"],  prev["itl_s"],  prev["itl_n"])
                e2e_ms   = _delta_ms(curr["e2e_s"],  curr["e2e_n"],  prev["e2e_s"],  prev["e2e_n"])
                queue_ms = _delta_ms(curr["q_s"],    curr["q_n"],    prev["q_s"],    prev["q_n"])

                d_prompt   = curr["prompt_total"] - prev["prompt_total"]
                d_gpu_hits = curr["gpu_hits"]   - prev["gpu_hits"]
                d_hst_hits = curr["host_hits"]  - prev["host_hits"]
                gpu_hit_pct  = round(d_gpu_hits / d_prompt * 100, 1) if d_prompt > 0 else None
                host_hit_pct = round(d_hst_hits / d_prompt * 100, 1) if d_prompt > 0 else None

                d_evict  = curr["evict"]  - prev["evict"]
                d_loadbk = curr["loadbk"] - prev["loadbk"]

                t = Turn(
                    turn_idx, ts, isl, osl,
                    ttft_ms, itl_ms, e2e_ms, queue_ms,
                    gpu_hit_pct, host_hit_pct,
                    d_evict if d_evict > 0 else 0,
                    d_loadbk if d_loadbk > 0 else 0,
                )
                turns.append(t)

                print(
                    f"{turn_idx:>4}  {ts:>8}"
                    f"  {fmt_f(isl, 0):>7}"
                    f"  {fmt_f(osl, 0):>7}"
                    f"  {fmt_f(ttft_ms,  1, 'ms'):>8}"
                    f"  {fmt_f(itl_ms,   2, 'ms'):>8}"
                    f"  {fmt_f(e2e_ms,   1, 'ms'):>8}"
                    f"  {fmt_f(queue_ms, 1, 'ms'):>7}"
                    f"  {fmt_f(gpu_hit_pct,  1, '%'):>7}"
                    f"  {fmt_f(host_hit_pct, 1, '%'):>7}"
                    f"  {fmt_tok(t.evict_toks):>8}"
                    f"  {fmt_tok(t.loadbk_toks):>8}"
                )

        if not prev:
            count = curr["e2e_n"] or 0
            if count > 0:
                print(f"[seeded at request #{int(count)} — earlier requests not shown]")

        prev = curr
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
