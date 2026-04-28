#!/usr/bin/env python3
"""Aggregate completed cXXX.json levels from results_benchmarks/ and print a table."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results_benchmarks"

def fmt(v, unit=""):
    if v is None:
        return "-"
    if isinstance(v, float) and v >= 1000:
        return f"{v/1000:.1f}s"
    if isinstance(v, (int, float)):
        return f"{v:.0f}{unit}"
    return str(v)

def collect_latest():
    """Group c*.json files by setup, keeping the newest run per concurrency."""
    by_setup = {}  # setup -> {concurrency: (mtime, json_dict)}
    for d in RESULTS.glob("bench_hybrid-*"):
        setup = d.name.split("_")[1]  # e.g. "hybrid-mtier"
        for f in d.glob("c*.json"):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            c = data["concurrency"]
            mt = f.stat().st_mtime
            cur = by_setup.setdefault(setup, {}).get(c)
            if cur is None or mt > cur[0]:
                by_setup[setup][c] = (mt, data)
    return by_setup

def print_table(setup, levels):
    print(f"\n══ {setup} ══")
    hdr = (f"{'C':>3} {'n_ok':>5} {'wall':>5} {'out_tok/s':>10} "
           f"{'TTFT_p50':>9} {'TTFT_p95':>9} {'Queue_p50':>10} {'E2E_p50':>8} "
           f"{'ITL_p50':>8} {'KV%':>5} {'gpu_hit%':>9} {'host_hit%':>10} "
           f"{'evicted':>10} {'load_back':>10}")
    print(hdr)
    print("-" * len(hdr))
    for c in sorted(levels):
        lv = levels[c][1]
        tp = lv["throughput"]; ca = lv.get("cache", {}); off = lv.get("kv_offload", {})
        print(f"{c:>3} {lv['n_ok']:>5} {lv['wall_time_s']:>5.0f} "
              f"{fmt(tp['output_tokens_per_sec']):>10} "
              f"{fmt(lv['ttft_ms']['p50'],'ms'):>9} {fmt(lv['ttft_ms']['p95'],'ms'):>9} "
              f"{fmt(lv['queue_ms']['p50'],'ms'):>10} {fmt(lv['e2e_ms']['p50'],'ms'):>8} "
              f"{fmt(lv['itl_ms']['p50'],'ms'):>8} "
              f"{fmt(ca.get('kv_usage_pct'),'%'):>5} "
              f"{fmt(ca.get('gpu_hit_rate_pct'),'%'):>9} "
              f"{fmt(ca.get('cpu_hit_rate_pct'),'%'):>10} "
              f"{fmt(off.get('evicted_tokens')):>10} "
              f"{fmt(off.get('load_back_tokens')):>10}")

if __name__ == "__main__":
    by_setup = collect_latest()
    if not by_setup:
        print("No results yet.")
    else:
        for setup in sorted(by_setup):
            print_table(setup, by_setup[setup])
