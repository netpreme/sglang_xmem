import argparse
import csv
import os
import time
from datetime import datetime

import torch
import sglang as sgl

HOST_CACHE_SIZE_GB = 60
PAGE_SIZE = 64
MEM_FRACTION_STATIC = 0.5
NUM_DECODED_TOKENS_PER_PROMPT = 1
HICACHE_MEM_LAYOUT = "layer_first"

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
KV_SIZE_PER_TOKEN = 2 * 128 * 4 * 2 * 48

host_bytes_to_use = HOST_CACHE_SIZE_GB << 30
cache_tokens_capacity = host_bytes_to_use // KV_SIZE_PER_TOKEN

HIT_PROMPT_SIZE = 30720
MISS_PROMPT_SIZE = 30720
NUM_PROMPTS = 200
HIT_PERCENTS_TO_TEST = (0, 20, 40, 60, 80, 100)

PROMPT_SIZES_IN_K_TO_TEST = (1,) + tuple(range(10, 91, 10))


def _print_ttft_table(prompt_sizes_k, prefill_ms, cpu_ms, xmem_ms):
    col_w = max(8, *(len(f"{s}K") + 2 for s in prompt_sizes_k))
    label_w = 28
    sep = "-" * (label_w + col_w * len(prompt_sizes_k) + 1)

    def row(label, values, fmt):
        cells = "".join(fmt(v).rjust(col_w) for v in values)
        print(f"{label:<{label_w}}{cells}")

    print()
    print(sep)
    row("Prompt length", [f"{s}K" for s in prompt_sizes_k], str)
    print(sep)
    row("TTFT (prefill), ms",        prefill_ms, str)
    row("TTFT (cached, CPU), ms",    cpu_ms,     str)
    row("TTFT (cached, X-Mem), ms",  xmem_ms,    str)
    print(sep)
    speedups = [
        f"{c/m:.2f}x" if m else "N/A"
        for c, m in zip(cpu_ms, xmem_ms)
    ]
    row("X-Mem Speed-Up", speedups, str)
    print(sep)
    print()


def _print_tput_table(hit_percents, cpu_tps, xmem_tps):
    col_w = max(8, *(len(f"{p}%") + 2 for p in hit_percents))
    label_w = 28
    sep = "-" * (label_w + col_w * len(hit_percents) + 1)

    def row(label, values, fmt):
        cells = "".join(fmt(v).rjust(col_w) for v in values)
        print(f"{label:<{label_w}}{cells}")

    print()
    print(sep)
    row("Cache hit %", [f"{p}%" for p in hit_percents], str)
    print(sep)
    row("Throughput (CPU), tok/s",   cpu_tps,  str)
    row("Throughput (X-Mem), tok/s", xmem_tps, str)
    print(sep)
    speedups = [
        f"{m/c:.2f}x" if c else "N/A"
        for c, m in zip(cpu_tps, xmem_tps)
    ]
    row("X-Mem Speed-Up", speedups, str)
    print(sep)
    print()


def make_sampling_params():
    return {
        "max_new_tokens": NUM_DECODED_TOKENS_PER_PROMPT,
        "ignore_eos": True,
        "temperature": 0,
    }


def main(run_ttft: bool, run_tput: bool, backends=(False, True)):
    sampling_params = make_sampling_params()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "benchmark_results")
    os.makedirs(results_dir, exist_ok=True)

    ttft_rows = []
    tput_rows = []

    for use_xmem in backends:
        backend_name = "X-Mem" if use_xmem else "CPU DRAM"
        print(f"{'='*60}")
        print(f"Testing with: {backend_name}")
        print(f"{'='*60}")

        llm = sgl.Engine(
            model_path=MODEL,
            page_size=PAGE_SIZE,
            mem_fraction_static=MEM_FRACTION_STATIC,
            enable_hierarchical_cache=True,
            hicache_size=HOST_CACHE_SIZE_GB,
            hicache_write_policy="write_through",
            hicache_io_backend="kernel",
            hicache_mem_layout=HICACHE_MEM_LAYOUT,
            hicache_use_xmem=use_xmem,
            log_level="warning",
        )

        if run_ttft:
            max_prompt_size = max(PROMPT_SIZES_IN_K_TO_TEST) << 10
            iterations_count = cache_tokens_capacity // max_prompt_size
            for i, prompt_size_k in enumerate(PROMPT_SIZES_IN_K_TO_TEST):
                prompt_size = prompt_size_k << 10

                print("Testing prompt length:", prompt_size_k, "K")
                total_prefill_time = 0
                for j in range(iterations_count):
                    input_ids = [i] + [j] * (prompt_size - 1)
                    start_time = time.perf_counter()
                    llm.generate(input_ids=input_ids, sampling_params=sampling_params)
                    total_prefill_time += time.perf_counter() - start_time

                average_prefill_time_ms = int(
                    1000 * (total_prefill_time / iterations_count)
                )
                print("Average prefill time:", average_prefill_time_ms, "ms")

                total_host_load_time = 0
                for j in range(iterations_count):
                    input_ids = [i] + [j] * (prompt_size - 1)
                    start_time = time.perf_counter()
                    llm.generate(input_ids=input_ids, sampling_params=sampling_params)
                    total_host_load_time += time.perf_counter() - start_time

                average_host_load_time_ms = int(
                    1000 * (total_host_load_time / iterations_count)
                )

                print("Average host load time:", average_host_load_time_ms, "ms")
                print()
                ttft_rows.append(
                    {
                        "backend": backend_name,
                        "prompt_size_k": prompt_size_k,
                        "prefill_time_ms": average_prefill_time_ms,
                        "host_load_time_ms": average_host_load_time_ms,
                    }
                )

        if run_tput:
            max_prompt_size = max(HIT_PROMPT_SIZE, MISS_PROMPT_SIZE)
            cache_prompts_capacity = cache_tokens_capacity // max_prompt_size

            hit_input_ids = [
                [0] + [i + 3] * (HIT_PROMPT_SIZE - 1)
                for i in range(cache_prompts_capacity)
            ]
            miss_input_ids = [
                [1] + [i + 3] * (MISS_PROMPT_SIZE - 1) for i in range(NUM_PROMPTS)
            ]
            reset_cache_input_ids = [
                [2] + [i + 3] * (max_prompt_size - 1)
                for i in range(cache_prompts_capacity)
            ]

            for hit_percent in HIT_PERCENTS_TO_TEST:
                hit_indexes = set()
                if hit_percent > 0:
                    hit_indexes = {
                        int(i * 100 / hit_percent) for i in range(hit_percent)
                    }
                assert len(hit_indexes) == hit_percent

                hit_prompts_count = max(1, cache_prompts_capacity * hit_percent // 100)
                input_ids_batch = []
                hit_idx = 0
                for i in range(NUM_PROMPTS):
                    if (i % 100) in hit_indexes:
                        input_ids_batch.append(hit_input_ids[hit_idx])
                        hit_idx = (hit_idx + 1) % hit_prompts_count
                    else:
                        input_ids_batch.append(miss_input_ids[i])

                llm.generate(
                    input_ids=reset_cache_input_ids,
                    sampling_params=sampling_params,
                )

                if hit_percent:
                    llm.generate(
                        input_ids=hit_input_ids[:hit_prompts_count],
                        sampling_params=sampling_params,
                    )

                time.sleep(1)

                print("Testing hit percent:", hit_percent)

                start_time = time.perf_counter()
                outputs = llm.generate(
                    input_ids=input_ids_batch,
                    sampling_params=sampling_params,
                )
                total_time = time.perf_counter() - start_time

                if isinstance(outputs, list):
                    num_tokens = sum(
                        o["meta_info"]["prompt_tokens"]
                        + o["meta_info"]["completion_tokens"]
                        for o in outputs
                    )
                else:
                    num_tokens = (
                        outputs["meta_info"]["prompt_tokens"]
                        + outputs["meta_info"]["completion_tokens"]
                    )
                tokens_per_sec = int(num_tokens / total_time)
                print(tokens_per_sec, "tokens/sec")
                print()
                tput_rows.append(
                    {
                        "backend": backend_name,
                        "hit_percent": hit_percent,
                        "tokens_per_sec": tokens_per_sec,
                        "total_time_s": round(total_time, 3),
                        "num_tokens": num_tokens,
                    }
                )

        llm.shutdown()
        torch.cuda.empty_cache()
        if use_xmem:
            try:
                from xmem.mtier_sdk import reset_all
                reset_all()
                print("X-Mem allocations released.")
            except Exception as e:
                print(f"Warning: failed to reset X-Mem: {e}")
        time.sleep(5)

    if ttft_rows:
        path = os.path.join(results_dir, f"ttft_{timestamp}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ttft_rows[0].keys())
            writer.writeheader()
            writer.writerows(ttft_rows)
        print(f"TTFT results saved to {path}")

        # Print summary table
        sizes = list(PROMPT_SIZES_IN_K_TO_TEST)
        cpu_by_size = {r["prompt_size_k"]: r for r in ttft_rows if r["backend"] == "CPU DRAM"}
        xmem_by_size = {r["prompt_size_k"]: r for r in ttft_rows if r["backend"] == "X-Mem"}
        prefill_ms = [cpu_by_size.get(s, xmem_by_size.get(s, {})).get("prefill_time_ms", 0) for s in sizes]
        cpu_ms = [cpu_by_size.get(s, {}).get("host_load_time_ms", 0) for s in sizes]
        xmem_ms = [xmem_by_size.get(s, {}).get("host_load_time_ms", 0) for s in sizes]
        _print_ttft_table(sizes, prefill_ms, cpu_ms, xmem_ms)

    if tput_rows:
        path = os.path.join(results_dir, f"tput_{timestamp}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=tput_rows[0].keys())
            writer.writeheader()
            writer.writerows(tput_rows)
        print(f"Throughput results saved to {path}")

        # Print summary table
        percents = list(HIT_PERCENTS_TO_TEST)
        cpu_by_pct = {r["hit_percent"]: r for r in tput_rows if r["backend"] == "CPU DRAM"}
        xmem_by_pct = {r["hit_percent"]: r for r in tput_rows if r["backend"] == "X-Mem"}
        cpu_tps = [cpu_by_pct.get(p, {}).get("tokens_per_sec", 0) for p in percents]
        xmem_tps = [xmem_by_pct.get(p, {}).get("tokens_per_sec", 0) for p in percents]
        _print_tput_table(percents, cpu_tps, xmem_tps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ttft", action="store_true", default=False)
    group.add_argument("--tput", action="store_true", default=False)
    parser.add_argument(
        "--backend",
        choices=["both", "cpu", "xmem"],
        default="both",
    )
    args = parser.parse_args()

    if args.backend == "both":
        backends = (False, True)
    elif args.backend == "cpu":
        backends = (False,)
    else:
        backends = (True,)

    main(args.ttft, args.tput, backends)
