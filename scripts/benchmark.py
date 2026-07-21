"""
Benchmarks a trained checkpoint: device info, model size, cold-start time,
single-sentence latency percentiles, throughput by batch size, peak memory,
and a handful of real example outputs -- everything generate_report.py needs
besides the accuracy numbers metrics.py already produces.

No new dependencies: device/memory introspection uses torch (already a
dependency) plus the Python standard library only.

Usage:
    python benchmark.py --model ./canine-fa-diacritizer --out benchmark.json
"""

import argparse
import json
import os
import platform
import subprocess
import time

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from transformers import CanineTokenizer, CanineForTokenClassification

from diacritizer.preprocessing import load_and_prepare, ID2LABEL
from diacritizer.metrics import predict_examples

load_dotenv()


# ----------------------------------------------------------------------------
# Device / hardware introspection -- torch + stdlib only, no new deps.
# ----------------------------------------------------------------------------
def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def gpu_name(device):
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return "Apple Silicon (MPS)"
    return None


def cpu_name():
    system = platform.system()
    try:
        if system == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


def total_ram_gb():
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024 ** 3), 1)
    except (ValueError, AttributeError, OSError):
        return None


def peak_rss_gb():
    """Process peak resident set size -- the closest stdlib-only proxy for
    "how much RAM does running this need" on CPU (ru_maxrss is bytes on
    macOS, kilobytes on Linux)."""
    try:
        import resource
        val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        val_bytes = val if platform.system() == "Darwin" else val * 1024
        return round(val_bytes / (1024 ** 3), 3)
    except Exception:
        return None


def reset_peak_memory(device):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps" and hasattr(torch.mps, "reset_peak_memory_stats"):
        torch.mps.reset_peak_memory_stats()


def peak_memory_gb(device):
    if device == "cuda":
        return round(torch.cuda.max_memory_allocated() / (1024 ** 3), 3)
    if device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        return round(torch.mps.current_allocated_memory() / (1024 ** 3), 3)
    return peak_rss_gb()


def model_size_info(model, model_path):
    n_params = sum(p.numel() for p in model.parameters())
    disk_mb = None
    if os.path.isdir(model_path):
        total = sum(
            os.path.getsize(os.path.join(model_path, f))
            for f in os.listdir(model_path)
            if f.endswith((".safetensors", ".bin", ".pt"))
        )
        disk_mb = round(total / (1024 ** 2), 1) if total else None
    return {"params": n_params, "disk_mb": disk_mb}


# ----------------------------------------------------------------------------
# Benchmarks
# ----------------------------------------------------------------------------
def run_batch(model, tokenizer, texts, device):
    enc = tokenizer(texts, padding=True, truncation=True, max_length=2048,
                     return_tensors="pt").to(device)
    with torch.no_grad():
        model(**enc)


def benchmark_latency(model, tokenizer, texts, device, n_runs, n_warmup):
    pool = texts or ["نمونه"]
    times = []
    for i in range(n_warmup + n_runs):
        t0 = time.perf_counter()
        run_batch(model, tokenizer, [pool[i % len(pool)]], device)
        if device == "cuda":
            torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000
        if i >= n_warmup:
            times.append(dt_ms)
    arr = np.array(times)
    return {
        "p50": round(float(np.percentile(arr, 50)), 1),
        "p95": round(float(np.percentile(arr, 95)), 1),
        "p99": round(float(np.percentile(arr, 99)), 1),
    }


def benchmark_throughput(model, tokenizer, texts, device, batch_sizes, n_batches, n_warmup_batches):
    pool = texts or ["نمونه"]
    results = []
    for bs in batch_sizes:
        batch = [pool[i % len(pool)] for i in range(bs)]
        for _ in range(n_warmup_batches):
            run_batch(model, tokenizer, batch, device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_batches):
            run_batch(model, tokenizer, batch, device)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        results.append({"batch": bs, "sentences_per_sec": round(bs * n_batches / dt, 1)})
    return results


def benchmark_memory(model, tokenizer, texts, device):
    pool = texts or ["نمونه"]
    scenarios = []

    reset_peak_memory(device)
    run_batch(model, tokenizer, [pool[0]], device)
    if device == "cuda":
        torch.cuda.synchronize()
    scenarios.append({"scenario": f"{device}, single", "gb": peak_memory_gb(device)})

    batch = [pool[i % len(pool)] for i in range(32)]
    reset_peak_memory(device)
    run_batch(model, tokenizer, batch, device)
    if device == "cuda":
        torch.cuda.synchronize()
    scenarios.append({"scenario": f"{device}, batch 32", "gb": peak_memory_gb(device)})
    return scenarios


def build_examples(model, tokenizer, dataset, device, n):
    """
    Structured, not pre-rendered: the report template reconstructs the
    diacritized text itself (and highlights mismatches) from chars +
    per-character label NAMES, so it isn't limited to whatever comparison
    this script happened to bake into a string.
    """
    rows = dataset.select(range(min(n, len(dataset))))
    examples = predict_examples(model, tokenizer, rows, device, batch_size=n)
    out = []
    for ex in examples:
        out.append({
            "chars": ex["chars"],
            "pred_labels": [ID2LABEL[i] for i in ex["pred"]],
            "true_labels": [ID2LABEL[i] for i in ex["true"]],
            "exact_match": ex["pred"] == ex["true"],
        })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="local checkpoint dir, or an HF Hub repo id")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="benchmark.json")
    ap.add_argument("--batch_sizes", default="1,8,16,32,64")
    ap.add_argument("--n_examples", type=int, default=5)
    ap.add_argument("--n_latency_runs", type=int, default=30)
    ap.add_argument("--n_warmup", type=int, default=5)
    ap.add_argument("--n_throughput_batches", type=int, default=3)
    ap.add_argument("--n_warmup_batches", type=int, default=1)
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]
    hub_token = os.environ.get("HF_MODEL_TOKEN")

    print("[benchmark] loading model (timing cold start) ...")
    t0 = time.perf_counter()
    tokenizer = CanineTokenizer.from_pretrained(args.model, token=hub_token)
    model = CanineForTokenClassification.from_pretrained(args.model, token=hub_token)
    device = pick_device()
    model.to(device)
    model.eval()
    cold_start_s = round(time.perf_counter() - t0, 2)
    print(f"[benchmark] cold start: {cold_start_s}s on {device}")

    size_info = model_size_info(model, args.model)

    print("[benchmark] loading a small sample of real sentences ...")
    ds = load_and_prepare(
        data_cfg["hf_repo"], raw_col=data_cfg["raw_col"], diac_col=data_cfg["diac_col"],
        max_chars=data_cfg["max_chars"], test_size=data_cfg["test_size"],
        dev_size=data_cfg["dev_size"], token=os.environ.get("HF_DATASET_TOKEN"),
    )
    test = ds["test"]
    sample_texts = ["".join(row["chars"]) for row in test.select(range(min(20, len(test))))]

    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]

    print(f"[benchmark] latency ({args.n_latency_runs} runs on {device}) ...")
    latency = benchmark_latency(model, tokenizer, sample_texts, device,
                                 args.n_latency_runs, args.n_warmup)

    print(f"[benchmark] throughput by batch size on {device} ...")
    throughput = benchmark_throughput(model, tokenizer, sample_texts, device,
                                       batch_sizes, args.n_throughput_batches, args.n_warmup_batches)

    print(f"[benchmark] memory on {device} ...")
    memory = benchmark_memory(model, tokenizer, sample_texts, device)

    cpu_section = None
    if device != "cpu":
        print("[benchmark] also benchmarking CPU for comparison ...")
        model.to("cpu")
        cpu_latency = benchmark_latency(model, tokenizer, sample_texts, "cpu",
                                         max(5, args.n_latency_runs // 3), args.n_warmup)
        run_batch(model, tokenizer, [sample_texts[0]], "cpu")
        memory.append({"scenario": "cpu, single", "gb": peak_rss_gb()})
        cpu_throughput = benchmark_throughput(model, tokenizer, sample_texts, "cpu", [1], 1, 0)[0]
        cpu_section = {"latency_ms": cpu_latency,
                        "throughput_sentences_per_sec": cpu_throughput["sentences_per_sec"]}
        model.to(device)

    print(f"[benchmark] generating {args.n_examples} example outputs ...")
    examples = build_examples(model, tokenizer, test, device, args.n_examples)

    payload = {
        "device": {
            "primary": device,
            "gpu": gpu_name(device),
            "cpu": cpu_name(),
            "ram_gb": total_ram_gb(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda if device == "cuda" else None,
            "precision": "fp16" if device == "cuda" else "fp32",
        },
        "model_size": size_info,
        "cold_start_s": cold_start_s,
        "latency_ms": latency,
        "throughput": throughput,
        "memory": memory,
        "cpu_comparison": cpu_section,
        "examples": examples,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[benchmark] wrote {args.out}")


if __name__ == "__main__":
    main()
