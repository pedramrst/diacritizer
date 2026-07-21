"""
Renders the manager-facing HTML performance report by combining metrics.json
(from metrics.py / train.py) and benchmark.json (from benchmark.py) into
report_template.html.

Pure standard library -- no ML dependencies here. This step only reads two
JSON files and writes HTML; it does not import torch, transformers, or
datasets, so it has zero new (or even existing project) dependencies beyond
what ships with Python.

Usage:
    python generate_report.py --metrics metrics.json --benchmark benchmark.json --out report.html
"""

import argparse
import json
import os
from datetime import date


def load_json(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_payload(args):
    metrics = load_json(args.metrics)
    benchmark = load_json(args.benchmark)

    baseline = metrics.pop("baseline", {})
    accuracy = {"model": metrics, "baseline": baseline}

    return {
        "is_mock": args.mock,
        "mock_note": args.mock_note if args.mock else None,
        "model_name": args.model_name,
        "base_model": args.base_model,
        "evaluated_at": args.evaluated_at or date.today().isoformat(),
        "dataset": args.dataset,
        "split_sizes": args.split_sizes,
        "test_set": {
            "n_sentences": args.n_sentences,
            "n_chars": metrics.get("n_chars"),
            "n_words": metrics.get("n_words"),
        },
        "accuracy": accuracy,
        "device": benchmark.get("device", {}),
        "model_size": benchmark.get("model_size", {}),
        "cold_start_s": benchmark.get("cold_start_s"),
        "latency_ms": benchmark.get("latency_ms", {}),
        "throughput": benchmark.get("throughput", []),
        "memory": benchmark.get("memory", []),
        "cpu_comparison": benchmark.get("cpu_comparison"),
        "examples": benchmark.get("examples", []),
    }


def render_html(template_path, payload, out_path):
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()
    data_json = json.dumps(payload, ensure_ascii=False)
    # Safe to embed inside an inline <script>: an unescaped "</script" inside
    # a JSON string value would otherwise prematurely close the tag.
    data_json_safe = data_json.replace("</script", "<\\/script")
    html = template.replace("__REPORT_DATA__", data_json_safe)

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", default="metrics.json", help="from metrics.py --out or train.py")
    ap.add_argument("--benchmark", default="benchmark.json", help="from benchmark.py")
    ap.add_argument("--template", default="templates/report_template.html")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--model_name", default="canine-fa-diacritizer")
    ap.add_argument("--base_model", default="google/canine-s")
    ap.add_argument("--dataset", default="avaeziaiteam/harakat-dataset (7,180 sentences, deduplicated)")
    ap.add_argument("--split_sizes", default="6,462 train / 359 dev / 359 test")
    ap.add_argument("--n_sentences", type=int, default=359)
    ap.add_argument("--evaluated_at", default=None, help="defaults to today's date")
    ap.add_argument("--mock", action="store_true",
                    help="show the 'sample report / mock data' banner in the output")
    ap.add_argument("--mock_note", default="Some or all numbers on this page are placeholders, "
                                             "not a real evaluation run.")
    args = ap.parse_args()

    payload = build_payload(args)
    render_html(args.template, payload, args.out)
    print(f"[info] wrote {args.out}")

    if not os.path.exists(args.metrics):
        print(f"[warn] {args.metrics} not found -- the accuracy section will be empty. "
              f"Run: python metrics.py --model <path> --out {args.metrics}")
    if not os.path.exists(args.benchmark):
        print(f"[warn] {args.benchmark} not found -- speed/memory/examples will be empty. "
              f"Run: python benchmark.py --model <path> --out {args.benchmark}")


if __name__ == "__main__":
    main()
