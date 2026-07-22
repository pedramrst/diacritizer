"""
Run train.py once per config file in a directory, then print a comparison
table of their test-set metrics.

Each config's checkpoints are namespaced under --out_root by the config's
filename, so runs never collide. All configs share one TensorBoard root
(--out_root/runs) -- train.py automatically nests each run under
<run_name>/ (defaulting run_name to the config filename), so TensorBoard
still shows every run as a separate, comparable line when pointed at that
shared parent: `tensorboard --logdir <out_root>/runs`.

train.py always writes a metrics.json (the full DER/WER/DER*/WER* test
report) to its output dir -- this script just reads those back and
tabulates them.

Usage:
    python sweep.py --configs configs/ --out_root ./sweeps/run1
    python sweep.py --configs configs/ --out_root ./sweeps/run1 --extra_args "--push_to_hub"

To continue training the winning config for more epochs afterwards, just
point train.py's --model at that config's output dir (or its Hub repo) and
rerun with a higher --epochs -- no special "continue" flag needed, since
--model already accepts any checkpoint directory or repo id.

If a sweep run is interrupted, just rerun the same sweep.py command -- each
train.py invocation auto-resumes from its own --out if a checkpoint is
already there (see train.py --no_auto_resume to disable that).
"""

import argparse
import glob
import json
import os
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", default="configs",
                    help="directory of *.yaml config files, one per experiment")
    ap.add_argument("--out_root", default="./sweeps",
                    help="each config's output goes under out_root/<config filename>")
    ap.add_argument("--extra_args", default="",
                    help="extra args appended verbatim to every train.py invocation, "
                         "e.g. '--push_to_hub --epochs 3'")
    args = ap.parse_args()

    config_files = sorted(glob.glob(os.path.join(args.configs, "*.yaml")))
    if not config_files:
        raise SystemExit(f"no *.yaml files found under {args.configs}")

    tb_root = os.path.join(args.out_root, "runs")

    results = []
    for cfg_path in config_files:
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        out_dir = os.path.join(args.out_root, name)
        os.makedirs(out_dir, exist_ok=True)

        # --run_name defaults to the config filename, so leaving it unset
        # here already gives each config its own subfolder under tb_root.
        train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
        cmd = [sys.executable, train_script, "--config", cfg_path,
               "--out", out_dir, "--tensorboard_dir", tb_root]
        if args.extra_args:
            cmd += args.extra_args.split()

        print(f"\n[sweep] === running {name} ===\n{' '.join(cmd)}\n")
        subprocess.run(cmd, check=True)

        metrics_path = os.path.join(out_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, encoding="utf-8") as f:
                results.append({"config": name, **json.load(f)})
        else:
            print(f"[sweep][warn] no metrics.json found for {name}")

    if not results:
        print("[sweep] no results to compare")
        return

    print("\n[sweep] === comparison (test set) ===")
    keys = ["DER", "WER", "DER*", "WER*", "macro_f1_diacritics"]
    print(f"{'config':<30}" + "".join(f"{k:>16}" for k in keys))
    for r in sorted(results, key=lambda r: r.get("macro_f1_diacritics", 0), reverse=True):
        print(f"{r['config']:<30}" + "".join(f"{r.get(k, float('nan')):>16.4f}" for k in keys))


if __name__ == "__main__":
    main()
