"""
One-time migration: move a Hub repo from the old nested layout
(best/config.json, best/model.safetensors, ...) to a standard, single-model
repo (config.json, model.safetensors, ... at the repo ROOT), matching what
train.py now pushes directly (see checkpointing.py).

Pure Hub-to-Hub operation -- downloads only the real model files out of
--subfolder (ignores any nested checkpoint-<step>/ dirs left behind by the
old push-to-hub bug), re-uploads them to the repo root, then deletes the old
--subfolder entirely. Doesn't touch local disk beyond a scratch tmp dir, so
it can run anywhere with network access and HF_MODEL_TOKEN (this machine, a
Vast.ai instance, CI, ...) -- it does not need the original training
checkpoints to be present locally.

Usage:
    python scripts/migrate_hub_layout.py --hub_repo_id PedramR/canine-fa-diacritizer
"""

import argparse
import os
import tempfile

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

load_dotenv()

MODEL_FILES = (
    "config.json",
    "metrics.json",
    "model.safetensors",
    "tokenizer_config.json",
    "training_args.bin",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hub_repo_id", required=True)
    ap.add_argument("--subfolder", default="best",
                    help="old subfolder holding the real model files, to be moved to the root")
    ap.add_argument("--keep_old_subfolder", action="store_true",
                    help="don't delete --subfolder from the Hub after migrating (default: delete it)")
    args = ap.parse_args()

    token = os.environ.get("HF_MODEL_TOKEN")
    if not token:
        raise SystemExit("HF_MODEL_TOKEN must be set in .env")

    api = HfApi(token=token)
    repo_files = set(api.list_repo_files(args.hub_repo_id))
    present = [f for f in MODEL_FILES if f"{args.subfolder}/{f}" in repo_files]
    if not present:
        raise SystemExit(f"no known model files found under {args.subfolder}/ in {args.hub_repo_id}")
    print(f"[info] found {present} under {args.subfolder}/ -- fetching...")

    with tempfile.TemporaryDirectory() as tmp:
        snapshot_download(
            repo_id=args.hub_repo_id,
            token=token,
            local_dir=tmp,
            allow_patterns=[f"{args.subfolder}/{f}" for f in present],
        )
        local_subfolder = os.path.join(tmp, args.subfolder)

        print(f"[info] uploading {present} to the repo root...")
        api.upload_folder(
            folder_path=local_subfolder,
            repo_id=args.hub_repo_id,
            path_in_repo=None,  # repo root
            token=token,
            commit_message=f"Move {args.subfolder}/ contents to the repo root (standard layout)",
        )

    print(f"[info] pushed to https://huggingface.co/{args.hub_repo_id}")

    if not args.keep_old_subfolder:
        print(f"[info] deleting {args.subfolder}/ (including any nested checkpoint dirs)...")
        api.delete_folder(
            path_in_repo=args.subfolder,
            repo_id=args.hub_repo_id,
            token=token,
            commit_message=f"Remove old {args.subfolder}/ subfolder",
        )
        print(f"[info] deleted {args.subfolder}/")


if __name__ == "__main__":
    main()
