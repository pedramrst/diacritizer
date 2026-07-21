"""
Persian Diacritization via character-level token classification.

Model: google/canine-s  (tokenization-free, operates directly on Unicode chars)
Task : per-character labeling over the full label scheme in preprocessing.py
       (NONE, FATHA, DAMMA, KASRA, SOKUN, SHADDA, SHADDA_FATHA, SHADDA_DAMMA,
       SHADDA_KASRA, FATHATAN).

Why CANINE and not ParsBERT/ALBERT:
    Diacritics attach to individual LETTERS. A subword tokenizer (SentencePiece
    in albert-fa-zwnj, WordPiece in ParsBERT) packs many letters into one token,
    so it cannot cleanly carry a per-letter label. CANINE reads one Unicode
    character at a time and upsamples back to full length for token
    classification -- a structural match for this task. This mirrors the Arabic
    SOTA (CATT), which deliberately uses a character-level encoder.

Data loading, diacritic stripping, and label assignment live in
preprocessing.py (shared with diacritize.py so training and inference always
agree on what a label id means).

config.yaml's `training:` section is passed to transformers.TrainingArguments
almost verbatim -- use TrainingArguments' OWN field names there (num_train_epochs,
per_device_train_batch_size, output_dir, gradient_accumulation_steps, ...), and
whatever you add just works, no code change needed. --epochs/--lr/--batch_size/
--seed/--out are optional CLI shortcuts for the common ones (only applied if you
actually pass them); anything else, edit config.yaml directly. A few keys are
filled in as DEFAULTS this pipeline needs (eval_strategy="epoch",
metric_for_best_model="macro_f1_diacritics", report_to=["tensorboard"], ...) --
config/CLI values still win if you set them, but overriding eval_strategy/
save_strategy/metric_for_best_model/output_dir/logging_dir needs care, since
checkpoint retention, auto-resume, and TensorBoard/Hub push all assume they
match what those defaults set up (see README).

HF tokens come from .env:
    HF_DATASET_TOKEN -- read access to the private dataset repo
    HF_MODEL_TOKEN   -- write access to the (different-account) model repo,
                        only needed with --push_to_hub

Checkpoint durability (important on ephemeral instances, e.g. Vast.ai): with
--push_to_hub, the best `checkpoints.keep_best_n` checkpoints (by
macro_f1_diacritics on dev, each with a per-checkpoint metrics.json) plus the
TensorBoard logs are pushed to the Hub as training progresses -- not only at
the end -- via checkpointing.TopKCheckpointCallback. At the end, the last
epoch's full checkpoint (with optimizer/scheduler state) is pushed as
final/, and the best-at-end model as best/. To resume after losing the
instance mid-run, download a checkpoint-<step>/ or final/ folder back down
(see checkpointing.download_checkpoint_from_hub) and pass its local path to
--resume_from_checkpoint.
"""

import argparse
import json
import os

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from transformers import (
    CanineTokenizer,
    CanineForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from diacritizer.preprocessing import LABEL2ID, ID2LABEL, NUM_LABELS, load_and_prepare
from diacritizer.metrics import predict_examples, sequence_metrics, baseline_examples, format_report
from diacritizer.checkpointing import TopKCheckpointCallback, upload_folder_to_hub

load_dotenv()

# Filled in as defaults -- config.yaml's training: section (and any CLI
# shortcut) still overrides these if set, but the checkpoint retention /
# auto-resume / TensorBoard+Hub push machinery all assume eval_strategy==
# save_strategy and metric_for_best_model=="macro_f1_diacritics"; changing
# those needs the corresponding code to change too, not just the config.
PIPELINE_DEFAULTS = {
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "macro_f1_diacritics",
    "greater_is_better": True,
    "report_to": ["tensorboard"],
}


# ----------------------------------------------------------------------------
# 1. Tokenize for CANINE
# ----------------------------------------------------------------------------
# CANINE consumes the raw string and maps each char -> Unicode code point. It
# adds [CLS]/[SEP], so we align labels with a leading/trailing -100 (ignored by
# the loss). We feed the model the JOINED clean string; because CANINE is
# strictly one-code-point-per-char, our per-char labels line up 1:1 with the
# characters between the special tokens.
def build_tokenize_fn(tokenizer, max_chars):
    def tok(batch):
        texts = ["".join(chars) for chars in batch["chars"]]
        enc = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_chars + 2,  # +2 for [CLS] and [SEP]
        )
        all_labels = []
        for labels in batch["labels"]:
            # -100 at [CLS], the real labels, -100 at [SEP]. -100 is ignored in loss.
            aligned = [-100] + labels[:max_chars] + [-100]
            all_labels.append(aligned)
        enc["labels"] = all_labels
        return enc
    return tok


# ----------------------------------------------------------------------------
# 2. Metrics: DER (per-char) + per-class F1. NOT plain accuracy.
# ----------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    # Flatten and drop -100 (special tokens / padding).
    flat_pred, flat_true = [], []
    for p_row, l_row in zip(preds, labels):
        for p, l in zip(p_row, l_row):
            if l != -100:
                flat_pred.append(p)
                flat_true.append(l)
    flat_pred = np.array(flat_pred)
    flat_true = np.array(flat_true)

    total = len(flat_true)
    # Diacritic Error Rate: fraction of characters with a wrong label.
    der = float(np.mean(flat_pred != flat_true))

    # Per-class precision / recall / F1 (macro), computed manually to avoid deps.
    metrics = {"DER": der, "char_accuracy": 1.0 - der, "n_chars": total}
    f1s = []
    for cls_id, cls_name in ID2LABEL.items():
        tp = int(np.sum((flat_pred == cls_id) & (flat_true == cls_id)))
        fp = int(np.sum((flat_pred == cls_id) & (flat_true != cls_id)))
        fn = int(np.sum((flat_pred != cls_id) & (flat_true == cls_id)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        metrics[f"f1_{cls_name}"] = f1
        # Exclude the dominant NONE class from the macro average of interest.
        if cls_name != "NONE":
            f1s.append(f1)
    metrics["macro_f1_diacritics"] = float(np.mean(f1s)) if f1s else 0.0
    return metrics


# ----------------------------------------------------------------------------
# 3. Config + CLI
# ----------------------------------------------------------------------------
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_argparser(cfg):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml",
                    help="path to the YAML config (already loaded for defaults below)")

    data_cfg, model_cfg = cfg["data"], cfg["model"]
    log_cfg, hub_cfg = cfg["logging"], cfg["hub"]
    ckpt_cfg = cfg["checkpoints"]

    ap.add_argument("--data", default=data_cfg["hf_repo"],
                    help="HF dataset repo id, or a local .csv/.parquet path")
    ap.add_argument("--raw_col", default=data_cfg["raw_col"])
    ap.add_argument("--diac_col", default=data_cfg["diac_col"])
    ap.add_argument("--max_chars", type=int, default=data_cfg["max_chars"],
                    help="max characters per example (<=2046)")
    ap.add_argument("--test_size", type=float, default=data_cfg["test_size"])
    ap.add_argument("--dev_size", type=float, default=data_cfg["dev_size"])

    ap.add_argument("--model", default=model_cfg["name"],
                    help="google/canine-s (subword loss) or google/canine-c")

    # Optional shortcuts for the training: fields people tweak most often.
    # Only applied if actually passed (default=None) -- everything else in
    # transformers.TrainingArguments is set via config.yaml's training:
    # section directly, using TrainingArguments' own field names.
    ap.add_argument("--out", default=None, help="shortcut for training.output_dir")
    ap.add_argument("--epochs", type=int, default=None, help="shortcut for training.num_train_epochs")
    ap.add_argument("--lr", type=float, default=None, help="shortcut for training.learning_rate")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="shortcut for training.per_device_train_batch_size "
                         "and training.per_device_eval_batch_size (both)")
    ap.add_argument("--seed", type=int, default=None,
                    help="shortcut for training.seed; also seeds the train/dev/test split")

    ap.add_argument("--tensorboard_dir", default=log_cfg["tensorboard_dir"],
                    help="parent TensorBoard log dir; this run's events go in "
                         "<tensorboard_dir>/<run_name>/, so pointing TensorBoard at "
                         "the parent shows every run side by side")
    ap.add_argument("--run_name", default=None,
                    help="name for this run's TensorBoard subfolder (and Hub "
                         "tensorboard_logs/ subfolder); defaults to the --config "
                         "filename, e.g. configs/lr_5e-5.yaml -> 'lr_5e-5'")

    ap.add_argument("--push_to_hub", action="store_true", default=hub_cfg["push_to_hub"])
    ap.add_argument("--hub_repo_id", default=hub_cfg["repo_id"])
    ap.add_argument("--hub_private", action="store_true", default=hub_cfg["private"])

    ap.add_argument("--keep_best_n", type=int, default=ckpt_cfg["keep_best_n"],
                    help="how many best-by-metric checkpoints to keep locally / push to the Hub")

    ap.add_argument("--resume_from_checkpoint", default=None,
                    help="local checkpoint dir (with optimizer/scheduler state) to resume from")
    ap.add_argument("--resume_from_hub_repo", default=None,
                    help="if set, download --resume_from_hub_subfolder from this HF repo "
                         "into --resume_download_dir before resuming (use after losing the instance)")
    ap.add_argument("--resume_from_hub_subfolder", default="final",
                    help="which subfolder to download for --resume_from_hub_repo "
                         "(final, best, or checkpoint-<step>)")
    ap.add_argument("--resume_download_dir", default="./resume_checkpoint")
    ap.add_argument("--no_auto_resume", action="store_true",
                    help="by default, if the resolved output_dir already has a checkpoint "
                         "(e.g. this exact command was interrupted and rerun), training "
                         "resumes from it automatically; pass this to force a fresh run instead")
    return ap


def build_training_kwargs(cfg, args):
    """
    config.yaml's training: section, using transformers.TrainingArguments'
    own field names, passed through almost verbatim -- PIPELINE_DEFAULTS
    fill in what this pipeline needs, but config/CLI values win if set.
    """
    kwargs = {**PIPELINE_DEFAULTS, **cfg["training"]}

    if args.out is not None:
        kwargs["output_dir"] = args.out
    if args.epochs is not None:
        kwargs["num_train_epochs"] = args.epochs
    if args.lr is not None:
        kwargs["learning_rate"] = args.lr
    if args.batch_size is not None:
        kwargs["per_device_train_batch_size"] = args.batch_size
        kwargs["per_device_eval_batch_size"] = args.batch_size
    if args.seed is not None:
        kwargs["seed"] = args.seed

    if "output_dir" not in kwargs:
        raise SystemExit("training.output_dir must be set in config.yaml (or pass --out)")

    # Prefer bf16 on GPUs that support it (Ampere+: A100, RTX 30xx/40xx, ...)
    # -- same speed as fp16 but no loss-scaling/NaN risk from its narrower
    # dynamic range, which matters here given how imbalanced the label
    # distribution is (a handful of rare classes, e.g. Fathatan/Shadda
    # combos, produce small-magnitude gradients). Falls back to fp16 on
    # older GPUs, neither on CPU/MPS. Config/CLI can still force either.
    if "fp16" not in kwargs and "bf16" not in kwargs:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            kwargs["bf16"] = True
        elif torch.cuda.is_available():
            kwargs["fp16"] = True
    # Our own concern (Hub push + run-name nesting both key off this same
    # path), not something config.yaml's training.logging_dir should diverge
    # from -- always wins over whatever's in cfg["training"].
    kwargs["logging_dir"] = args.tensorboard_dir
    return kwargs


# ----------------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------------
def main():
    # Parse --config first so its values become the defaults for everything else.
    pre_ap = argparse.ArgumentParser(add_help=False)
    pre_ap.add_argument("--config", default="config.yaml")
    pre_args, _ = pre_ap.parse_known_args()
    cfg = load_config(pre_args.config)
    print(f"[info] using config: {pre_args.config}")

    args = build_argparser(cfg).parse_args()

    if args.run_name is None:
        args.run_name = os.path.splitext(os.path.basename(args.config))[0]
    # Nest under run_name so multiple configs can share one tensorboard_dir
    # root and still show up as separate, comparable runs in the TensorBoard
    # UI (point `tensorboard --logdir <tensorboard_dir>` at the parent).
    args.tensorboard_dir = os.path.join(args.tensorboard_dir, args.run_name)

    training_kwargs = build_training_kwargs(cfg, args)
    out_dir = training_kwargs["output_dir"]
    seed = training_kwargs.get("seed", 42)

    dataset_token = os.environ.get("HF_DATASET_TOKEN")
    model_token = os.environ.get("HF_MODEL_TOKEN")
    if args.push_to_hub and not model_token:
        raise SystemExit("--push_to_hub requires HF_MODEL_TOKEN to be set in .env")

    if args.resume_from_hub_repo:
        from diacritizer.checkpointing import download_checkpoint_from_hub
        args.resume_from_checkpoint = download_checkpoint_from_hub(
            args.resume_from_hub_repo, args.resume_from_hub_subfolder,
            model_token, args.resume_download_dir,
        )
        print(f"[info] downloaded {args.resume_from_hub_subfolder} from "
              f"{args.resume_from_hub_repo} -> {args.resume_from_checkpoint}")
    elif not args.resume_from_checkpoint and not args.no_auto_resume:
        # Same command, same output_dir, rerun after an interruption (crash,
        # OOM kill, instance reboot but disk survived): pick up where it left
        # off instead of silently starting over. Use --no_auto_resume to opt
        # out (e.g. output_dir has stale checkpoints from a different config).
        from transformers.trainer_utils import get_last_checkpoint
        if os.path.isdir(out_dir):
            last_ckpt = get_last_checkpoint(out_dir)
            if last_ckpt:
                args.resume_from_checkpoint = last_ckpt
                print(f"[info] found an existing checkpoint at {last_ckpt}, resuming "
                      f"automatically (pass --no_auto_resume to start fresh instead)")

    # Load + prepare (train/dev/test split)
    ds = load_and_prepare(
        args.data,
        raw_col=args.raw_col,
        diac_col=args.diac_col,
        max_chars=args.max_chars,
        test_size=args.test_size,
        dev_size=args.dev_size,
        seed=seed,
        token=dataset_token,
    )

    tokenizer = CanineTokenizer.from_pretrained(args.model)
    model = CanineForTokenClassification.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    tok_fn = build_tokenize_fn(tokenizer, args.max_chars)
    tokenized = {
        name: d.map(tok_fn, batched=True, remove_columns=d.column_names)
        for name, d in ds.items()
    }

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # transformers>=5.11's TensorBoardCallback reads TENSORBOARD_LOGGING_DIR
    # (not TrainingArguments.logging_dir, which it silently ignores) -- must
    # be set before Trainer(...) is constructed, since that's when the
    # callback reads it. training_kwargs["logging_dir"] is kept too, for
    # older versions that still honor it.
    os.environ["TENSORBOARD_LOGGING_DIR"] = args.tensorboard_dir

    training_args = TrainingArguments(**training_kwargs)

    # Pushes the best-N checkpoints (+ TensorBoard logs) to the Hub as
    # training progresses, not just at the end -- see checkpointing.py.
    topk_callback = TopKCheckpointCallback(
        metric_name=f"eval_{training_args.metric_for_best_model}",
        keep_best_n=args.keep_best_n,
        dev_dataset=ds["dev"],
        tokenizer=tokenizer,
        hub_repo_id=args.hub_repo_id if args.push_to_hub else None,
        hub_token=model_token if args.push_to_hub else None,
        hub_private=args.hub_private,
        tensorboard_dir=args.tensorboard_dir if args.push_to_hub else None,
        run_name=args.run_name,
        batch_size=training_args.per_device_eval_batch_size,
        greater_is_better=training_args.greater_is_better,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[topk_callback],
    )
    topk_callback.set_trainer(trainer)

    if not args.resume_from_checkpoint:
        # Eval the freshly-loaded (not-yet-fine-tuned) model before any
        # training steps, so we have a baseline to compare epochs against.
        # Trainer.evaluate() logs through the same compute_metrics() +
        # TensorBoardCallback path as every epoch, at global_step=0 -- it
        # shows up as the first point on the same eval_* curves in
        # TensorBoard, not a separate chart. Skipped when resuming: the
        # model here is the fresh --model weights, not the checkpoint
        # Trainer is about to load, so "initial" wouldn't mean anything.
        print("\n[info] Initial (pre-training) evaluation on dev -- logged at step 0:")
        initial_metrics = trainer.evaluate()
        for k, v in initial_metrics.items():
            print(f"  {k}: {v}")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Quick per-epoch-style metrics (also logged to TensorBoard), then the
    # full standard report (DER/WER/DER*/WER*, see metrics.py) on raw test
    # examples run through the just-trained (best) model.
    print("\n[info] Final test-set evaluation (quick):")
    # metric_key_prefix="test" (not the default "eval_") keeps this one-off
    # held-out report off the per-epoch dev curves in TensorBoard, and tells
    # TopKCheckpointCallback.on_evaluate not to treat it as a dev eval.
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    print("\n[info] Final test-set evaluation (full report):")
    examples = predict_examples(trainer.model, tokenizer, ds["test"],
                                 trainer.args.device, training_args.per_device_eval_batch_size)
    test_report = sequence_metrics(examples)
    test_report["baseline"] = sequence_metrics(baseline_examples(ds["test"]))
    print(format_report(test_report))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(test_report, f, ensure_ascii=False, indent=2)

    # trainer.model is already the BEST checkpoint here (load_best_model_at_end);
    # this is the lightweight (no optimizer state) save meant for inference.
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"\n[info] saved to {out_dir}")
    print(f"[info] tensorboard logs at {args.tensorboard_dir} "
          f"(run: tensorboard --logdir {args.tensorboard_dir})")

    if args.push_to_hub:
        upload_folder_to_hub(out_dir, args.hub_repo_id, "best", model_token, args.hub_private)
        print(f"[info] pushed best/ to https://huggingface.co/{args.hub_repo_id}")

        # The last epoch's FULL checkpoint (with optimizer/scheduler state),
        # kept on disk by topk_callback regardless of its score -- lets you
        # truly resume training (not just reload weights) if it ended early.
        if topk_callback.last_checkpoint_dir:
            upload_folder_to_hub(topk_callback.last_checkpoint_dir, args.hub_repo_id,
                                  "final", model_token, args.hub_private)
            print(f"[info] pushed final/ to https://huggingface.co/{args.hub_repo_id}")


if __name__ == "__main__":
    main()
