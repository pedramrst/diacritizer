"""
Fast end-to-end sanity check of the whole pipeline, without downloading real
CANINE weights or waiting for a real training run. Meant to be run right
after `make setup` on a fresh machine (e.g. a new Vast instance) to confirm
imports, dataset access, training, checkpoint retention, resume, and metrics
all actually work there before committing to a real (expensive) run.

Uses a tiny randomly-initialized CANINE config (not google/canine-s) and a
12-row slice of the real dataset, so it finishes in well under a minute on
CPU. Exits non-zero (with a normal Python traceback) on any failure.
"""

import os
import shutil
import tempfile

from dotenv import load_dotenv
from transformers import (
    CanineConfig,
    CanineForTokenClassification,
    CanineTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from transformers.trainer_utils import get_last_checkpoint

from diacritizer.preprocessing import LABEL2ID, ID2LABEL, NUM_LABELS, load_and_prepare
from train import build_tokenize_fn, compute_metrics
from diacritizer.checkpointing import TopKCheckpointCallback
from diacritizer.metrics import predict_examples, sequence_metrics, format_report

load_dotenv()

N_ROWS = 12
MAX_CHARS = 48
BATCH_SIZE = 4


def make_tiny_model():
    cfg = CanineConfig(
        num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID,
        hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=64, local_transformer_stride=4,
    )
    return CanineForTokenClassification(cfg)


def run_trainer(out_dir, tokenizer, tokenized, num_epochs, resume_from=None):
    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1_diacritics",
        greater_is_better=True,
        logging_steps=1,
        report_to=[],
        save_total_limit=None,
    )
    callback = TopKCheckpointCallback(
        metric_name="eval_macro_f1_diacritics",
        keep_best_n=2,
        dev_dataset=tokenized["_dev_raw"],
        tokenizer=tokenizer,
        hub_repo_id=None,
    )
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = Trainer(
        model=make_tiny_model(),
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[callback],
    )
    callback.set_trainer(trainer)
    trainer.train(resume_from_checkpoint=resume_from)
    return trainer, callback


def main():
    print("[smoke-test] 1/6 loading a small dataset slice ...")
    ds = load_and_prepare(max_chars=MAX_CHARS, token=os.environ.get("HF_DATASET_TOKEN"))
    ds = {k: v.select(range(N_ROWS)) for k, v in ds.items()}

    print("[smoke-test] 2/6 loading tokenizer + tokenizing ...")
    tokenizer = CanineTokenizer.from_pretrained("google/canine-s")
    tok_fn = build_tokenize_fn(tokenizer, MAX_CHARS)
    tokenized = {name: d.map(tok_fn, batched=True, remove_columns=d.column_names)
                 for name, d in ds.items()}
    tokenized["_dev_raw"] = ds["dev"]

    out_dir = tempfile.mkdtemp(prefix="diacritizer-smoketest-")
    try:
        print("[smoke-test] 3/6 training 2 epochs ...")
        trainer1, cb1 = run_trainer(out_dir, tokenizer, tokenized, num_epochs=2)
        first_step = trainer1.state.global_step
        assert os.path.isdir(os.path.join(out_dir, f"checkpoint-{first_step}")), \
            "expected a checkpoint dir after training"

        print("[smoke-test] 4/6 simulating an interruption + resume for 3 more epochs ...")
        last_ckpt = get_last_checkpoint(out_dir)
        assert last_ckpt is not None, "get_last_checkpoint found nothing to resume from"
        trainer2, cb2 = run_trainer(out_dir, tokenizer, tokenized, num_epochs=5,
                                     resume_from=last_ckpt)
        assert trainer2.state.global_step > first_step, \
            "resumed run did not continue past the first run's step count"

        print("[smoke-test] 5/6 checking checkpoint retention pruned old checkpoints ...")
        kept = sorted(os.listdir(out_dir))
        assert all(name.startswith("checkpoint-") for name in kept), \
            f"unexpected entries in output dir: {kept}"
        # keep_best_n=2 keeps the top-2-by-score PLUS the most recent checkpoint
        # (so "final" always survives even if it scores worse) -- that's up to
        # keep_best_n + 1 checkpoints, not exactly keep_best_n.
        assert len(kept) <= 3, f"expected at most keep_best_n(2) + 1 checkpoints, found {kept}"

        print("[smoke-test] 6/6 running the standard metrics report ...")
        examples = predict_examples(trainer2.model, tokenizer, ds["test"],
                                     trainer2.args.device, BATCH_SIZE)
        report = sequence_metrics(examples)
        print(format_report(report))
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    print("\n[smoke-test] PASSED")


if __name__ == "__main__":
    main()
