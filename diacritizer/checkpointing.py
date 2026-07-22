"""
Checkpoint retention + HF Hub TensorBoard durability for training on ephemeral
instances (e.g. Vast.ai, which can disappear between sessions).

TopKCheckpointCallback hooks into Trainer's own per-epoch checkpoints
(save_strategy="epoch" already writes a full checkpoint-<step>/ dir with
model + optimizer + scheduler + trainer_state, everything resume_from_checkpoint
needs). It:
  - on every dev evaluation (each epoch, plus train.py's initial pre-training
    eval), computes the full DER/WER/DER*/WER* report (see diacritizer.metrics)
    -- not just the flattened DER/F1 Trainer's own compute_metrics returns --
    and logs the extra keys (WER, DER*, WER*, ...) to TensorBoard alongside
    the regular eval_* scalars, at the same step. Skipped for train.py's final
    test-set report (metric_key_prefix="test"), which stays a one-off, held-out
    evaluation rather than something plotted per epoch.
  - keeps only the best-N-by-metric checkpoints on local disk, PLUS whichever
    is most recent (so "final" always survives even if it didn't score well),
    deleting everything else to save disk space.
  - re-pushes the TensorBoard log directory (tensorboard_logs/<run_name>/) to
    the HF Hub on every save, so logs are never more than one epoch stale if
    the instance dies.

Checkpoints themselves are NOT pushed to the Hub (only kept locally) -- the
Hub repo is meant to stay a standard, single-model repo (config.json,
model.safetensors, ... at the repo root, loadable via plain
`from_pretrained(repo_id)`), with tensorboard_logs/ alongside it. train.py
pushes the best-at-end model (trainer.save_model() output, no optimizer
state, meant for inference) to the repo root at the very end of training. If
a training instance is lost mid-run, resume from whatever checkpoint
survived on that instance's local disk (--resume_from_checkpoint) -- there
is no Hub-based fallback.
"""

import os
import shutil

from huggingface_hub import HfApi
from transformers import TrainerCallback

from diacritizer.metrics import predict_examples, sequence_metrics


def upload_folder_to_hub(local_dir, repo_id, path_in_repo, token, private=True,
                         ignore_patterns=None):
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        token=token,
        commit_message=f"Update {path_in_repo or '(root)'}",
        ignore_patterns=ignore_patterns,
    )


class TopKCheckpointCallback(TrainerCallback):
    def __init__(self, metric_name, keep_best_n, dev_dataset, tokenizer,
                 hub_repo_id=None, hub_token=None, hub_private=True,
                 tensorboard_dir=None, run_name="run", batch_size=16,
                 greater_is_better=True):
        self.metric_name = metric_name  # e.g. "eval_macro_f1_diacritics"
        self.keep_best_n = keep_best_n
        self.dev_dataset = dev_dataset
        self.tokenizer = tokenizer
        self.hub_repo_id = hub_repo_id
        self.hub_token = hub_token
        self.hub_private = hub_private
        self.tensorboard_dir = tensorboard_dir
        self.run_name = run_name
        self.batch_size = batch_size
        self.greater_is_better = greater_is_better

        self.trainer = None  # set via set_trainer() right after Trainer(...)
        self._scored = []  # [(score, step, local_dir), ...], sorted best-first
        self._last_full_report = None  # cache: sequence_metrics() dict from the latest dev eval

    def set_trainer(self, trainer):
        self.trainer = trainer

    def _latest_metric(self, state):
        for entry in reversed(state.log_history):
            if self.metric_name in entry:
                return entry[self.metric_name]
        return None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        # metrics uses whatever metric_key_prefix the evaluate() call used
        # (default "eval_" for per-epoch dev evals and train.py's initial
        # pre-training eval; "test_" for the final held-out report) -- only
        # recompute/log the full report for the former.
        if self.trainer is None:
            return control
        if metrics is not None and not any(k.startswith("eval_") for k in metrics):
            return control
        try:
            examples = predict_examples(
                self.trainer.model, self.tokenizer, self.dev_dataset,
                self.trainer.args.device, self.batch_size,
            )
            self._last_full_report = sequence_metrics(examples)
        except Exception as e:
            print(f"[warn] could not compute full dev report: {e}")
            return control

        extra = {
            f"eval_{k}": v for k, v in self._last_full_report.items()
            if k in ("WER", "DER*", "WER*", "char_accuracy*",
                      "macro_f1_diacritics*", "n_words", "n_words_with_non_final_chars")
        }
        if extra:
            self.trainer.log(extra)
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        # On a fresh run state.log_history is empty, so this is a no-op. On a
        # RESUMED run (--resume_from_checkpoint), Trainer has already restored
        # state.log_history from the checkpoint's trainer_state.json before
        # this fires -- rebuild our best-N bookkeeping from it, matching each
        # past eval entry's step to a still-present checkpoint-<step> dir.
        # Without this, a resumed run would "forget" earlier top-N
        # checkpoints and could prune them away as if they were never scored.
        if self._scored or not state.log_history:
            return control
        for entry in state.log_history:
            if self.metric_name in entry and "step" in entry:
                step = entry["step"]
                d = os.path.join(args.output_dir, f"checkpoint-{step}")
                if os.path.isdir(d):
                    self._scored.append((entry[self.metric_name], step, d))
        self._scored.sort(key=lambda x: x[0], reverse=self.greater_is_better)
        self._scored = self._scored[:self.keep_best_n]
        if self._scored:
            print(f"[info] resumed: recovered {len(self._scored)} previously-scored "
                  f"checkpoint(s) from trainer_state.json")
        return control

    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            return control

        score = self._latest_metric(state)
        if score is not None:
            self._scored.append((score, state.global_step, ckpt_dir))
            self._scored.sort(key=lambda x: x[0], reverse=self.greater_is_better)

        keep_dirs = {d for _, _, d in self._scored[:self.keep_best_n]}
        keep_dirs.add(ckpt_dir)  # always keep the most recent checkpoint

        for name in os.listdir(args.output_dir):
            d = os.path.join(args.output_dir, name)
            if name.startswith("checkpoint-") and d not in keep_dirs and os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        self._scored = [e for e in self._scored if e[2] in keep_dirs]

        if self.hub_repo_id and self.tensorboard_dir and os.path.isdir(self.tensorboard_dir):
            upload_folder_to_hub(self.tensorboard_dir, self.hub_repo_id,
                                  f"tensorboard_logs/{self.run_name}",
                                  self.hub_token, self.hub_private)

        return control
