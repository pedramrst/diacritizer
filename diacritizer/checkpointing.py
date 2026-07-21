"""
Checkpoint retention + HF Hub durability for training on ephemeral instances
(e.g. Vast.ai, which can disappear between sessions).

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
  - pushes every checkpoint it decides to keep to the HF Hub, as
    checkpoint-<step>/, alongside a metrics.json with that same full dev
    report (reused from the cache above, not recomputed).
  - re-pushes the TensorBoard log directory on every save, so logs are never
    more than one epoch stale if the instance dies.

train.py additionally pushes the very last checkpoint as final/ and the
best-at-end model (trainer.save_model() output, no optimizer state, meant for
inference) as best/.
"""

import json
import os
import shutil

from huggingface_hub import HfApi, snapshot_download
from transformers import TrainerCallback

from diacritizer.metrics import predict_examples, sequence_metrics


def upload_folder_to_hub(local_dir, repo_id, path_in_repo, token, private=True):
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        path_in_repo=path_in_repo,
        token=token,
        commit_message=f"Update {path_in_repo}",
    )


def download_checkpoint_from_hub(repo_id, path_in_repo, token, local_dir):
    """Pull a checkpoint-<step>/, final/, or best/ folder back down (e.g. after
    losing the training instance) and return its local path, ready to pass to
    --resume_from_checkpoint."""
    snapshot_download(
        repo_id=repo_id,
        token=token,
        local_dir=local_dir,
        allow_patterns=f"{path_in_repo}/*",
    )
    return os.path.join(local_dir, path_in_repo)


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
        self.last_checkpoint_dir = None
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
        self.last_checkpoint_dir = ckpt_dir

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

        is_top_n = ckpt_dir in {d for _, _, d in self._scored[:self.keep_best_n]}
        if self.hub_repo_id and score is not None and is_top_n:
            self._push_checkpoint(ckpt_dir, f"checkpoint-{state.global_step}", score)

        if self.hub_repo_id and self.tensorboard_dir and os.path.isdir(self.tensorboard_dir):
            upload_folder_to_hub(self.tensorboard_dir, self.hub_repo_id,
                                  f"tensorboard_logs/{self.run_name}",
                                  self.hub_token, self.hub_private)

        return control

    def _push_checkpoint(self, local_dir, path_in_repo, score):
        # on_evaluate always runs immediately before on_save in Trainer's own
        # per-epoch loop, so the cache is fresh -- no need to recompute here.
        if self._last_full_report is not None:
            report = {self.metric_name: score, **self._last_full_report}
            with open(os.path.join(local_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
        else:
            print(f"[warn] no cached dev report available for {path_in_repo}; metrics.json omitted")

        upload_folder_to_hub(local_dir, self.hub_repo_id, path_in_repo,
                              self.hub_token, self.hub_private)
        print(f"[info] pushed {path_in_repo} (score={score:.4f}) to "
              f"https://huggingface.co/{self.hub_repo_id}/tree/main/{path_in_repo}")
