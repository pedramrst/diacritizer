# 🔤 Diacritizer

Persian diacritization (restoring Fatha/Damma/Kasra/Sokun/Shadda/Fathatan
marks on undiacritized text) via character-level token classification on
[CANINE](https://huggingface.co/google/canine-s). CANINE reads one Unicode
character at a time, so it doesn't need a subword tokenizer that would pack
several letters (and their diacritics) into a single opaque token — a
structural match for this task, similar to Arabic SOTA approaches (e.g. CATT).

✨ **At a glance**
- 🧠 Model: fine-tuned [`google/canine-s`](https://huggingface.co/google/canine-s)
- 🗂️ Dataset: [`avaeziaiteam/harakat-dataset`](https://huggingface.co/datasets/avaeziaiteam/harakat-dataset) (private)
- 🏷️ 10-class per-character label scheme (Fatha, Damma, Kasra, Sokun, Shadda + combinations)
- ☁️ Built for training on ephemeral Vast.ai GPU instances — checkpoint durability, auto-resume, no Docker

## 📚 Table of contents

- [🚀 Quickstart](#quickstart)
- [📁 Project layout](#project-layout)
- [🧰 Makefile](#makefile)
- [📦 Setup](#setup)
- [📊 Data](#data)
- [🧩 Config](#config)
- [🏃 Usage](#usage)
  - [📺 TensorBoard: one run per config, one shared folder](#tensorboard-one-run-per-config-one-shared-folder)
  - [🔁 Resuming an interrupted run](#resuming-an-interrupted-run)
  - [☁️ Checkpoint durability on the Hub](#checkpoint-durability-on-the-hub)
- [📈 Metrics](#metrics)
- [🧾 Report](#report)
- [🧪 Comparing configs (sweeps)](#comparing-configs-sweeps)

## 🚀 Quickstart

```bash
make setup                                    # create venv, install deps + the diacritizer package, seed .env
cp .env.example .env                          # 🔒 then fill in your real HF tokens
make smoke-test                               # ~30s sanity check, no real weights downloaded
make train                                    # fine-tune CANINE + live TensorBoard at http://localhost:6006
make report MODEL=./canine-fa-diacritizer     # manager-facing HTML report
```

That's the whole loop. Everything below explains what each piece does and
how to customize it.

## 📁 Project layout

<details>
<summary>Click to expand the full directory tree</summary>

```
diacritizer/                the importable library -- pip install -e .  (make setup does this)
  preprocessing.py           shared label scheme + dataset loading/splitting; also a CLI
                              (python -m diacritizer.preprocessing)
  metrics.py                 standard DER/WER/DER*/WER* + per-class report; also a CLI
                              (python -m diacritizer.metrics)
  checkpointing.py           best-N checkpoint retention + HF Hub push/download

scripts/                    entry points -- run as `python scripts/<name>.py`
  train.py                   fine-tunes CANINE on the diacritization task
  sweep.py                   runs train.py once per config in a directory, compares results
  diacritize.py               applies a trained model to raw text
  serve.py                     lightweight interactive web app for trying a trained model
  benchmark.py                speed / memory / device benchmarking + example outputs
  generate_report.py          renders the HTML report from metrics.json + benchmark.json (no ML deps)
  smoke_test.py               fast tiny-model pipeline check (no real weights) -- run after setup

templates/report_template.html   the report's static HTML/CSS/JS shell
notebooks/exploration.ipynb      ad-hoc dataset exploration
configs/                    example sweep config variants
config.yaml                  hyperparameters / paths (see below)
pyproject.toml                makes diacritizer/ installable (no dependencies of its own --
                              deps stay in requirements.txt)
Makefile                     high-level commands (see below)
requirements.txt              direct dependencies only, version-pinned (not a full `pip freeze`)
```

</details>

## 🧰 Makefile

No Docker — Vast already provides a CUDA-ready image, and this repo is a
single Python framework (torch/transformers), so a Makefile covers what's
needed without the overhead of maintaining a Dockerfile. `make help` lists
everything; the ones worth knowing:

```bash
make setup                                    # create venv, install deps, seed .env
make smoke-test                               # fast pipeline check on a new machine (no real weights)
make train                                    # train.py + TensorBoard together, CONFIG=... ARGS="..."
make sweep                                    # sweep.py + TensorBoard together, CONFIGS_DIR=... OUT_ROOT=... ARGS="..."
make diacritize MODEL=... TEXT="..."          # run inference
make serve SERVE_MODEL=... PORT=8000          # interactive web app for trying a model
make report MODEL=...                         # evaluate + benchmark MODEL, generate report.html
make clean                                    # remove __pycache__ / caches (checkpoints and runs/ are left alone on purpose)
```

💡 **Which config file does `make train` use?** `CONFIG` (default
`config.yaml`, the single file at the project root) — see the `train` recipe
in the `Makefile`, which runs `scripts/train.py --config $(CONFIG)`
(`train.py` also prints `using config: ...` on startup, so you can always
double check). Override with `make train CONFIG=configs/lr_5e-5.yaml` to
train a single specific variant. `make sweep`, by contrast, always uses
*every* `*.yaml` file under `CONFIGS_DIR` (default `configs/`) — one
`train.py` run per file — since sweeping means comparing more than one.

`make train` and `make sweep` start TensorBoard in the background (bound to
`0.0.0.0` so it's reachable if you're forwarding the port from a Vast
instance) pointed at `TB_LOGDIR`/`OUT_ROOT/runs`, run the Python command in
the foreground, and kill TensorBoard automatically when that command exits
or is interrupted (Ctrl+C) — so you get live metrics at
`http://localhost:6006` without a second terminal, and no orphaned
TensorBoard process left running afterward.

`TB_PORT` (default `6006`) can be set three ways, in this priority order:

```bash
make train TB_PORT=7000        # 1. one-off CLI override -- always wins
TB_PORT=7000 make train        # 2. one-off shell env var
# or uncomment TB_PORT=... in .env for a persistent default (make setup seeds
# this file; the Makefile reads it via -include .env)
```

⚠️ Once `.env` sets `TB_PORT`, it takes priority over a plain shell env var
(method 2) — that's just how Make resolves variables from an included file
vs. the environment. Only the explicit `make train TB_PORT=...` CLI form
(method 1) is guaranteed to always win. `TB_LOGDIR`/`CONFIGS_DIR`/`OUT_ROOT`
etc. all follow the same three-way override rules.

> 💡 Since checkpoints/`runs/`/`sweeps/` represent real trained-model work
> (expensive to regenerate), `make clean` intentionally does not touch them —
> only `__pycache__` and other safe, regenerable caches.

## 📦 Setup

Clone the repo (SSH, if you have a key on GitHub — otherwise use the HTTPS
form) and `cd` into it:

```bash
git clone git@github.com:pedramrst/diacritizer.git
# or: git clone https://github.com/pedramrst/diacritizer.git
cd diacritizer
```

Then set up the environment:

```bash
make setup
```

<details>
<summary>What that wraps, if you'd rather run it by hand</summary>

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .          # makes `diacritizer` importable from anywhere in the venv
```

</details>

Either way, copy `.env.example` to `.env` and fill in real tokens (`make
setup` does this for you automatically if `.env` doesn't already exist):

```bash
cp .env.example .env
```

```
HF_DATASET_TOKEN=...   # read access to the private dataset repo (dataset account)
HF_MODEL_TOKEN=...      # write access to the model repo (a DIFFERENT HF account)
```

🔒 `HF_DATASET_TOKEN` is only needed to download
[`avaeziaiteam/harakat-dataset`](https://huggingface.co/datasets/avaeziaiteam/harakat-dataset)
(private). `HF_MODEL_TOKEN` is only needed when training with `--push_to_hub`,
since the trained model is published under a separate HF account from the
dataset.

## 📊 Data

Two columns: `raw` (undiacritized) and `harakat` (diacritized). The `raw`
column is **not** trusted for character alignment (about 6.6% of rows insert
an extra Ezafe `ی` that isn't present in `raw`) — the clean base-character
sequence is always derived by stripping diacritics out of `harakat` itself,
which guarantees a perfect 1:1 mapping between characters and labels. Rows
are deduped on `raw` (2 exact duplicates in the current dataset) so the same
sentence can never end up split across train/dev/test.

Label scheme (`diacritizer/preprocessing.py`), one label per base character:

```
NONE, FATHA, DAMMA, KASRA, SOKUN, SHADDA,
SHADDA_FATHA, SHADDA_DAMMA, SHADDA_KASRA, FATHATAN
```

Shadda (gemination) stacks with a vowel on the same character (e.g. `چِّ`),
so it gets its own compound labels rather than being dropped.
`SUPERSCRIPT_ALEF` (9 occurrences total) is excluded as a label — too rare
to learn or evaluate — and folded into `NONE`.

## 🧩 Config

All fine-tuning hyperparameters live in `config.yaml`. **`training:` is passed
to `transformers.TrainingArguments` almost verbatim** — its keys are
`TrainingArguments`' own field names (`num_train_epochs`,
`per_device_train_batch_size`, `output_dir`, ...), so anything valid there —
`gradient_accumulation_steps`, `lr_scheduler_type`, `bf16`,
`dataloader_num_workers`, whatever — can just be added to `training:` with
**no code change needed**.

<details>
<summary>Click to expand config.yaml</summary>

```yaml
data:
  hf_repo: avaeziaiteam/harakat-dataset
  raw_col: raw
  diac_col: harakat
  max_chars: 512
  # Each is an independent fraction of the FULL dataset (not of each other) --
  # 0.05 + 0.05 below means 90/5/5 train/dev/test directly.
  test_size: 0.05
  dev_size: 0.05

model:
  name: google/canine-s

training:
  output_dir: ./canine-fa-diacritizer
  num_train_epochs: 8
  learning_rate: 5.0e-5
  per_device_train_batch_size: 16
  per_device_eval_batch_size: 16
  weight_decay: 0.01
  warmup_ratio: 0.1
  lr_scheduler_type: cosine
  optim: adamw_torch
  max_grad_norm: 1.0
  seed: 42
  logging_steps: 50
  train_sampling_strategy: group_by_length  # batches similar-length sentences -- less padding waste
  dataloader_num_workers: 2
  # add any other transformers.TrainingArguments field here too

logging:
  tensorboard_dir: ./runs

hub:
  push_to_hub: false
  repo_id: your-username/canine-fa-diacritizer
  private: true

checkpoints:
  keep_best_n: 3
```

</details>

⚠️ A handful of `training:` keys are filled in as defaults this pipeline
relies on — `eval_strategy`/`save_strategy` ("epoch"),
`metric_for_best_model` ("macro_f1_diacritics"), `load_best_model_at_end`,
`report_to` (`["tensorboard"]`) — and `logging_dir` is always set from
`--tensorboard_dir`/`logging.tensorboard_dir`, not `training:`, since Hub
push and run-name nesting both key off that same path. Config/CLI values
still win if you set any of these yourself, but checkpoint retention,
auto-resume, and TensorBoard/Hub push all assume they match what those
defaults set up — overriding them needs the corresponding code to agree, not
just the config. Everything else in `training:` is safe to change freely.

💡 `bf16`/`fp16` aren't in the example above because they're auto-detected —
`bf16` on GPUs that support it (Ampere+: A100, RTX 30xx/40xx), `fp16` on
older CUDA GPUs, neither on CPU/MPS. bf16 is preferred where available since
it avoids fp16's loss-scaling/NaN risk, which matters more here than usual
given how imbalanced the label distribution is (a few rare classes produce
small-magnitude gradients). Set either explicitly in `training:` to force it.

`--epochs`/`--lr`/`--batch_size`/`--seed`/`--out` are optional CLI shortcuts
for the fields people override most often — only applied if you actually
pass them, e.g. `python scripts/train.py --epochs 12 --batch_size 32`.
Anything else is config-only: edit `training:` directly.

## 🏃 Usage

Explore the dataset (label distribution, edge cases):

```bash
python -m diacritizer.preprocessing
```

Train (or `make train CONFIG=config.yaml` to also get TensorBoard alongside it):

```bash
python scripts/train.py --config config.yaml
```

This does a train/dev/test split (ratios from config) and evaluates the full
DER/WER/DER\*/WER\* report plus per-class F1 on dev each epoch (see
[📈 Metrics](#metrics) for what these mean). Before the first training step,
it also evaluates the freshly-loaded (not yet fine-tuned) model once, so you
have a baseline — this shows up as step 0 on the same TensorBoard curves as
every epoch after it, not a separate chart (skipped automatically when
resuming, since there the loaded weights aren't really "initial"). The best
checkpoint (by `macro_f1_diacritics`) is kept via `load_best_model_at_end`,
saved to `training.output_dir`, and evaluated once more on the held-out test
set.

### 📺 TensorBoard: one run per config, one shared folder

<details>
<summary>Click to expand</summary>

**What gets logged:** every `logging_steps` (default 50), `loss`/`grad_norm`/
`learning_rate`; every epoch (plus once before training starts, at step 0),
everything `compute_metrics()` returns as `eval_*` — `eval_DER`,
`eval_char_accuracy`, `eval_f1_<CLASS>` for all 10 classes, and
`eval_macro_f1_diacritics` (the model-selection metric) — **plus** the full
`eval_WER`, `eval_DER*`, `eval_WER*`, `eval_macro_f1_diacritics*` from
`diacritizer.metrics.sequence_metrics()`, all against **dev**, all on the
same step so every eval_\* scalar lines up on one x-axis. This costs a second
forward pass over dev each epoch (dev is small, so this is normally
negligible) — `TopKCheckpointCallback` (see [☁️ Checkpoint durability on the
Hub](#checkpoint-durability-on-the-hub)) computes it once per eval and
reuses the same result for the per-checkpoint Hub `metrics.json`, rather than
computing it twice. The baseline comparison from [📈 Metrics](#metrics) is
*not* part of this — that's only computed once, on **test**, at the very end.

⚠️ transformers' `TensorBoardCallback` only reads the `TENSORBOARD_LOGGING_DIR`
environment variable (it silently ignores `TrainingArguments.logging_dir` in
current versions) — `train.py` sets that env var itself before constructing
the `Trainer`, so `--tensorboard_dir`/`TB_LOGDIR` are honored either way; you
don't need to set that env var yourself.

Each run's events go to `<tensorboard_dir>/<run_name>/`, not straight into
`tensorboard_dir` — `run_name` defaults to the `--config` filename (e.g.
`configs/lr_5e-5.yaml` → `lr_5e-5`), or set it explicitly with `--run_name`.
So every config you train shares the same `tensorboard_dir` root, but still
shows up as its own separate, comparable line in the UI — point TensorBoard
at the shared parent, not at any one run's subfolder:

```bash
tensorboard --logdir runs   # shows every run trained so far, side by side
```

This is standard TensorBoard behavior (any subdirectory of `--logdir` is
treated as its own run) — it's not something specific to this repo, just
worth knowing so you don't accidentally point `--logdir` at one run's folder
and only see that one.

</details>

### 🔁 Resuming an interrupted run

<details>
<summary>Click to expand</summary>

If `scripts/train.py` gets killed (crash, OOM, a Vast instance reboot that
keeps its disk) and you rerun the exact same command, it resumes
automatically: it checks `--out` for an existing checkpoint and continues
from it rather than starting over. Pass `--no_auto_resume` to force a clean
run instead (e.g. if `--out` has stale checkpoints from a different config).
This also works after a sweep run is interrupted — just rerun the same
`scripts/sweep.py` command.

</details>

### ☁️ Checkpoint durability on the Hub

<details>
<summary>Click to expand</summary>

Push checkpoints to the Hub as training progresses, not just at the end
(uses `HF_MODEL_TOKEN`) — important since Vast/spot instances aren't
persistent:

```bash
python scripts/train.py --config config.yaml --push_to_hub --hub_repo_id your-username/canine-fa-diacritizer
```

This creates/updates `your-username/canine-fa-diacritizer` on the Hub with:

```
checkpoint-<step>/           the keep_best_n checkpoints seen so far, by dev macro_f1_diacritics
                              (full Trainer checkpoint incl. optimizer/scheduler state -- resumable)
                              + metrics.json (full DER/WER/DER*/WER* report on dev)
final/                        the LAST epoch's full checkpoint, always kept regardless of its
                              score, pushed once training ends -- also resumable
best/                         trainer.save_model() output for the best-by-metric model
                              (no optimizer state -- for inference / as a --model value,
                              not for --resume_from_checkpoint) + metrics.json (test set)
tensorboard_logs/<run_name>/  re-pushed after every epoch, so logs are never more than
                              one epoch stale if the instance disappears mid-run
```

⚠️ If you push more than one config's runs to the *same* `--hub_repo_id`,
`checkpoint-<step>/`/`final/`/`best/` will collide across configs since
they're not namespaced by run — use a distinct `--hub_repo_id` per config,
or only push the winner after a sweep, if you need more than one config's
checkpoints preserved on the Hub at once. `tensorboard_logs/<run_name>/` is
safe to share, since it is namespaced.

If the instance is lost mid-run, resume from a Hub checkpoint (downloads it
first):

```bash
python scripts/train.py --config config.yaml --push_to_hub --hub_repo_id your-username/canine-fa-diacritizer \
    --resume_from_hub_repo your-username/canine-fa-diacritizer --resume_from_hub_subfolder final
```

⚠️ `--resume_from_hub_subfolder` accepts `final`, `checkpoint-<step>`, but
**not** `best` — that folder has no optimizer/scheduler state to resume
from. If training instead finished normally and you just want to keep
fine-tuning the winning config longer, no special resume flag is needed —
just point `--model` at that config's output dir (or its `best/` on the Hub)
and rerun with a higher `--epochs`.

</details>

Run inference:

```bash
python scripts/diacritize.py --model ./canine-fa-diacritizer --text "کتاب من"
python scripts/diacritize.py --model your-username/canine-fa-diacritizer --text "..."
```

To try a model interactively instead of one `--text` at a time, `make serve`
(or `python scripts/serve.py --model ...`) starts a local web app at
`http://127.0.0.1:8000` with a textarea for raw Persian text and the
diacritized result rendered back. It's a stdlib-only `http.server` app (no
new dependencies) that loads the model once and reuses the same inference
code as `diacritize.py`.

```bash
make serve                                          # defaults to the public Hub repo (PedramR/canine-fa-diacritizer)
make serve SERVE_MODEL=./canine-fa-diacritizer       # or a local checkpoint
make serve SERVE_MODEL=your-username/canine-fa-diacritizer PORT=8080
```

## 📈 Metrics

`diacritizer/metrics.py` reports the standard set used in the Arabic/Persian
diacritization literature (Fadel et al. 2019; CATT):

- **DER** — Diacritic Error Rate: fraction of characters with the wrong label.
- **WER** — Word Error Rate: fraction of words with ≥1 wrong character.
- **DER\*** / **WER\*** — the same, ignoring the word-final character of
  every word. In Persian that position is usually the Ezafe kasra, which
  depends on syntax beyond the sentence's characters, so it's reported
  separately rather than mixed into the headline number.
- Per-class precision/recall/F1, and macro-F1 (excluding `NONE`).
- A **baseline** comparison (always predict "no diacritic") nested under
  `report["baseline"]`, computed on the same split — context for whether the
  headline numbers are actually good, not just non-zero.

`scripts/train.py` prints the full report on the test set automatically at
the end of training, and always writes it to `<output_dir>/metrics.json`. To
evaluate any saved checkpoint (local or on the Hub) later, optionally saving
JSON too:

```bash
python -m diacritizer.metrics --model ./canine-fa-diacritizer --split test --out metrics.json
```

## 🧾 Report

`make report MODEL=./canine-fa-diacritizer` (or the three scripts below run
by hand) produces a manager-facing HTML report — accuracy vs. a baseline,
per-diacritic F1, throughput/latency/memory, device info, and real
input/output examples from the test set.

✅ **No new dependencies.** `scripts/benchmark.py` gets device/memory info
from `torch` (already required) plus the Python standard library
(`platform`, `resource`, `os.sysconf`) — no `psutil` or anything else.
`scripts/generate_report.py` doesn't even import torch/transformers/datasets;
it only reads two JSON files and writes HTML.

The pipeline is three independent, rerunnable steps (`make report` just runs
them in order):

```bash
python -m diacritizer.metrics --model ./canine-fa-diacritizer --split test --out metrics.json
python scripts/benchmark.py --model ./canine-fa-diacritizer --out benchmark.json
python scripts/generate_report.py --metrics metrics.json --benchmark benchmark.json --out report.html
```

<details>
<summary>What each step actually measures</summary>

- **`diacritizer.metrics --out`** — accuracy + baseline comparison (above).
- **`scripts/benchmark.py`** — cold-start time; model size (parameter count +
  disk size, when `--model` is a local dir); single-sentence latency
  (p50/p95/p99); throughput by batch size (`--batch_sizes 1,8,16,32,64`);
  peak memory (GPU/MPS via `torch`, CPU via peak RSS); a CPU-only comparison
  pass when the primary device isn't CPU; and `--n_examples` real
  input/output/reference triples from the test set (not cherry-picked).
- **`scripts/generate_report.py`** — fills `templates/report_template.html`
  with both JSON files' contents as one embedded data blob; the page's own JS
  renders the charts, stat tiles, and example cards from it client-side.
  Missing pieces (e.g. no local disk size for a Hub-hosted model) render as
  "—" rather than crashing.

</details>

💡 `report.html`/`metrics.json`/`benchmark.json` are gitignored — they're
generated per model, not source. Pass `--mock` to `scripts/generate_report.py`
to show a "sample report" banner (useful when sharing a template preview
rather than a real result).

## 🧪 Comparing configs (sweeps)

To try several hyperparameter configs and compare them, put one YAML file per
variant under `configs/` (two examples are included, differing only in
`learning_rate`) and run:

```bash
python scripts/sweep.py --configs configs/ --out_root ./sweeps/run1
# or, with TensorBoard alongside it:
make sweep OUT_ROOT=./sweeps/run1
```

This runs `scripts/train.py` once per config file, each writing its
checkpoints to its own `sweeps/run1/<config filename>/` (so they never
collide), then prints a comparison table sorted by `macro_f1_diacritics` on
the test set. Pass `--extra_args "--push_to_hub"` to also push every trial's
checkpoints to the Hub (see the caveat in [☁️ Checkpoint durability on the
Hub](#checkpoint-durability-on-the-hub) about using distinct `--hub_repo_id`s
per config if you do).

💡 Sweep configs are typically short (few epochs) to compare cheaply — once
you've picked a winner, keep fine-tuning it longer by pointing `--model` at
its output dir (or Hub `best/`) with a higher `--epochs`; no separate resume
mechanism is needed for that (resume is only for continuing an interrupted
run, see above).

All configs' TensorBoard logs share one root, `sweeps/run1/runs/`, each
under its own `<config filename>/` subfolder (`scripts/sweep.py` leaves
`--run_name` at its default). View every trial side by side:

```bash
tensorboard --logdir ./sweeps/run1/runs
```

If a sweep run is interrupted partway through, just rerun the same
`scripts/sweep.py` command — each config's `scripts/train.py` call
auto-resumes from its own `--out` if a checkpoint is already there.
