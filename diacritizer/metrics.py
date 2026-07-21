"""
Standard evaluation metrics for Persian diacritization, following the
conventions used in the Arabic diacritization literature (Fadel et al. 2019;
CATT):

    DER   - Diacritic Error Rate: fraction of characters given the wrong label.
    WER   - Word Error Rate: fraction of words with >=1 wrong character label.
    DER*  - DER computed ignoring the word-FINAL character of every word.
    WER*  - WER computed ignoring the word-FINAL character of every word.

DER*/WER* exist because the word-final diacritic is disproportionately hard:
in Arabic it's the i'rab case ending; in Persian it's most often the Ezafe
kasra (the linking vowel between two words), which -- unlike most
diacritics -- depends on syntax/semantics beyond the sentence's characters.
Reporting DER*/WER* alongside DER/WER isolates how much error is concentrated
at that single position vs. spread across the rest of the text.

Also reports per-class precision/recall/F1 and macro-F1 (excluding NONE,
which dominates the label distribution and would otherwise wash out the
signal from the actual diacritic classes).

Usage as a library: `sequence_metrics(examples)` where `examples` is a list
of {"chars": [...], "true": [...], "pred": [...]} dicts (one per sentence,
`chars` including whitespace/ZWNJ, `true`/`pred` are label ids aligned to
`chars`). train.py calls this on the test set after training.

Usage standalone: evaluate any saved checkpoint (local dir or HF Hub repo id)
against any split of the dataset:
    python metrics.py --model ./canine-fa-diacritizer --split test
"""

import numpy as np

from diacritizer.preprocessing import ID2LABEL

SEPARATORS = {" ", "‌"}  # space, ZWNJ -- both mark a word boundary


def _word_final_mask(chars):
    n = len(chars)
    mask = [False] * n
    for i, ch in enumerate(chars):
        if ch in SEPARATORS:
            continue
        mask[i] = (i == n - 1) or (chars[i + 1] in SEPARATORS)
    return mask


def char_metrics(y_true, y_pred):
    """Per-character DER/accuracy + per-class precision/recall/F1 and
    macro-F1 (excluding NONE) over flat label-id sequences."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    total = len(y_true)
    der = float(np.mean(y_true != y_pred)) if total else 0.0

    metrics = {"DER": der, "char_accuracy": 1.0 - der, "n_chars": total}
    f1s = []
    for cls_id, cls_name in ID2LABEL.items():
        tp = int(np.sum((y_pred == cls_id) & (y_true == cls_id)))
        fp = int(np.sum((y_pred == cls_id) & (y_true != cls_id)))
        fn = int(np.sum((y_pred != cls_id) & (y_true == cls_id)))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        metrics[f"precision_{cls_name}"] = prec
        metrics[f"recall_{cls_name}"] = rec
        metrics[f"f1_{cls_name}"] = f1
        if cls_name != "NONE":
            f1s.append(f1)
    metrics["macro_f1_diacritics"] = float(np.mean(f1s)) if f1s else 0.0
    return metrics


def sequence_metrics(examples):
    """
    examples: list of {"chars": [...], "true": [...], "pred": [...]}, one
    entry per sentence, all three lists the same length (chars includes
    separators; true/pred are label ids).

    Returns a dict with DER, WER, DER*, WER*, and the per-class report.
    """
    all_true, all_pred = [], []
    all_true_ni, all_pred_ni = [], []  # "non-word-final" subset -> DER*

    n_words = n_word_errors = 0
    n_words_ni = n_word_errors_ni = 0

    def flush_word(word_true, word_pred, word_final):
        nonlocal n_words, n_word_errors, n_words_ni, n_word_errors_ni
        if not word_true:
            return
        n_words += 1
        if word_true != word_pred:
            n_word_errors += 1
        ni_true = [t for t, f in zip(word_true, word_final) if not f]
        ni_pred = [p for p, f in zip(word_pred, word_final) if not f]
        if ni_true:
            n_words_ni += 1
            if ni_true != ni_pred:
                n_word_errors_ni += 1

    for ex in examples:
        chars, true, pred = ex["chars"], ex["true"], ex["pred"]
        final_mask = _word_final_mask(chars)

        word_true, word_pred, word_final = [], [], []
        for ch, t, p, is_final in zip(chars, true, pred, final_mask):
            if ch in SEPARATORS:
                flush_word(word_true, word_pred, word_final)
                word_true, word_pred, word_final = [], [], []
                continue
            word_true.append(t)
            word_pred.append(p)
            word_final.append(is_final)
            all_true.append(t)
            all_pred.append(p)
            if not is_final:
                all_true_ni.append(t)
                all_pred_ni.append(p)
        flush_word(word_true, word_pred, word_final)

    metrics = char_metrics(all_true, all_pred)
    metrics["WER"] = n_word_errors / n_words if n_words else 0.0
    metrics["n_words"] = n_words

    star = char_metrics(all_true_ni, all_pred_ni)
    metrics["DER*"] = star["DER"]
    metrics["char_accuracy*"] = star["char_accuracy"]
    metrics["macro_f1_diacritics*"] = star["macro_f1_diacritics"]
    metrics["WER*"] = n_word_errors_ni / n_words_ni if n_words_ni else 0.0
    metrics["n_words_with_non_final_chars"] = n_words_ni

    return metrics


def baseline_examples(dataset):
    """
    The "predict no diacritic at all" baseline, for the SAME dataset split,
    reusing sequence_metrics unchanged (pred is just every position labeled
    NONE) -- gives a naive reference point: how much is the model actually
    buying you over doing nothing.
    """
    from diacritizer.preprocessing import LABEL2ID
    none_id = LABEL2ID["NONE"]
    examples = []
    for chars, labels in zip(dataset["chars"], dataset["labels"]):
        examples.append({"chars": chars, "true": labels, "pred": [none_id] * len(labels)})
    return examples


def predict_examples(model, tokenizer, dataset, device, batch_size=16):
    """
    Run `model` over every row of `dataset` (must have "chars"/"labels"
    columns, i.e. an UNTOKENIZED split from preprocessing.load_and_prepare)
    and return the {"chars", "true", "pred"} list sequence_metrics expects.
    """
    import torch

    examples = []
    model.eval()
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start:start + batch_size]
        texts = ["".join(chars) for chars in batch["chars"]]
        enc = tokenizer(texts, padding=True, truncation=True,
                         max_length=2048, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**enc).logits
        preds = torch.argmax(logits, dim=-1).cpu().tolist()

        for chars, labels, pred_row in zip(batch["chars"], batch["labels"], preds):
            # Drop [CLS] (index 0); CANINE is one-code-point-per-char so the
            # remaining predictions align 1:1 with `chars`.
            pred = pred_row[1:1 + len(chars)]
            examples.append({"chars": chars, "true": labels, "pred": pred})
    return examples


def format_report(metrics):
    lines = [
        f"n_chars={metrics['n_chars']}  n_words={metrics['n_words']}",
        "",
        f"  DER   (char, all positions)      : {metrics['DER']:.4f}",
        f"  WER   (word, all positions)      : {metrics['WER']:.4f}",
        f"  DER*  (char, excl. word-final)   : {metrics['DER*']:.4f}",
        f"  WER*  (word, excl. word-final)   : {metrics['WER*']:.4f}",
        f"  macro_f1_diacritics  (excl. NONE): {metrics['macro_f1_diacritics']:.4f}",
        f"  macro_f1_diacritics* (excl. NONE): {metrics['macro_f1_diacritics*']:.4f}",
        "",
        "  per-class:",
    ]
    for name in ID2LABEL.values():
        lines.append(
            f"    {name:>16}: precision={metrics[f'precision_{name}']:.4f} "
            f"recall={metrics[f'recall_{name}']:.4f} f1={metrics[f'f1_{name}']:.4f}"
        )
    return "\n".join(lines)


def main():
    import argparse
    import os

    import torch
    import yaml
    from dotenv import load_dotenv
    from transformers import CanineTokenizer, CanineForTokenClassification

    from diacritizer.preprocessing import load_and_prepare

    load_dotenv()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="local checkpoint dir, or an HF Hub repo id")
    ap.add_argument("--config", default="config.yaml",
                    help="used for dataset/split settings (--data, columns, split sizes)")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default=None,
                    help="if set, also write the report (incl. a nested 'baseline' "
                         "comparison) as JSON to this path")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_cfg = cfg["data"]

    ds = load_and_prepare(
        data_cfg["hf_repo"],
        raw_col=data_cfg["raw_col"],
        diac_col=data_cfg["diac_col"],
        max_chars=data_cfg["max_chars"],
        test_size=data_cfg["test_size"],
        dev_size=data_cfg["dev_size"],
        token=os.environ.get("HF_DATASET_TOKEN"),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hub_token = os.environ.get("HF_MODEL_TOKEN")
    tokenizer = CanineTokenizer.from_pretrained(args.model, token=hub_token)
    model = CanineForTokenClassification.from_pretrained(
        args.model, token=hub_token
    ).to(device)

    examples = predict_examples(model, tokenizer, ds[args.split], device, args.batch_size)
    report = sequence_metrics(examples)
    report["baseline"] = sequence_metrics(baseline_examples(ds[args.split]))

    print(f"[info] {args.split} set ({len(ds[args.split])} rows), model={args.model}\n")
    print(format_report(report))
    print("\n[info] baseline (always predict NONE):")
    print(format_report(report["baseline"]))

    if args.out:
        import json
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[info] wrote {args.out}")


if __name__ == "__main__":
    main()
