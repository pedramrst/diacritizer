"""
Shared preprocessing for the Persian diacritization dataset
(avaeziaiteam/harakat-dataset on the HF Hub).

Both train.py and diacritize.py import the label scheme and the
char/label splitting logic from here, so training and inference always agree
on what a label id means.

Dataset shape: two columns, `raw` (no diacritics) and `harakat` (diacritized).
We do NOT trust `raw` for alignment (see MISMATCH note below) -- the base
character sequence is always derived by stripping diacritics out of
`harakat` itself, which guarantees perfect 1:1 alignment between characters
and labels.

Label scheme (data-driven from what's actually in `harakat`):
    NONE, FATHA, DAMMA, KASRA, SOKUN, SHADDA,
    SHADDA_FATHA, SHADDA_DAMMA, SHADDA_KASRA, FATHATAN

Shadda (gemination) stacks with a vowel mark on the same base character
(e.g. "چِّ" che+KASRA+SHADDA), so it needs its own compound
labels rather than being folded into "NONE" or dropped -- it accounts for
~4.6k occurrences in the dataset, too many to discard.

SUPERSCRIPT_ALEF occurs only 9 times in the whole dataset -- too rare to
learn or evaluate (a dev/test split can easily end up with zero examples of
it). It's excluded as a label: still stripped out of the character stream
like every other diacritic, but treated as NONE rather than given its own
class.

A handful of rows (~90 out of 7182) stack SOKUN with a vowel or with SHADDA
on the same character; this is contradictory (SOKUN means "no vowel") and is
almost certainly an annotation slip. We resolve it by preferring the
vowel/SHADDA and dropping the SOKUN in that case (see `_label_for_marks`).
"""

import unicodedata

from datasets import load_dataset, Dataset, DatasetDict

HF_REPO = "avaeziaiteam/harakat-dataset"

FATHA = "َ"
DAMMA = "ُ"
KASRA = "ِ"
SOKUN = "ْ"
SHADDA = "ّ"
FATHATAN = "ً"
SUPERSCRIPT_ALEF = "ٰ"

DIACRITIC_CHARS = {FATHA, DAMMA, KASRA, SOKUN, SHADDA, FATHATAN, SUPERSCRIPT_ALEF}

LABELS = [
    "NONE",
    "FATHA",
    "DAMMA",
    "KASRA",
    "SOKUN",
    "SHADDA",
    "SHADDA_FATHA",
    "SHADDA_DAMMA",
    "SHADDA_KASRA",
    "FATHATAN",
]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)

# For re-inserting diacritics at inference time (diacritize.py).
LABEL2MARKS = {
    "NONE": "",
    "FATHA": FATHA,
    "DAMMA": DAMMA,
    "KASRA": KASRA,
    "SOKUN": SOKUN,
    "SHADDA": SHADDA,
    "SHADDA_FATHA": SHADDA + FATHA,
    "SHADDA_DAMMA": SHADDA + DAMMA,
    "SHADDA_KASRA": SHADDA + KASRA,
    "FATHATAN": FATHATAN,
}


def _label_for_marks(marks):
    """Map an (unordered) set of combining marks found after one base
    character to a single label name."""
    marks = set(marks)
    if FATHATAN in marks:
        return "FATHATAN"
    # SUPERSCRIPT_ALEF is excluded as a label (see module docstring) -- fall
    # through and treat it like any other unrecognized mark (-> NONE), while
    # DIACRITIC_CHARS still makes sure it's stripped out of the char stream.

    vowel = None
    if FATHA in marks:
        vowel = "FATHA"
    elif DAMMA in marks:
        vowel = "DAMMA"
    elif KASRA in marks:
        vowel = "KASRA"

    if SHADDA in marks:
        return f"SHADDA_{vowel}" if vowel else "SHADDA"
    if vowel:
        return vowel
    if SOKUN in marks:
        return "SOKUN"
    return "NONE"


def split_diacritized(text: str):
    """
    Walk a diacritized string. For each BASE character, emit the character
    and the label of whatever diacritic mark(s) immediately follow it. The
    diacritic characters themselves are dropped from the output sequence.

    Returns (list_of_base_chars, list_of_label_ids) of equal length.
    """
    chars, labels = [], []
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch in DIACRITIC_CHARS:
            # A diacritic with no preceding base char (malformed) -> skip safely.
            i += 1
            continue
        j = i + 1
        marks = []
        while j < n and text[j] in DIACRITIC_CHARS:
            marks.append(text[j])
            j += 1
        chars.append(ch)
        labels.append(LABEL2ID[_label_for_marks(marks)])
        i = j
    return chars, labels


def strip_diacritics(text: str) -> str:
    return "".join(c for c in text if c not in DIACRITIC_CHARS)


def render_with_labels(chars, label_ids) -> str:
    """Inverse of split_diacritized: re-insert the diacritic for each label
    id right after its base character. Used to turn model predictions (or
    gold labels) back into readable diacritized text."""
    out = []
    for ch, lid in zip(chars, label_ids):
        out.append(ch)
        out.append(LABEL2MARKS[ID2LABEL[lid]])
    return "".join(out)


def load_and_prepare(repo_or_path=HF_REPO, diac_col="harakat", raw_col="raw",
                      max_chars=None, test_size=0.05, dev_size=0.05, seed=42,
                      token=None):
    """
    Load the dataset (from the HF Hub by default, or a local csv/parquet
    path), turn every row into (chars, labels), and return a DatasetDict
    with train/dev/test splits.

    `test_size` and `dev_size` are each independent fractions of the FULL
    dataset (not of each other) -- test_size=0.05, dev_size=0.05 means 5%
    test, 5% dev, 90% train, directly. (Internally this still takes two
    sequential train_test_split calls, since that's all the underlying API
    supports, but the second split's fraction is rescaled so the externally
    visible test_size/dev_size stay absolute, independent fractions of the
    original total -- setting either one doesn't change what the other means.)

    `token` is an HF access token, needed since the dataset repo is private
    (falls back to any cached `huggingface-cli login` token if not given).
    """
    if test_size + dev_size >= 1.0:
        raise ValueError(
            f"test_size ({test_size}) + dev_size ({dev_size}) = "
            f"{test_size + dev_size} must be < 1.0 -- nothing would be left for train"
        )
    if repo_or_path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(repo_or_path)
        rows = df.to_dict("records")
    elif repo_or_path.endswith(".csv"):
        import pandas as pd
        df = pd.read_csv(repo_or_path)
        rows = df.to_dict("records")
    else:
        rows = load_dataset(repo_or_path, token=token)["train"]

    seen = set()
    records = []
    mismatches = 0
    duplicates = 0
    empties = 0

    for row in rows:
        diac = str(row[diac_col])

        # Dedupe on `raw` alone (not (raw, harakat)): two rows sharing the
        # same raw sentence must never land in different splits, even if a
        # future data update gives them different `harakat` annotations.
        key = str(row[raw_col]) if raw_col in row else diac
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)

        chars, labels = split_diacritized(diac)

        # Sanity check against the provided raw column, if present. We trust
        # the stripped-from-harakat version regardless (see module docstring).
        if raw_col in row:
            if strip_diacritics(diac) != str(row[raw_col]):
                mismatches += 1

        if max_chars is not None and len(chars) > max_chars:
            chars = chars[:max_chars]
            labels = labels[:max_chars]

        if len(chars) == 0:
            empties += 1
            continue

        records.append({"chars": chars, "labels": labels})

    if duplicates:
        print(f"[info] dropped {duplicates} duplicate rows")
    if empties:
        print(f"[info] dropped {empties} empty rows")
    if mismatches:
        print(f"[warn] {mismatches} rows: stripped(harakat) != provided `{raw_col}` "
              f"column (e.g. Ezafe-ی insertions). Using the stripped version, "
              f"the `{raw_col}` column is not used for alignment.")

    ds = Dataset.from_list(records)
    split = ds.train_test_split(test_size=test_size, seed=seed)
    # split["train"] is now only (1 - test_size) of the full dataset, so
    # asking it for `dev_size` of ITSELF would give dev_size * (1 - test_size)
    # of the original total, not dev_size -- rescale so dev_size stays an
    # absolute fraction of the full dataset regardless of test_size.
    dev_fraction_of_remainder = dev_size / (1.0 - test_size)
    devtrain = split["train"].train_test_split(test_size=dev_fraction_of_remainder, seed=seed)
    result = DatasetDict({
        "train": devtrain["train"],
        "dev": devtrain["test"],
        "test": split["test"],
    })
    print(f"[info] prepared train={len(result['train'])} "
          f"dev={len(result['dev'])} test={len(result['test'])}")
    return result


def main():
    import argparse
    import collections

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=HF_REPO,
                    help="HF dataset repo id, or a local .csv/.parquet path")
    ap.add_argument("--out", default=None,
                    help="if set, save the processed DatasetDict here (save_to_disk)")
    args = ap.parse_args()

    ds = load_and_prepare(args.data)

    label_counts = collections.Counter()
    for split in ds.values():
        for labels in split["labels"]:
            for lbl in labels:
                label_counts[ID2LABEL[lbl]] += 1
    print("[info] label distribution (all splits):")
    for name in LABELS:
        print(f"  {name:>20}: {label_counts[name]}")

    if args.out:
        ds.save_to_disk(args.out)
        print(f"[info] saved processed dataset to {args.out}")


if __name__ == "__main__":
    main()
