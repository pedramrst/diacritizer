"""
Apply a trained CANINE diacritizer to raw Persian text: re-insert diacritics.

Usage:
    python diacritize.py --model ./canine-fa-diacritizer --text "کتاب من"
    python diacritize.py --model your-username/canine-fa-diacritizer --text "..."
"""

import argparse
import os

import torch
from dotenv import load_dotenv
from transformers import CanineTokenizer, CanineForTokenClassification

from diacritizer.preprocessing import render_with_labels

load_dotenv()


def diacritize(text, model, tokenizer, device):
    enc = tokenizer(text, return_tensors="pt", truncation=True,
                    max_length=2048).to(device)
    with torch.no_grad():
        logits = model(**enc).logits
    preds = torch.argmax(logits, dim=-1)[0].tolist()

    # CANINE is one-code-point-per-char: drop [CLS] (first) and [SEP] (last),
    # then the remaining predictions align 1:1 with the input characters.
    char_preds = preds[1:1 + len(text)]
    return render_with_labels(text, char_preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="local checkpoint dir, or an HF Hub repo id")
    ap.add_argument("--text", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Only relevant when --model is a (possibly private) HF Hub repo id.
    hub_token = os.environ.get("HF_MODEL_TOKEN")
    tokenizer = CanineTokenizer.from_pretrained(args.model, token=hub_token)
    model = CanineForTokenClassification.from_pretrained(
        args.model, token=hub_token
    ).to(device).eval()

    print(diacritize(args.text, model, tokenizer, device))


if __name__ == "__main__":
    main()
