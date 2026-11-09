#!/usr/bin/env python3
"""Normalizer (Flan-T5, Singlish -> English): BLEU and chrF against reference English rewrites.

Test sets:
  seat-test   SEAT (kumarsujit474/singlish-english-aligned-translation-seat on the HF Hub, 1,603
              human-translated pairs): the 161-pair test split the normalizer was held out from.
              SEAT ships as one file, so normalizer_training.ipynb splits it 80 / 10 / 10 with
              sklearn's train_test_split (random_state=42, then the 20% halved the same way);
              this rebuilds that split (the rows of model_comparison.csv, in the same order).
  seat-all    every SEAT pair (optional; 80% of them were training data)
  gold        the Singlish user turns of eval/gold/*.jsonl, raw -> english_version: restaurant
              requests the normalizer has certainly not seen

Baseline: copying the Singlish input unchanged ("no normalizer"). Scores are sacrebleu's corpus
BLEU and chrF (cased, default tokenization). The model runs as src.normalizer.hf.normalize does
(same prefix, beam 4, up to 128 new tokens), one input at a time by default: in a batch, one input
that loops to 128 tokens ("Wahhh lobang Lai Liao") stalls the whole batch.

    python eval/eval_normalizer.py                       # SEAT test + gold (~2 min on CPU)
    python eval/eval_normalizer.py --sets gold           # just the gold dialogues

Writes every output to outputs/eval/normalizer_<set>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

SEAT_REPO = "kumarsujit474/singlish-english-aligned-translation-seat"
SEAT_FILE = "SEAT_Dataset.csv"
SEED = 42  # normalizer_training.ipynb's random_state


def load_seat() -> list[dict]:
    """Every SEAT row in file order (the split below depends on the order and count: 1,603)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    df = pd.read_csv(hf_hub_download(SEAT_REPO, SEAT_FILE, repo_type="dataset"))
    return [{"id": f"seat{i:04d}", "src": str(s).strip(), "ref": str(e).strip()}
            for i, (s, e) in enumerate(zip(df["singlish"], df["english"]))]


def seat_test(rows: list[dict]) -> list[dict]:
    """The notebook's test split: 20% held out, then halved into validation and test."""
    from sklearn.model_selection import train_test_split

    _, rest = train_test_split(rows, test_size=0.2, random_state=SEED)
    _, test = train_test_split(rest, test_size=0.5, random_state=SEED)
    return test


def load_gold_turns(path: Path) -> list[dict]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    out = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            for i, t in enumerate(d["turns"]):
                if t["style"] == "singlish" and t.get("english_version"):
                    out.append({"id": f"{d['dialogue_id']}_{i}", "src": t["raw"], "ref": t["english_version"]})
    return out


def normalize_all(texts: list[str], batch: int) -> list[str]:
    import torch

    from src.normalizer.hf import MAX_NEW_TOKENS, PREFIX, _load

    tok, model = _load()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [""] * len(texts)
    start = time.time()
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        enc = tok([PREFIX + texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, num_beams=4)
        for i, text in zip(idx, tok.batch_decode(gen, skip_special_tokens=True)):
            out[i] = text.strip() or texts[i]  # normalize() falls back to the input on an empty output
        print(f"  {min(b + batch, len(order))}/{len(order)} ({time.time() - start:.0f} s)", flush=True)
    return out


def scores(hyps: list[str], refs: list[str]) -> dict:
    import sacrebleu

    return {"BLEU": sacrebleu.corpus_bleu(hyps, [refs]).score, "chrF": sacrebleu.corpus_chrf(hyps, [refs]).score,
            "exact": sum(h.strip().lower() == r.strip().lower() for h, r in zip(hyps, refs)) / len(refs)}


def report(name: str, rows: list[dict]) -> dict:
    refs = [r["ref"] for r in rows]
    model = scores([r["pred"] for r in rows], refs)
    copy = scores([r["src"] for r in rows], refs)
    print(f"{name:10s} n={len(rows):5d}  model BLEU {model['BLEU']:5.1f}  chrF {model['chrF']:5.1f}  "
          f"exact {model['exact']:5.1%}   |  copy-input BLEU {copy['BLEU']:5.1f}  chrF {copy['chrF']:5.1f}")
    return {"n": len(rows), "model": model, "copy_input": copy}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", nargs="+", default=["seat-test", "gold"],
                    choices=["seat-test", "seat-all", "gold"])
    ap.add_argument("--gold", type=Path, default=ROOT / "eval" / "gold")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "eval")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sets: dict[str, list[dict]] = {}
    if {"seat-test", "seat-all"} & set(args.sets):
        seat = load_seat()
        if "seat-all" in args.sets:
            sets["seat-all"] = seat
        if "seat-test" in args.sets:
            sets["seat-test"] = seat_test(seat)
    if "gold" in args.sets:
        sets["gold"] = load_gold_turns(args.gold)

    # each distinct input is normalized once (seat-test is a subset of seat-all)
    texts = sorted({r["src"] for rows in sets.values() for r in rows})
    print(f"normalizing {len(texts)} distinct inputs")
    pred = dict(zip(texts, normalize_all(texts, args.batch)))

    summary = {}
    print()
    for name, rows in sets.items():
        rows = [{**r, "pred": pred[r["src"]]} for r in rows]
        if not rows:
            print(f"{name:10s} no rows")
            continue
        summary[name] = report(name, rows)
        with open(args.out_dir / f"normalizer_{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (args.out_dir / "normalizer.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
