"""Train the SingBERT style classifier ("english" | "singlish"), evaluate it, and push it.

The data is built from the [TRACK] files already in data/, one example per user turn:
  - singlish: the user line of each synthetic row with style "singlish"
  - english:  the normalized line of those rows (the English side of each pair), the user line of
              synthetic rows with style "english", and MultiWOZ user turns
Texts are de-duplicated, one-word messages dropped, and dev texts that also occur in train are
removed, since the synthetic templates repeat phrases. Train is balanced by downsampling English,
half from the synthetic side and half from MultiWOZ, so length and topic don't give the class away.
The synthetic Singlish is templated; add real messages (CoSEM, NUS SMS Corpus) with --extra.

Run from anywhere in a checkout (needs backend/requirements-ml.txt):

    python training/style_singbert.py                 # train, then evaluate on dev
    python training/style_singbert.py --push          # ... and push to makan-where-cs425/makan-singbert-style-classifier
    python training/style_singbert.py --eval-only makan-where-cs425/makan-singbert-style-classifier   # score a trained model

The eval runs the model through the backend's own src.style.hf.detect_style, and scores the
keyword stub on the same rows as a baseline. Checkpoints go to --out (default outputs/style_singbert,
gitignored); rerunning resumes from the newest. --push needs HF_TOKEN with write access to the repo.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "backend"))

FILES = {
    "train": ["synthetic_data/dialogues_synth_train.jsonl", "woz_converted_data/dialogues_train.jsonl"],
    "dev": ["synthetic_data/dialogues_synth_dev.jsonl", "woz_converted_data/dialogues_dev.jsonl"],
}
ID2LABEL = {0: "english", 1: "singlish"}  # what src/style/hf.py expects in config.json
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def last_line(prompt: str, prefix: str) -> str:
    return [line[len(prefix):] for line in prompt.split("\n") if line.startswith(prefix)][-1].strip()


def read(split: str, extra: list[Path], min_words: int) -> list[dict]:
    """{"text", "label", "source"} rows for one split, de-duplicated by text."""
    rows = []
    for f in FILES[split]:
        with open(DATA / f, encoding="utf-8") as fh:
            for row in map(json.loads, filter(str.strip, fh)):
                user = last_line(row["input"], "User: ")
                if row["source"] == "multiwoz":
                    rows.append({"text": user, "label": "english", "source": "multiwoz"})
                elif row["style"] == "singlish":
                    rows.append({"text": user, "label": "singlish", "source": "synthetic"})
                    rows.append({"text": last_line(row["input"], "Normalized: "), "label": "english",
                                 "source": "synthetic"})
                else:
                    rows.append({"text": user, "label": "english", "source": "synthetic"})
    for f in extra:
        with open(f, encoding="utf-8") as fh:
            for row in map(json.loads, filter(str.strip, fh)):
                label = ID2LABEL[row["label"]] if isinstance(row["label"], int) else row["label"].lower()
                assert label in LABEL2ID, f"{f}: label must be english/singlish or 0/1, got {row['label']!r}"
                rows.append({"text": row["text"].strip(), "label": label, "source": row.get("source", f.stem)})

    out, seen = [], {}
    for r in rows:
        key = r["text"].lower()
        if len(r["text"].split()) < min_words:
            continue
        if key in seen:
            assert seen[key] == r["label"], f"{r['text']!r} is labelled both ways"
            continue
        seen[key] = r["label"]
        out.append(r)
    return out


def balance(rows: list[dict], rng: random.Random) -> list[dict]:
    """All Singlish rows, and as many English rows: half synthetic, the rest MultiWOZ / extra."""
    sing = [r for r in rows if r["label"] == "singlish"]
    syn_en = [r for r in rows if r["label"] == "english" and r["source"] == "synthetic"]
    other_en = [r for r in rows if r["label"] == "english" and r["source"] != "synthetic"]
    take_syn = min(len(syn_en), len(sing) // 2)
    take_other = min(len(other_en), len(sing) - take_syn)
    out = sing + rng.sample(syn_en, take_syn) + rng.sample(other_en, take_other)
    rng.shuffle(out)
    return out


def summary(rows: list[dict]) -> str:
    return str(dict(Counter(f"{r['source']}/{r['label']}" for r in rows)))


def encode(tok, rows: list[dict], max_len: int) -> list[dict]:
    enc = tok([r["text"] for r in rows], truncation=True, max_length=max_len)
    return [{"input_ids": ids, "attention_mask": mask, "labels": LABEL2ID[r["label"]]}
            for ids, mask, r in zip(enc["input_ids"], enc["attention_mask"], rows)]


def train(args, train_rows: list[dict], dev_rows: list[dict]) -> Path:
    import numpy as np
    import torch
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding,
                              Trainer, TrainingArguments)

    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2, id2label=ID2LABEL,
                                                               label2id=LABEL2ID)

    def metrics(pred):
        gold, got = pred.label_ids, np.argmax(pred.predictions, axis=-1)
        return scores([ID2LABEL[int(g)] for g in gold], [ID2LABEL[int(p)] for p in got])

    steps = math.ceil(math.ceil(len(train_rows) / args.batch) * args.epochs)
    targs = TrainingArguments(
        output_dir=str(args.out), num_train_epochs=args.epochs, learning_rate=args.lr, weight_decay=0.01,
        warmup_steps=max(1, int(0.1 * steps)), per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=64, fp16=torch.cuda.is_available(), logging_steps=max(1, steps // 20),
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, report_to="none", seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=encode(tok, train_rows, args.max_len),
                      eval_dataset=encode(tok, dev_rows, args.max_len), data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=metrics)
    print(f"{steps} optimizer steps, dev scores after each epoch")
    trainer.train(resume_from_checkpoint=any(args.out.glob("checkpoint-*")) or None)

    final = args.out / "final"  # the model after the last epoch: dev is only looked at, not selected on
    trainer.save_model(str(final))
    tok.save_pretrained(final)
    print("saved", final)
    return final


def scores(gold: list[str], pred: list[str]) -> dict:
    """Accuracy, precision / recall / F1 per class, and macro F1 (the plan's metric)."""
    out = {"n": len(gold), "acc": sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0}
    f1s = []
    for c in LABEL2ID:
        tp = sum(g == p == c for g, p in zip(gold, pred))
        n_pred, n_gold = pred.count(c), gold.count(c)
        prec, rec = tp / n_pred if n_pred else 0.0, tp / n_gold if n_gold else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out |= {f"{c}_p": prec, f"{c}_r": rec, f"{c}_f1": f1}
        f1s.append(f1)
    out["macro_f1"] = sum(f1s) / len(f1s)
    return out


def evaluate(model: str, rows: list[dict], out: Path) -> None:
    """Scores the model as the backend serves it, and the stub as a baseline, overall and per source."""
    os.environ["MAKAN_STYLE_MODEL"] = str(model)
    from src.style import hf, stub

    preds = {"model": [hf.detect_style(r["text"]) for r in rows], "stub": [stub.detect_style(r["text"]) for r in rows]}
    for name, pred in preds.items():
        for source in ["all"] + sorted({r["source"] for r in rows}):
            idx = [i for i, r in enumerate(rows) if source == "all" or r["source"] == source]
            s = scores([rows[i]["label"] for i in idx], [pred[i] for i in idx])
            if source != "all":  # one class per source for MultiWOZ: macro F1 means nothing there
                s = {k: s[k] for k in ("n", "acc")}
            print(f"{name:5} {source:9}", {k: round(v, 3) for k, v in s.items()})
    wrong = [r | {"pred": p} for r, p in zip(rows, preds["model"]) if p != r["label"]]
    with open(out / "eval_style_dev_errors.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in wrong)
    print(f"{len(wrong)} model errors -> {out / 'eval_style_dev_errors.jsonl'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="zanelim/singbert", help="BERT-base pretrained on Singlish / Manglish")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "style_singbert")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128, help="tokens; longer messages are truncated")
    ap.add_argument("--min-words", type=int, default=2, help="drop shorter messages ('ok', 'thanks')")
    ap.add_argument("--extra", type=Path, nargs="*", default=[],
                    help="more train jsonl files with {'text', 'label': 'english'|'singlish'} (e.g. CoSEM, NUS SMS)")
    ap.add_argument("--extra-dev", type=Path, nargs="*", default=[], help="the same, added to dev")
    ap.add_argument("--limit", type=int, help="train on only this many rows (smoke test)")
    ap.add_argument("--eval-n", type=int, default=0, help="score only this many dev rows; 0 = all")
    ap.add_argument("--eval-only", metavar="MODEL", help="evaluate this trained model (path or Hub id), no training")
    ap.add_argument("--push", nargs="?", const="makan-where-cs425/makan-singbert-style-classifier", metavar="REPO",
                    help="after training, push model + tokenizer to this private Hub repo")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    train_all = read("train", args.extra, args.min_words)
    seen = {r["text"].lower() for r in train_all}
    train_rows = balance(train_all, rng)[:args.limit]
    dev_all = read("dev", args.extra_dev, args.min_words)
    dev_rows = [r for r in dev_all if r["text"].lower() not in seen]
    print("train:", summary(train_rows))
    print(f"dev:   {summary(dev_rows)} ({len(dev_all) - len(dev_rows)} texts also in train dropped)")
    if args.eval_n:
        dev_rows = rng.sample(dev_rows, min(args.eval_n, len(dev_rows)))

    model = args.eval_only or train(args, train_rows, dev_rows)
    evaluate(model, dev_rows, args.out)

    if args.push and not args.eval_only:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        msg = f"SingBERT style classifier, {args.epochs:g} epoch(s)"
        AutoModelForSequenceClassification.from_pretrained(model).push_to_hub(args.push, private=True, commit_message=msg)
        AutoTokenizer.from_pretrained(model).push_to_hub(args.push, private=True)
        print(f"pushed https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
