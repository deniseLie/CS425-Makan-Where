#!/usr/bin/env python3
"""Style detector (SingBERT, "english" | "singlish"): macro F1 against a keyword baseline.

Test sets:
  dev    the held-out dev rows training/style_singbert.py builds (user turns of the synthetic and
         MultiWOZ [TRACK] dev files, texts that also occur in train removed). Unbalanced: MultiWOZ
         adds many English turns.
  gold   the user turns of eval/gold/*.jsonl with their gold style label: hand-written messages,
         not from the synthetic templates the classifier was trained on

The model runs as src.style.hf.detect_style does (the same pipeline and label mapping), batched.
The baseline is style_singbert.keyword_style: Singlish if any particle / construction matches.

    python eval/eval_style.py
    python eval/eval_style.py --sets gold

Writes the model's mistakes to outputs/eval/style_<set>_errors.jsonl.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

_spec = importlib.util.spec_from_file_location("style_singbert", ROOT / "training" / "style_singbert.py")
sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sb)


def dev_rows(min_words: int = 2) -> list[dict]:
    """The same dev rows as `python training/style_singbert.py` evaluates on."""
    seen = {r["text"].lower() for r in sb.read("train", [], min_words)}
    return [r for r in sb.read("dev", [], min_words) if r["text"].lower() not in seen]


def gold_rows(path: Path) -> list[dict]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    out = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                out += [{"text": t["raw"], "label": t["style"], "source": "gold"} for t in d["turns"]]
    return out


def classify(texts: list[str], batch: int) -> list[str]:
    from src.style.hf import _FALLBACK_LABELS, _classifier

    labels = []
    for out in _classifier()(texts, truncation=True, batch_size=batch):
        label = str(out["label"]).lower()
        labels.append(_FALLBACK_LABELS.get(label, label))
    bad = set(labels) - {"english", "singlish"}
    if bad:
        raise ValueError(f"style model returned unknown label(s) {bad}")
    return labels


def report(name: str, rows: list[dict], preds: dict[str, list[str]]) -> dict:
    gold = [r["label"] for r in rows]
    out = {"n": len(rows), "n_singlish": gold.count("singlish")}
    print(f"\n{name}: {len(rows)} texts ({gold.count('singlish')} singlish / {gold.count('english')} english)")
    for system, pred in preds.items():
        s = sb.scores(gold, pred)
        out[system] = s
        print(f"  {system:8s} macro F1 {s['macro_f1']:.3f}  acc {s['acc']:.3f}  "
              f"singlish P/R/F1 {s['singlish_p']:.3f}/{s['singlish_r']:.3f}/{s['singlish_f1']:.3f}  "
              f"english F1 {s['english_f1']:.3f}")
    sources = sorted({r["source"] for r in rows})
    if len(sources) > 1:
        for source in sources:
            idx = [i for i, r in enumerate(rows) if r["source"] == source]
            accs = {system: sum(pred[i] == rows[i]["label"] for i in idx) / len(idx) for system, pred in preds.items()}
            print(f"    {source:9s} n={len(idx):5d}  acc " + "  ".join(f"{k} {v:.3f}" for k, v in accs.items()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", nargs="+", default=["dev", "gold"], choices=["dev", "gold"])
    ap.add_argument("--gold", type=Path, default=ROOT / "eval" / "gold")
    ap.add_argument("--eval-n", type=int, default=0, help="score a random sample of this many dev rows; 0 = all")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "eval")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sets = {}
    if "dev" in args.sets:
        rows = dev_rows()
        if args.eval_n:
            rows = random.Random(42).sample(rows, min(args.eval_n, len(rows)))
        sets["dev"] = rows
    if "gold" in args.sets:
        sets["gold"] = gold_rows(args.gold)

    summary = {}
    for name, rows in sets.items():
        if not rows:
            print(f"{name}: no rows")
            continue
        texts = [r["text"] for r in rows]
        preds = {"model": classify(texts, args.batch), "keyword": [sb.keyword_style(t) for t in texts]}
        summary[name] = report(name, rows, preds)
        wrong = [{**r, "pred": p} for r, p in zip(rows, preds["model"]) if p != r["label"]]
        with open(args.out_dir / f"style_{name}_errors.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in wrong)
    (args.out_dir / "style.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
