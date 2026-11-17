#!/usr/bin/env python3
"""Training data for the Flan-T5 normalizer with entities masked as the pipeline masks them.

At run time src.normalizer swaps places, dishes, cuisines and dietary words for typed
placeholders ("in tampines got halal food" -> "in Place1 got Diet1 food") and swaps them back
after the model. The deployed model never saw a placeholder in training, so it often drops or
garbles them. This builds training data that has them, following Crego et al. (2016, SYSTRAN)
and Post et al. (2019, "An Exploration of Placeholding in NMT"):

  - masks come from the same code the pipeline runs (src.normalizer.entities.mask_pair), so
    training and inference can't drift apart;
  - a term is masked only when it appears in both the Singlish and the English, with the same
    placeholder in each, so the target teaches the model to copy it ("kopi" -> "coffee" stays
    as it is: that is a translation to learn);
  - SEAT rows are kept as they are and a masked copy is added, so the model sees entities both
    ways (SYSTRAN trains on a mix);
  - SEAT has few restaurant entities (about 25 of 1,282 training rows name an area or a dish),
    so in-domain Singlish/English pairs are added from the synthetic dialogues
    (data/synthetic_data, "User: ... / Normalized: ..."), masked.

Splits: SEAT is split exactly as normalizer_training.ipynb splits it (80 / 10 / 10,
random_state=42) and the training split gets the notebook's cleaning (seven corrected targets,
two rows removed). The SEAT test split is not written: eval/eval_normalizer.py rebuilds it and
runs it through normalize(), masking included. Synthetic pairs come from
dialogues_synth_train.jsonl (training) and dialogues_synth_dev.jsonl (validation); any whose
Singlish is also a turn of eval/gold is dropped.

    python training/build_normalizer_data.py                  # 1,000 synthetic training pairs
    python training/build_normalizer_data.py --synthetic 0    # SEAT only

Writes outputs/normalizer_data/train.csv and validation.csv (columns singlish, english, source,
masked; the notebook's preprocess_function reads singlish and english). Train with the same
prefix the pipeline uses: "Normalize this Singlish into natural English: ".
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from src.normalizer.entities import mask_pair  # noqa: E402

SEAT_REPO = "kumarsujit474/singlish-english-aligned-translation-seat"
SEAT_FILE = "SEAT_Dataset.csv"
SEED = 42  # normalizer_training.ipynb's random_state

# normalizer_training.ipynb, "cleaned training experiment": applied to the training split only.
CORRECTIONS = {
    "I thought we gg hawker tabao":
        "I thought we were going to get takeaway from the hawker centre.",
    "then 10.45 can liao lol": "Then 10:45 works, lol.",
    "U dont play play hor": "Don't play around, okay?",
    "What she need arh": "What does she need?",
    "i think most ppl here alr knows who their gifter is la":
        "I think most people here already know who their gifter is.",
    "And I on vacation lah": "And I'm on vacation.",
    "Dun liddat leh": "Don't be like that.",
}
EXCLUDE = {
    "mai lai ah hahah dont sue leh my pants will drop",
    "the other day one cmi macam practice session",
}

TURN = re.compile(r"^User: (.*)\nNormalized: (.*)$", re.MULTILINE)


def seat_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The notebook's cleaned training split and its validation split."""
    from huggingface_hub import hf_hub_download
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(hf_hub_download(SEAT_REPO, SEAT_FILE, repo_type="dataset"))
    train, rest = train_test_split(df, test_size=0.2, random_state=SEED)
    val, _test = train_test_split(rest, test_size=0.5, random_state=SEED)

    train = train[~train["singlish"].isin(EXCLUDE)].copy()
    for source, target in CORRECTIONS.items():
        train.loc[train["singlish"].eq(source), "english"] = target
    return train[["singlish", "english"]], val[["singlish", "english"]]


def gold_inputs(path: Path) -> set[str]:
    out = set()
    for f in sorted(path.glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.update(t["raw"].strip().lower() for t in json.loads(line)["turns"])
    return out


def synthetic_pairs(path: Path, exclude: set[str]) -> list[tuple[str, str]]:
    """Distinct (singlish, english) user turns of the Singlish synthetic dialogues, in file order."""
    seen, pairs = set(), []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("style") != "singlish":
            continue
        for sg, en in TURN.findall(row["input"]):
            sg, en = sg.strip(), en.strip()
            key = sg.lower()
            if not sg or not en or key == en.lower() or key in seen or key in exclude:
                continue
            seen.add(key)
            pairs.append((sg, en))
    return pairs


def masked_rows(pairs, source: str, keep_original: bool) -> list[dict]:
    """Each pair masked; with keep_original, the unmasked pair too when masking changed it."""
    rows = []
    for sg, en in pairs:
        msg, men, mapping = mask_pair(str(sg), str(en))
        if keep_original or not mapping:
            rows.append({"singlish": sg, "english": en, "source": source, "masked": False})
        if mapping:
            rows.append({"singlish": msg, "english": men, "source": source, "masked": True})
    return rows


def sample(pairs: list, n: int, rng: random.Random) -> list:
    return pairs if n >= len(pairs) else rng.sample(pairs, n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--synthetic", type=int, default=1000,
                    help="synthetic pairs added to training (validation gets an eighth as many)")
    ap.add_argument("--synthetic-dir", type=Path, default=ROOT / "data" / "synthetic_data")
    ap.add_argument("--gold", type=Path, default=ROOT / "eval" / "gold")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "normalizer_data")
    args = ap.parse_args()
    rng = random.Random(SEED)

    seat_train, seat_val = seat_splits()
    gold = gold_inputs(args.gold)
    syn_train = synthetic_pairs(args.synthetic_dir / "dialogues_synth_train.jsonl", gold)
    syn_val = synthetic_pairs(args.synthetic_dir / "dialogues_synth_dev.jsonl", gold | {s.lower() for s, _ in syn_train})

    train = (masked_rows(zip(seat_train["singlish"], seat_train["english"]), "seat", keep_original=True)
             + masked_rows(sample(syn_train, args.synthetic, rng), "synthetic", keep_original=False))
    val = (masked_rows(zip(seat_val["singlish"], seat_val["english"]), "seat", keep_original=False)
           + masked_rows(sample(syn_val, args.synthetic // 8, rng), "synthetic", keep_original=False))
    rng.shuffle(train)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", val)):
        df = pd.DataFrame(rows, columns=["singlish", "english", "source", "masked"])
        df.to_csv(args.out_dir / f"{name}.csv", index=False)
        counts = df.groupby(["source", "masked"]).size().to_dict()
        print(f"{name:10s} {len(df):5d} rows  " + "  ".join(
            f"{src}{' masked' if m else ''}: {n}" for (src, m), n in sorted(counts.items())))
    print(f"synthetic pairs available: {len(syn_train)} train, {len(syn_val)} dev "
          f"({len(gold)} gold inputs excluded)")
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
