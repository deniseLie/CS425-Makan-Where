#!/usr/bin/env python3
"""Recommender evaluation on the gold dialogues (eval/gold/*.jsonl, format in eval/gold/README.md)."""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from src import recommender  # noqa: E402
from src.policy import MAX_ASKS, decide  # noqa: E402
from src.recommender import filters, ranking, stub  # noqa: E402
from src.schemas import DialogueState  # noqa: E402


def load_gold(path):
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    out = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def precision_at_k(ranked, relevant, k):
    return sum(1 for p in ranked[:k] if p in relevant) / k


def hit_at_k(ranked, relevant, k):
    return float(any(p in relevant for p in ranked[:k]))


def ndcg_at_k(ranked, relevant_ordered, k):
    n = len(relevant_ordered)
    gain = {pid: n - i for i, pid in enumerate(relevant_ordered)}
    dcg = sum(gain.get(p, 0) / math.log2(i + 2) for i, p in enumerate(ranked[:k]))
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(gain.values(), reverse=True)[:k]))
    return dcg / ideal if ideal else 0.0


def recommend(state, query, k, weights=None, use_stub=False):
    if use_stub:
        places = stub.rank(state, query, None, k=k)
    else:
        places = ranking.rank(state, query, None, k=k, weights=weights)
    if not places:  # same fallback the policy uses
        places = decide(state, MAX_ASKS, recommender, query_norm=query).places
    return [p.place_id for p in places]


def evaluate(gold, k, weights=None, use_stub=False):
    rows = recommender.load_places().set_index("place_id")
    p_sum = n_sum = h_sum = 0.0
    n_returned = n_bad = 0
    for d in gold:
        last = d["turns"][-1]
        state = DialogueState.from_dict(last["gold_state"])
        query = last.get("english_version") or last["raw"]
        ranked = recommend(state, query, k, weights, use_stub)
        relevant = d["relevant_place_ids"]
        p_sum += precision_at_k(ranked, set(relevant), k)
        n_sum += ndcg_at_k(ranked, relevant, k)
        h_sum += hit_at_k(ranked, set(relevant), k)
        cands = {c["place_id"]: c for c in filters.filtered(state).to_dict("records")}
        for pid in ranked:
            n_returned += 1
            place = cands.get(pid) or {**rows.loc[pid].to_dict(), "place_id": pid, "dist_km": None}
            if filters.violations(state, place):
                n_bad += 1
    n = max(len(gold), 1)
    return {"P@K": p_sum / n, "nDCG@K": n_sum / n, "hit@K": h_sum / n,
            "violation rate": n_bad / n_returned if n_returned else 0.0}


def show(name, m):
    print(f"{name:<22} P@K {m['P@K']:.3f}  nDCG@K {m['nDCG@K']:.3f}  hit@K {m['hit@K']:.3f}  "
          f"violations {m['violation rate']:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the recommender on gold dialogues.")
    parser.add_argument("--gold", type=Path, default=ROOT / "eval" / "gold")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--places", type=Path, default=None, help="places file (default: config)")
    parser.add_argument("--ablations", action="store_true")
    args = parser.parse_args()

    if args.places:
        recommender.use_places_csv(args.places)
    # the stub baseline reads the same catalogue
    stub.load_places = recommender.load_places
    stub.area_centroids.cache_clear()
    gold = load_gold(args.gold)
    if not gold:
        print(f"no gold dialogues found in {args.gold}")
        return
    print(f"dialogues: {len(gold)}  K = {args.k}")
    show("hybrid", evaluate(gold, args.k))
    if args.ablations:
        for part in ranking.WEIGHTS:
            show(f"without {part}", evaluate(gold, args.k, weights={part: 0.0}))
        show("rating only (stub)", evaluate(gold, args.k, use_stub=True))


if __name__ == "__main__":
    main()
