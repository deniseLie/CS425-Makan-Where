#!/usr/bin/env python3
"""Recommender evaluation on the gold dialogues (eval/gold/*.jsonl, format in eval/gold/README.md).

Each dialogue's last gold state and English request go to the ranker; its top K is scored against
the dialogue's judged places: P@K and hit@K count places judged relevant (grade >= 1), strict P@K
only grade 2, and nDCG@K uses the grades as gains (or, without grades, the order of
relevant_place_ids). The rating-only baseline ranks the same candidates by quality alone.

    python eval/eval_recommender.py --k 3 --ablations
    python eval/eval_recommender.py --pool outputs/eval/pool.jsonl   # places to judge, per dialogue

--pool writes, for each dialogue, the union of every system's top places (hybrid and rating-only
to depth 5, each ablation to depth K), shuffled and without system names, with the details a judge
needs. Judge each place 0 / 1 / 2 (see eval/gold/README.md) and store the grades in `relevance`.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from src import recommender  # noqa: E402
from src.policy import MAX_ASKS, decide  # noqa: E402
from src.recommender import filters, ranking  # noqa: E402
from src.schemas import DialogueState  # noqa: E402

# baseline: the same candidates, ranked by the quality score (Bayesian-averaged rating) alone
RATING_ONLY = {part: (1.0 if part == "quality" else 0.0) for part in ranking.WEIGHTS}


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


def ndcg_at_k(ranked, relevant_ordered, k, grades=None):
    """grades {place_id: 0|1|2} are the gains when given; otherwise the gain falls with the
    position in relevant_ordered (most relevant first)."""
    if grades:
        gain = {pid: g for pid, g in grades.items() if g > 0}
    else:
        n = len(relevant_ordered)
        gain = {pid: n - i for i, pid in enumerate(relevant_ordered)}
    dcg = sum(gain.get(p, 0) / math.log2(i + 2) for i, p in enumerate(ranked[:k]))
    ideal = sum(g / math.log2(i + 2) for i, g in enumerate(sorted(gain.values(), reverse=True)[:k]))
    return dcg / ideal if ideal else 0.0


def relevant_ids(d, min_grade=1):
    """Place ids judged at least min_grade; without grades, relevant_place_ids (grade 1)."""
    grades = d.get("relevance")
    if grades:
        return {pid for pid, g in grades.items() if g >= min_grade}
    return set(d["relevant_place_ids"]) if min_grade <= 1 else set()


def recommend(state, query, k, weights=None):
    places = ranking.rank(state, query, None, k=k, weights=weights)
    if not places:  # same fallback the policy uses
        places = decide(state, MAX_ASKS, recommender, query_norm=query).places
    return [p.place_id for p in places]


QUERIES = {}  # raw text -> the query to rank on instead of english_version (--query-from)


def last_request(d):
    last = d["turns"][-1]
    query = QUERIES.get(last["raw"]) or last.get("english_version") or last["raw"]
    return DialogueState.from_dict(last["gold_state"]), query


def evaluate(gold, k, weights=None):
    rows = recommender.load_places().set_index("place_id")
    p_sum = s_sum = n_sum = h_sum = 0.0
    n_returned = n_bad = 0
    for d in gold:
        state, query = last_request(d)
        ranked = recommend(state, query, k, weights)
        relevant = relevant_ids(d)
        p_sum += precision_at_k(ranked, relevant, k)
        s_sum += precision_at_k(ranked, relevant_ids(d, 2), k)
        n_sum += ndcg_at_k(ranked, d["relevant_place_ids"], k, d.get("relevance"))
        h_sum += hit_at_k(ranked, relevant, k)
        cands = {c["place_id"]: c for c in filters.filtered(state).to_dict("records")}
        for pid in ranked:
            n_returned += 1
            place = cands.get(pid) or {**rows.loc[pid].to_dict(), "place_id": pid, "dist_km": None}
            if filters.violations(state, place):
                n_bad += 1
    n = max(len(gold), 1)
    return {"P@K": p_sum / n, "strict P@K": s_sum / n, "nDCG@K": n_sum / n, "hit@K": h_sum / n,
            "violation rate": n_bad / n_returned if n_returned else 0.0}


def show(name, m):
    print(f"{name:<22} P@K {m['P@K']:.3f}  strict P@K {m['strict P@K']:.3f}  nDCG@K {m['nDCG@K']:.3f}  "
          f"hit@K {m['hit@K']:.3f}  violations {m['violation rate']:.1%}")


def systems(k):
    """(name, weights, depth) of every ranking compared; pooled to `depth` for judging."""
    out = [("hybrid", None, max(k, 5)), ("rating only", RATING_ONLY, max(k, 5))]
    return out + [(f"without {part}", {part: 0.0}, k) for part in ranking.WEIGHTS]


def pool(gold, k, seed=42):
    """Per dialogue, the union of every system's top places with the details a judge needs."""
    rng = random.Random(seed)
    out = []
    for d in gold:
        state, query = last_request(d)
        ids = []
        for _, weights, depth in systems(k):
            ids += [pid for pid in recommend(state, query, depth, weights) if pid not in ids]
        rng.shuffle(ids)  # judges must not see which system ranked a place, or where
        cands = filters.filtered(state).set_index("place_id")
        rows = recommender.load_places().set_index("place_id")
        places = []
        for pid in ids:
            r = cands.loc[pid] if pid in cands.index else rows.loc[pid]
            dist = r.get("dist_km")
            places.append({
                "place_id": pid, "name": r["name"], "cuisine": r["cuisine"] if isinstance(r["cuisine"], str) else None,
                "setting": r["setting"], "tags": r["tags"] or None, "description": r["description"] or None,
                "address": r["address"] if isinstance(r["address"], str) else None, "area": r["area"],
                "price": None if math.isnan(r["price_min"]) else
                f"${r['price_min']:g}-{r['price_max']:g}" + (" (estimate by type of place)"
                                                            if r["price_source"] == "estimate" else ""),
                "rating": None if math.isnan(r["rating"]) else f"{r['rating']:.1f} ({r['n_reviews']:.0f} reviews)",
                "halal": bool(r["halal"]), "vegetarian": bool(r["vegetarian"]),
                "opening_hours": r["opening_hours"] or None,
                "km_from_requested_spot": None if dist is None or math.isnan(dist) else round(float(dist), 2),
            })
        out.append({"dialogue_id": d["dialogue_id"],
                    "user_turns": [t.get("english_version") or t["raw"] for t in d["turns"]],
                    "final_state": {"slots": {s: v for s, v in state.slots.items() if v != "UNKNOWN"},
                                    "near": state.landmark, "radius_km": state.radius_km,
                                    "budget_max": state.budget_max},
                    "places": places})
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate the recommender on gold dialogues.")
    parser.add_argument("--gold", type=Path, default=ROOT / "eval" / "gold")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--places", type=Path, default=None, help="places file (default: config)")
    parser.add_argument("--ablations", action="store_true")
    parser.add_argument("--by-style", action="store_true", help="also score Singlish and English dialogues apart")
    parser.add_argument("--pool", type=Path, default=None, metavar="OUT",
                        help="write the places to judge per dialogue to OUT (jsonl) and stop")
    parser.add_argument("--query-from", type=Path, default=None, metavar="FILE",
                        help="rank on the normalizer's output instead of the gold English rewrite: a jsonl of "
                             "{src, pred} rows, e.g. outputs/eval/normalizer_gold.jsonl from eval_normalizer.py")
    args = parser.parse_args()

    if args.query_from:
        for line in args.query_from.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                QUERIES[r["src"]] = r["pred"]
    if args.places:
        recommender.use_places_csv(args.places)
    gold = load_gold(args.gold)
    if not gold:
        print(f"no gold dialogues found in {args.gold}")
        return
    if args.pool:
        rows = pool(gold, args.k)
        args.pool.parent.mkdir(parents=True, exist_ok=True)
        args.pool.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"{sum(len(r['places']) for r in rows)} places to judge for {len(rows)} dialogues -> {args.pool}")
        return
    unjudged = [d["dialogue_id"] for d in gold if not d.get("relevance") and not d["relevant_place_ids"]]
    if unjudged:
        print(f"warning: {len(unjudged)} dialogues have no judged places yet: {unjudged[:5]}...")
    print(f"dialogues: {len(gold)}  K = {args.k}")
    names = ["hybrid", "rating only"] + ([f"without {p}" for p in ranking.WEIGHTS] if args.ablations else [])
    for name, weights, _ in systems(args.k):
        if name in names:
            show(name, evaluate(gold, args.k, weights=weights))
    if args.by_style:
        for style in ("singlish", "english"):
            part = [d for d in gold if d["turns"][-1]["style"] == style]
            if part:
                print(f"\n{style} dialogues: {len(part)}")
                show("hybrid", evaluate(part, args.k))
                show("rating only", evaluate(part, args.k, weights=RATING_ONLY))


if __name__ == "__main__":
    main()
