#!/usr/bin/env python3
"""Tracker on the gold dialogues, with and without the normalizer.

Replays each gold dialogue (eval/gold/*.jsonl) through the backend's own state update,
MakanSession.step: the tracker, value translation, the budget / dietary rules, critiques and
landmark detection. The state after every user turn is compared with that turn's gold_state.
Under each user line the tracker reads a `Normalized:` line. This script compares three
versions of that line:

  normalizer   the Flan-T5 normalizer's output, as the backend runs today
  raw          the user's own text: no normalizer (build_prompt's `t.normalized or t.raw`)
  gold         the turn's english_version, what a perfect normalizer would write

The pipeline gives the same text to the ranker (query_norm) and to landmark detection. So the
script also ranks each dialogue's final tracked state and scores the top K against the gold
relevance judgments, as eval_recommender.py does from the gold state.

The replies between turns are the policy's template replies, not the Qwen generator's, and the
style is the gold label. The tracker reads the reply text, so this keeps generation out of the
comparison. The tracker runs as llm.complete does: Qwen2.5-1.5B with the [TRACK] adapter,
through the chat template, greedy, up to 256 new tokens. Prompts are batched across dialogues,
and on CPU the maths runs in float32 as in eval_generator.py. A laptop CPU takes about 30 minutes.

    python eval/eval_tracker.py                          # all three versions
    python eval/eval_tracker.py --variants normalizer raw

Writes the state after every turn to outputs/eval/tracker_gold.jsonl. Tracker outputs are cached
by prompt in tracker_gold.cache.jsonl, and the normalizer's outputs in
tracker_gold.normalized.json, so a rerun only generates what is new.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "eval"))

import eval_recommender as ev  # noqa: E402
from src import pipeline  # noqa: E402
from src.config import model_id  # noqa: E402
from src.generator import template_reply  # noqa: E402
from src.llm import MAX_NEW_TOKENS  # noqa: E402
from src.schemas import SLOTS, UNKNOWN, Turn  # noqa: E402
from src.tracker.hf import build_prompt, parse  # noqa: E402

VARIANTS = ("normalizer", "raw", "gold")
STATE_FIELDS = (*SLOTS, "radius_km", "budget_max", "landmark")

CACHE: dict[str, str] = {}  # md5 of a tracker prompt -> the tracker's text
CURRENT: dict = {}          # the gold turn and variant being stepped


def key(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def normalizer_outputs(texts: list[str], path: Path) -> dict[str, str]:
    """{raw text: normalize(raw text)}, cached in `path`. The model is freed afterwards."""
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    todo = [t for t in texts if t not in done]
    if todo:
        from src.normalizer import hf

        print(f"normalizing {len(todo)} turns", flush=True)
        for t in todo:
            done[t] = hf.normalize(t)
        path.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
        hf._load.cache_clear()
        gc.collect()
    return done


def int8_weights(model):
    """Low RAM on CPU: keep each layer's Linear weights as int8 with one scale per output row, and
    dequantize them to float32 when used. Only the weights are quantized, not the activations.
    The output layer shares the embedding's weights and stays in bfloat16. This takes the model
    from ~3 GB to ~1.8 GB. Call it after eval_generator.load_model."""
    import torch
    import torch.nn.functional as F
    from eval_generator import fp32_buffer

    class Int8Linear(torch.nn.Module):
        def __init__(self, lin):
            super().__init__()
            w = lin.weight.detach().float()
            scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127
            self.register_buffer("q", torch.round(w / scale[:, None]).clamp(-127, 127).to(torch.int8))
            self.register_buffer("scale", scale)
            self.bias = lin.bias

        def forward(self, x):
            return F.linear(x, fp32_buffer(self.q).mul_(self.scale[:, None]), self.bias)

    tied = model.get_input_embeddings().weight
    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if type(child).__name__ == "UpcastLinear" and child.weight is not tied:
                setattr(mod, name, Int8Linear(child))
    gc.collect()
    return model


def normalized(variant: str, turn: dict, norm_out: dict[str, str]) -> str:
    if variant == "normalizer":
        return norm_out[turn["raw"]]
    if variant == "gold":
        return turn.get("english_version") or turn["raw"]
    return turn["raw"]


def patch_pipeline(norm_out: dict[str, str]) -> None:
    """MakanSession.step with the variant's Normalized text, the gold style, template replies and
    the tracker's cached output."""
    pipeline.normalize = lambda raw: normalized(CURRENT["variant"], CURRENT["turn"], norm_out)
    pipeline.detect_style = lambda raw: CURRENT["turn"]["style"]
    pipeline.generate = lambda action, style, raw: template_reply(action, style)
    pipeline.track = lambda history, prev: parse(CACHE[key(build_prompt(history))], prev)


def same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) < 1e-6
    return str(a).strip().lower() == str(b).strip().lower()


def turn_scores(rows: list[dict]) -> dict:
    """Joint goal accuracy (all six slots right), slot accuracy, slot F1 over the slots that are
    filled in gold or prediction, landmark accuracy and full-state accuracy (slots, radius,
    budget cap and landmark)."""
    n = len(rows)
    if not n:
        return {"n": 0}
    joint = full = slot_ok = landmark = tp = fp = fn = 0
    for r in rows:
        got, gold = r["state"], r["gold_state"]
        ok = {s: same(got["slots"][s], gold["slots"][s]) for s in SLOTS}
        joint += all(ok.values())
        slot_ok += sum(ok.values())
        landmark += same(got.get("landmark"), gold.get("landmark"))
        full += all(ok.values()) and all(same(got.get(f), gold.get(f)) for f in STATE_FIELDS[len(SLOTS):])
        for s in SLOTS:
            g, p = gold["slots"][s], got["slots"][s]
            tp += ok[s] and g != UNKNOWN
            fp += not ok[s] and p != UNKNOWN
            fn += not ok[s] and g != UNKNOWN
    return {"n": n, "joint": joint / n, "slot_acc": slot_ok / (n * len(SLOTS)),
            "slot_f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0, "landmark": landmark / n, "full_state": full / n}


def rank_scores(finals: list[dict], gold: dict[str, dict], k: int) -> dict:
    """P@K / strict P@K / nDCG@K of the ranker on each dialogue's final tracked state."""
    n = len(finals)
    if not n:
        return {"n": 0}
    p = s = g = 0.0
    unjudged = 0
    for f in finals:
        d = gold[f["dialogue_id"]]
        ranked = f["ranked"]
        p += ev.precision_at_k(ranked, ev.relevant_ids(d), k)
        s += ev.precision_at_k(ranked, ev.relevant_ids(d, 2), k)
        g += ev.ndcg_at_k(ranked, d["relevant_place_ids"], k, d.get("relevance"))
        unjudged += sum(pid not in d.get("relevance", {}) for pid in ranked[:k])
    return {"n": n, "P@K": p / n, "strict P@K": s / n, "nDCG@K": g / n, "unjudged": unjudged}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--gold", type=Path, default=ROOT / "eval" / "gold")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--int8", action="store_true",
                    help="CPU with little free RAM: int8 layer weights (~1.8 GB instead of ~3 GB)")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval" / "tracker_gold.jsonl")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cache_path = args.out.with_suffix(".cache.jsonl")

    gold = ev.load_gold(args.gold)
    by_id = {d["dialogue_id"]: d for d in gold}
    norm_out = {}
    if "normalizer" in args.variants:
        texts = sorted({t["raw"] for d in gold for t in d["turns"]})
        norm_out = normalizer_outputs(texts, args.out.with_suffix(".normalized.json"))
    patch_pipeline(norm_out)
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                CACHE[r["id"]] = r["pred"]

    sessions = {(d["dialogue_id"], v): pipeline.MakanSession(log_dir=None) for d in gold for v in args.variants}
    records, tok, model = [], None, None
    for t in range(max(len(d["turns"]) for d in gold)):
        active = [(d, v) for d in gold if len(d["turns"]) > t for v in args.variants]
        prompts, run_key = {}, {}
        for d, v in active:
            turn = d["turns"][t]
            history = sessions[(d["dialogue_id"], v)].history
            new_turn = Turn(raw=turn["raw"], normalized=normalized(v, turn, norm_out), style=turn["style"])
            p = build_prompt(history + [new_turn])  # what step() will give the tracker
            run_key[(d["dialogue_id"], v)] = key(p)
            prompts[key(p)] = p
        todo = [{"id": i, "input": p} for i, p in prompts.items() if i not in CACHE]
        print(f"turn {t + 1}: {len(active)} dialogue runs, {len(prompts)} distinct tracker prompts, "
              f"{len(todo)} to generate", flush=True)
        if todo:
            from eval_generator import generate, load_model

            if model is None:
                tok, model = load_model(model_id("tracker"))
                if args.int8:
                    model = int8_weights(model)
            CACHE.update(generate(tok, model, todo, args.batch, MAX_NEW_TOKENS, cache_path))
        for d, v in active:
            turn = d["turns"][t]
            s = sessions[(d["dialogue_id"], v)]
            CURRENT.update(variant=v, turn=turn)
            _, debug = s.step(turn["raw"])
            records.append({"dialogue_id": d["dialogue_id"], "variant": v, "turn": t, "style": turn["style"],
                            "raw": turn["raw"], "normalized": debug["normalized"],
                            "tracker": CACHE[run_key[(d["dialogue_id"], v)]],
                            "state": s.state.to_dict(), "gold_state": turn["gold_state"]})

    finals = []
    for d in gold:
        last = d["turns"][-1]
        for v in args.variants:
            s = sessions[(d["dialogue_id"], v)]
            ranked = ev.recommend(s.state, normalized(v, last, norm_out), args.k)
            finals.append({"dialogue_id": d["dialogue_id"], "variant": v, "style": last["style"], "ranked": ranked})

    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    with open(args.out.with_suffix(".ranked.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in finals)

    summary = {}
    for subset in ("all", "singlish", "english"):
        print(f"\n{subset} turns / dialogues")
        for v in args.variants:
            rows = [r for r in records if r["variant"] == v and subset in ("all", r["style"])]
            fin = [f for f in finals if f["variant"] == v and subset in ("all", f["style"])]
            ts, rs = turn_scores(rows), rank_scores(fin, by_id, args.k)
            summary[f"{subset}/{v}"] = {"tracker": ts, "ranking": rs}
            print(f"  {v:10s} turns {ts['n']:3d}  joint {ts['joint']:.3f}  slot acc {ts['slot_acc']:.3f}  "
                  f"slot F1 {ts['slot_f1']:.3f}  landmark {ts['landmark']:.3f}  full state {ts['full_state']:.3f}"
                  f"  | {rs['n']:2d} dialogues  P@{args.k} {rs['P@K']:.3f}  strict {rs['strict P@K']:.3f}  "
                  f"nDCG@{args.k} {rs['nDCG@K']:.3f}  unjudged {rs['unjudged']}")
    if "normalizer" in args.variants and "raw" in args.variants:
        pairs = {}
        for r in records:
            pairs.setdefault((r["dialogue_id"], r["turn"]), {})[r["variant"]] = r
        only = {"normalizer": 0, "raw": 0}
        for p in pairs.values():
            right = {v: all(same(p[v]["state"]["slots"][s], p[v]["gold_state"]["slots"][s]) for s in SLOTS)
                     for v in only}
            for v in only:
                only[v] += right[v] and not right["raw" if v == "normalizer" else "normalizer"]
        print(f"\nturns with all six slots right only with the normalizer: {only['normalizer']}, "
              f"only without it: {only['raw']}")
        summary["only_right"] = only
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("states ->", args.out)


if __name__ == "__main__":
    main()
