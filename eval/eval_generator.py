#!/usr/bin/env python3
"""Reply generator: share of dev replies that pass every automatic check.

Generates a reply for each [REPLY] dev row (data/reply_data/reply_dev.jsonl, 276 rows) with the
trained adapter on the base model, and runs it through generator.checks.check_reply, the filter
every teacher-written training reply passed (100%): only places from the action, every picked
place and ask option named, no invented price / distance / rating / time, a fallback that says
what it relaxed, the requested style (SingBERT), no leaked prompt fragments, at most 120 words.

Generation matches llm.complete(): the prompt as one user message through the adapter's chat
template, greedy, up to 256 new tokens. Prompts are batched (left padding) and the LoRA is merged
into the base for speed. On a GPU the model runs in float16. On CPU the weights stay in bfloat16
as the backend serves them (~3 GB) but the maths runs in float32 (each layer's weights are upcast
when used): bfloat16 matmuls are very slow on CPUs without native support, and float32 weights
need 6 GB. A laptop CPU takes about 40 minutes for the 276 rows; a Colab T4 a few minutes.

    python eval/eval_generator.py                      # all 276 dev rows
    python eval/eval_generator.py --limit 20           # quick check
    python eval/eval_generator.py --adapter makan-where-cs425/makan-qwen-shared   # the ablation

Writes every reply with its failures to outputs/eval/generator_dev.jsonl. Replies are cached per
batch in <out>.cache.jsonl, so an interrupted run resumes where it stopped (delete it to rerun).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from src.config import BASE_LLM, model_id  # noqa: E402

REPLY_DIR = ROOT / "data" / "reply_data"
KINDS = ("ask", "recommend", "fallback_recommend")


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _fp32_maths(model):
    """CPU: keep Linear / Embedding weights in bfloat16, run everything else in float32."""
    import torch
    import torch.nn.functional as F

    class UpcastLinear(torch.nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.weight = lin.weight  # bfloat16; stays shared with the embedding when tied
            self.bias = None if lin.bias is None else torch.nn.Parameter(lin.bias.detach().float())

        def forward(self, x):
            return F.linear(x, self.weight.float(), self.bias)

    for mod in list(model.modules()):
        for name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear):
                setattr(mod, name, UpcastLinear(child))
    for mod in model.modules():
        if isinstance(mod, (UpcastLinear, torch.nn.Embedding)):
            continue
        for p in mod.parameters(recurse=False):
            p.data = p.data.float()
        for name, buf in list(mod.named_buffers(recurse=False)):
            if buf.is_floating_point():
                setattr(mod, name, buf.float())
    model.get_input_embeddings().register_forward_hook(lambda m, i, o: o.float())
    return model


def load_model(adapter: str):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpu = torch.cuda.is_available()
    dtype = torch.float16 if gpu else torch.bfloat16
    base = AutoModelForCausalLM.from_pretrained(BASE_LLM, dtype=dtype, device_map="auto" if gpu else None)
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload().eval()
    if not gpu:
        model = _fp32_maths(model)
    tok = AutoTokenizer.from_pretrained(adapter)  # the adapter repo ships the tokenizer used in training
    tok.padding_side = "left"
    return tok, model


def generate(tok, model, rows: list[dict], batch: int, max_new_tokens: int, cache: Path) -> dict[str, str]:
    """{row id: reply}; replies already in `cache` are reused, new ones appended per batch."""
    import torch

    out = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r["pred"]
    todo = [r for r in rows if r["id"] not in out]
    print(f"{len(rows) - len(todo)} replies cached, {len(todo)} to generate", flush=True)
    texts = {r["id"]: tok.apply_chat_template([{"role": "user", "content": r["input"]}], tokenize=False,
                                              add_generation_prompt=True) for r in todo}
    todo.sort(key=lambda r: len(texts[r["id"]]))  # similar lengths per batch: less padding
    start = time.time()
    for b in range(0, len(todo), batch):
        ids = [r["id"] for r in todo[b:b + batch]]
        enc = tok([texts[i] for i in ids], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        new = dict(zip(ids, (t.strip() for t in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:],
                                                                 skip_special_tokens=True))))
        out |= new
        with open(cache, "a", encoding="utf-8") as f:
            f.writelines(json.dumps({"id": i, "pred": p}, ensure_ascii=False) + "\n" for i, p in new.items())
        print(f"  {min(b + batch, len(todo))}/{len(todo)} replies, {time.time() - start:.0f} s", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--adapter", default=None, help="LoRA adapter repo or path (default: MAKAN_GENERATOR_ADAPTER)")
    ap.add_argument("--split", default="dev", choices=["dev", "train"])
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows; 0 = all")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "eval" / "generator_dev.jsonl")
    args = ap.parse_args()

    from src.generator.checks import check_reply
    from src.schemas import PolicyAction

    adapter = args.adapter or model_id("generator")
    rows = read(REPLY_DIR / f"reply_{args.split}.jsonl")
    full = {r["id"]: r for r in read(REPLY_DIR / f"replies_{args.split}.jsonl")}  # action, style, user_text
    rows = rows[:args.limit] if args.limit else rows
    # rogue_place is judged against the made-up places the reply data uses (as in qwen_denise.ipynb)
    catalogue = sorted({p["name"] for f in ("replies_train.jsonl", "replies_dev.jsonl")
                        for r in read(REPLY_DIR / f) for p in r["action"]["places"]})

    print(f"{len(rows)} {args.split} rows | base {BASE_LLM} | adapter {adapter}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tok, model = load_model(adapter)
    preds = generate(tok, model, rows, args.batch, args.max_new_tokens, args.out.with_suffix(".cache.jsonl"))

    results, codes, buckets = [], Counter(), Counter()
    for r in rows:
        pred = preds[r["id"]]
        src = full[r["id"]]
        action = PolicyAction.from_dict(src["action"])
        failures = check_reply(pred, action, src["style"], src["user_text"], catalogue)
        codes.update({c for c, _ in failures})  # one count per reply and code
        buckets[(action.kind, src["style"], not failures)] += 1
        results.append({"id": r["id"], "kind": action.kind, "style": src["style"], "user_text": src["user_text"],
                        "pred": pred, "gold": r["output"], "failures": failures})

    with open(args.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in results)

    n = len(results)
    n_ok = sum(not r["failures"] for r in results)
    summary = {"n": n, "pass_rate": n_ok / n, "by_bucket": {}, "failures": dict(codes.most_common())}
    print(f"\n[REPLY] {args.split}: {n_ok}/{n} replies pass every check = {n_ok / n:.1%}")
    for style in ("singlish", "english"):
        ok = sum(buckets[(k, style, True)] for k in KINDS)
        tot = ok + sum(buckets[(k, style, False)] for k in KINDS)
        if tot:
            print(f"  {style:9s} {ok}/{tot} = {ok / tot:.1%}")
            summary["by_bucket"][style] = ok / tot
    for kind in KINDS:
        for style in ("singlish", "english"):
            ok, bad = buckets[(kind, style, True)], buckets[(kind, style, False)]
            if ok + bad:
                print(f"  {kind:19s} {style:9s} {ok}/{ok + bad}")
                summary["by_bucket"][f"{kind}/{style}"] = ok / (ok + bad)
    print("replies failing each check:", {c: f"{k} ({k / n:.1%})" for c, k in codes.most_common()})
    print("replies ->", args.out)
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
