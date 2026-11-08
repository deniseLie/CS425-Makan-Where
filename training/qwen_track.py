"""Train the tracker's LoRA adapter ([TRACK]) on Qwen2.5-1.5B-Instruct, evaluate it, and push it.

The script version of qwen_denise.ipynb with TASK = "track", for a GPU cluster or `!python` in
Colab. It trains on the MultiWOZ and synthetic [TRACK] files exactly as they are in data/, through
the chat template the way src/llm.py serves them, with the loss on the answer JSON only.

Run from anywhere in a checkout (needs backend/requirements-ml.txt and bitsandbytes):

    python training/qwen_track.py                     # train, then evaluate on dev
    python training/qwen_track.py --push              # ... and push to makan-where-cs425/makan-qwen-track
    python training/qwen_track.py --eval-only outputs/qwen_track/final --split test   # final numbers

Checkpoints go to --out (default outputs/qwen_track, gitignored); rerunning resumes from the newest.
--push needs HF_TOKEN with write access to the repo.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "backend"))

from src.schemas import SLOTS  # noqa: E402

FILES = {
    "train": ["woz_converted_data/dialogues_train.jsonl", "synthetic_data/dialogues_synth_train.jsonl"],
    "dev": ["woz_converted_data/dialogues_dev.jsonl", "synthetic_data/dialogues_synth_dev.jsonl"],
    "test": ["woz_converted_data/dialogues_test.jsonl"],  # the synthetic data has no test split
}
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
MAX_NEW_TOKENS = 256  # as src/llm.py


def read(split: str) -> list[dict]:
    rows = []
    for f in FILES[split]:
        with open(DATA / f, encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def chat(tok, prompt: str, answer: str | None = None) -> str:
    """The text llm.complete() gives the model: the prompt as the only user message."""
    msgs = [{"role": "user", "content": prompt}]
    if answer is None:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.apply_chat_template(msgs + [{"role": "assistant", "content": answer}], tokenize=False)


def encode(tok, rows: list[dict], max_len: int) -> list[dict]:
    """Token ids with labels -100 on the prompt, so only the answer is learned. Rows longer than
    max_len are dropped: cutting them would cut the answer or the turns its slots come from."""
    out = []
    for row in rows:
        prompt = tok(chat(tok, row["input"]), add_special_tokens=False)["input_ids"]
        full = tok(chat(tok, row["input"], row["output"]), add_special_tokens=False)["input_ids"]
        assert full[:len(prompt)] == prompt, row.get("id") or row.get("dialogue_id")
        if len(full) <= max_len:
            out.append({"input_ids": full, "attention_mask": [1] * len(full),
                        "labels": [-100] * len(prompt) + full[len(prompt):]})
    print(f"  {len(rows)} rows, dropped {len(rows) - len(out)} over {max_len} tokens")
    return out


def load_model(base: str, use_4bit: bool, adapter: str | None = None):
    """(tokenizer, model): a fresh LoRA to train, or a trained adapter to evaluate."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    gpu = torch.cuda.is_available()
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.float16) if use_4bit and gpu else None
    tok = AutoTokenizer.from_pretrained(adapter or base)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant, device_map="auto" if gpu else None,
                                                 torch_dtype=torch.float16 if gpu else torch.float32)
    if adapter:
        return tok, PeftModel.from_pretrained(model, adapter).eval()
    if quant:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    return tok, get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                 task_type="CAUSAL_LM", target_modules=TARGETS))


def train(args) -> tuple:
    import torch
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    rng = random.Random(args.seed)
    train_rows, dev_rows = read("train"), read("dev")
    rng.shuffle(train_rows)
    train_rows = train_rows[:args.limit]
    dev_rows = rng.sample(dev_rows, min(args.eval_rows, len(dev_rows)))
    print("train:", Counter(r["source"] for r in train_rows), "| dev loss rows:", len(dev_rows))

    tok, model = load_model(args.base, not args.no_4bit)
    model.print_trainable_parameters()
    tok.padding_side = "right"
    train_enc, dev_enc = encode(tok, train_rows, args.max_len), encode(tok, dev_rows, args.max_len)

    gpu = torch.cuda.is_available()
    steps = math.ceil(len(train_enc) / (args.batch * args.grad_accum)) * args.epochs
    every = max(10, steps // 10)
    targs = TrainingArguments(
        output_dir=str(args.out), num_train_epochs=args.epochs, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * steps)), per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}, fp16=gpu,
        optim="paged_adamw_8bit" if gpu and not args.no_4bit else "adamw_torch",
        logging_steps=max(1, every // 5), eval_strategy="steps", eval_steps=every, save_strategy="steps",
        save_steps=every, save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
        report_to="none", seed=args.seed,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_enc, eval_dataset=dev_enc,
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    print(f"{steps} optimizer steps, eval + checkpoint every {every}")
    trainer.train(resume_from_checkpoint=any(args.out.glob("checkpoint-*")) or None)

    final = args.out / "final"  # the best checkpoint by dev loss, which the trainer loaded at the end
    model.save_pretrained(final)
    tok.save_pretrained(final)
    print("saved", final)
    return tok, model


def generate(tok, model, prompts: list[str], batch: int) -> list[str]:
    """Greedy, as llm.complete(); batched with left padding."""
    import torch

    model.eval()
    tok.padding_side = "left"
    outs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch):
            enc = tok([chat(tok, p) for p in prompts[i:i + batch]], return_tensors="pt", padding=True,
                      add_special_tokens=False).to(model.device)
            gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
            outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [o.strip() for o in outs]


def parse_json(text: str) -> dict | None:
    try:
        d = json.loads(text[text.index("{"):text.rindex("}") + 1])
        return d if isinstance(d, dict) else None
    except ValueError:
        return None


def norm(v) -> str:
    return str(v).strip().lower()


def scores(rows: list[dict], preds: list[str]) -> dict:
    """Joint goal accuracy (all six slots right), slot accuracy, F1 over the slots the user filled
    (gold not UNKNOWN), per-slot accuracy, feedback accuracy and the invalid-JSON rate."""
    n = len(rows)
    joint = fb_ok = invalid = tp = fp = fn = 0
    per_slot = Counter()
    for row, pred in zip(rows, preds):
        gold, got = json.loads(row["output"]), parse_json(pred)
        if got is None:
            invalid += 1
            got = {}
        right = {s: norm(got.get(s, "UNKNOWN")) == norm(gold[s]) for s in SLOTS}
        joint += all(right.values())
        per_slot.update(s for s, ok in right.items() if ok)
        fb_ok += norm(got.get("feedback", "none")) == norm(gold["feedback"])
        for s, ok in right.items():
            g, p = norm(gold[s]), norm(got.get(s, "UNKNOWN"))
            tp += ok and g != "unknown"
            fp += not ok and p != "unknown"
            fn += not ok and g != "unknown"
    return {"n": n, "joint_acc": joint / n, "slot_acc": sum(per_slot.values()) / (n * len(SLOTS)),
            "slot_f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0, "feedback_acc": fb_ok / n,
            "invalid_json": invalid / n, **{f"acc_{s}": per_slot[s] / n for s in SLOTS}}


def evaluate(tok, model, split: str, n: int, batch: int, seed: int, out: Path) -> None:
    rows = read(split)
    for source in sorted({r["source"] for r in rows}):
        pool = [r for r in rows if r["source"] == source]
        sample = random.Random(seed).sample(pool, min(n, len(pool))) if n else pool
        preds = generate(tok, model, [r["input"] for r in sample], batch)
        print(f"[TRACK] {split} / {source}:", {k: round(v, 3) for k, v in scores(sample, preds).items()})
        with open(out / f"eval_track_{split}_{source}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps({"input": r["input"], "pred": p, "gold": r["output"]}, ensure_ascii=False) + "\n"
                         for r, p in zip(sample, preds))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct", help="must match MAKAN_BASE_LLM")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "qwen_track")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4, help="effective batch = batch x grad-accum")
    ap.add_argument("--max-len", type=int, default=1024, help="longer rows are dropped, not truncated")
    ap.add_argument("--no-4bit", action="store_true", help="plain LoRA instead of QLoRA (faster on a big GPU)")
    ap.add_argument("--limit", type=int, help="train on only this many rows (smoke test)")
    ap.add_argument("--eval-rows", type=int, default=400, help="dev rows for the loss during training")
    ap.add_argument("--eval-n", type=int, default=200, help="rows generated per source in the eval; 0 = all")
    ap.add_argument("--eval-batch", type=int, default=8)
    ap.add_argument("--split", choices=["dev", "test"], default="dev", help="split for the final eval")
    ap.add_argument("--eval-only", metavar="ADAPTER", help="evaluate this trained adapter (path or Hub id), no training")
    ap.add_argument("--push", nargs="?", const="makan-where-cs425/makan-qwen-track", metavar="REPO",
                    help="after training, push adapter + tokenizer to this private Hub repo")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        tok, model = load_model(args.base, not args.no_4bit, adapter=args.eval_only)
    else:
        tok, model = train(args)
    evaluate(tok, model, args.split, args.eval_n, args.eval_batch, args.seed, args.out)

    if args.push and not args.eval_only:
        model.push_to_hub(args.push, private=True, commit_message=f"[TRACK] adapter, {args.epochs:g} epoch(s)")
        tok.push_to_hub(args.push, private=True)
        print(f"pushed https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
