"""Shared Qwen2.5-1.5B + LoRA adapter used by the tracker [TRACK] and generator [REPLY].

Loaded once, lazily, from the Hugging Face Hub (MAKAN_BASE_LLM + MAKAN_LORA_ADAPTER).
Needs requirements-ml.txt. The prompt text itself is built by tracker/hf.py and
generator/hf.py - training data must be built with the same functions.
"""

from __future__ import annotations

from functools import lru_cache

from src.config import BASE_LLM, model_id

MAX_NEW_TOKENS = 256


@lru_cache(maxsize=1)
def _load():
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = model_id("tracker")
    tok = AutoTokenizer.from_pretrained(adapter)  # adapter repo ships the tokenizer used in training
    # bfloat16 on CPU too: half the RAM of float32 (~3 GB). No device_map on CPU: "auto" offloads
    # layers to disk when RAM is short, and the adapter then loads onto them as a silent no-op.
    device_map = "auto" if torch.cuda.is_available() else None
    base = AutoModelForCausalLM.from_pretrained(BASE_LLM, torch_dtype=torch.bfloat16, device_map=device_map)
    model = PeftModel.from_pretrained(base, adapter).eval()
    return tok, model


def complete(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Greedy completion of one user message through the chat template."""
    import torch

    tok, model = _load()
    # return_dict=True: the transformers 5 default, asked for explicitly so 4.x behaves the same
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
