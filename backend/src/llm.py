"""Qwen2.5-1.5B with two LoRA adapters: the tracker's [TRACK] and the generator's [REPLY].

The base model (MAKAN_BASE_LLM) is loaded once, lazily; each adapter (MAKAN_TRACKER_ADAPTER,
MAKAN_GENERATOR_ADAPTER) is loaded onto it the first time its component calls complete().
Needs requirements-ml.txt. The prompt text itself is built by tracker/hf.py and
generator/hf.py - training data must be built with the same functions.
"""

from __future__ import annotations

import threading

from src.config import BASE_LLM, model_id

MAX_NEW_TOKENS = 256

_lock = threading.Lock()  # one completion at a time: set_adapter switches the shared model
_model = None             # PeftModel holding every adapter loaded so far, named by component
_toks: dict = {}          # component -> tokenizer from its adapter repo


def _load(component: str):
    """(tokenizer, model) with `component`'s adapter loaded. Call with _lock held."""
    global _model
    if component not in _toks:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter = model_id(component)
        if _model is None:
            # bfloat16 on CPU too: half the RAM of float32 (~3 GB). No device_map on CPU: "auto" offloads
            # layers to disk when RAM is short, and the adapter then loads onto them as a silent no-op.
            device_map = "auto" if torch.cuda.is_available() else None
            base = AutoModelForCausalLM.from_pretrained(BASE_LLM, torch_dtype=torch.bfloat16, device_map=device_map)
            _model = PeftModel.from_pretrained(base, adapter, adapter_name=component)
        else:
            _model.load_adapter(adapter, adapter_name=component)
        _model.eval()
        _toks[component] = AutoTokenizer.from_pretrained(adapter)  # adapter repo ships the tokenizer used in training
    return _toks[component], _model


def complete(prompt: str, component: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Greedy completion of one user message through the chat template, with `component`'s
    adapter ("tracker" or "generator") active."""
    import torch

    with _lock:
        tok, model = _load(component)
        model.set_adapter(component)
        # return_dict=True: the transformers 5 default, asked for explicitly so 4.x behaves the same
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
