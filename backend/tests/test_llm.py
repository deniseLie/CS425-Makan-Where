"""llm.complete() with fake torch / transformers / peft: one base model, one adapter per component."""

import contextlib
import sys
import types

from src import llm


class FakeEnc(dict):
    def to(self, device):
        return self


class FakeTok:
    pad_token_id = eos_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return FakeEnc(input_ids=types.SimpleNamespace(shape=(1, 3)))

    def decode(self, ids, skip_special_tokens):
        return ids  # FakePeft.generate puts the active adapter's repo here


class FakePeft:
    loads = []  # (call, repo, adapter_name)

    def __init__(self):
        self.repos, self.active, self.device = {}, None, "cpu"

    @classmethod
    def from_pretrained(cls, base, repo, adapter_name):
        cls.loads.append(("from_pretrained", repo, adapter_name))
        model = cls()
        model.repos[adapter_name] = repo
        return model

    def load_adapter(self, repo, adapter_name):
        self.loads.append(("load_adapter", repo, adapter_name))
        self.repos[adapter_name] = repo

    def eval(self):
        return self

    def set_adapter(self, name):
        self.active = name

    def generate(self, **kwargs):
        return {(0, slice(3, None)): self.repos[self.active]}


def test_each_component_runs_its_own_adapter(monkeypatch):
    bases = []
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False),
                                       bfloat16="bf16", no_grad=contextlib.nullcontext)
    fake_tf = types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=lambda *a, **k: bases.append(a) or "base"),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda repo: FakeTok()))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=FakePeft))
    monkeypatch.setattr(llm, "_model", None)
    monkeypatch.setattr(llm, "_toks", {})
    monkeypatch.setattr(FakePeft, "loads", [])
    monkeypatch.setenv("MAKAN_TRACKER_ADAPTER", "team/track")
    monkeypatch.setenv("MAKAN_GENERATOR_ADAPTER", "team/reply")

    assert llm.complete("[TRACK] ...", "tracker") == "team/track"
    assert llm.complete("[REPLY] ...", "generator") == "team/reply"
    assert llm.complete("[TRACK] ...", "tracker") == "team/track"
    assert len(bases) == 1
    assert FakePeft.loads == [("from_pretrained", "team/track", "tracker"), ("load_adapter", "team/reply", "generator")]
