"""eval/eval_recommender.py metrics on the sample places."""

import importlib.util
import json

import pytest

from src.config import ROOT
from src.schemas import DialogueState

spec = importlib.util.spec_from_file_location("eval_recommender", ROOT / "eval" / "eval_recommender.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def test_metric_functions():
    assert ev.precision_at_k(["a", "b", "c"], {"a", "c"}, 3) == pytest.approx(2 / 3)
    assert ev.hit_at_k(["x", "y"], {"y"}, 2) == 1.0 and ev.hit_at_k(["x"], {"y"}, 1) == 0.0
    assert ev.ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)
    assert ev.ndcg_at_k(["b", "a"], ["a", "b"], 2) < 1.0


def test_evaluate_on_sample(tmp_path):
    def dialogue(i, slots, english, relevant):
        state = DialogueState(slots=slots).to_dict()
        return {"dialogue_id": f"d{i}", "annotator": "test", "relevant_place_ids": relevant,
                "turns": [{"raw": english, "english_version": english, "style": "english", "gold_state": state}]}

    gold = [dialogue(1, {"cuisine": "korean", "area": "downtown core"}, "Korean near Bugis, kimchi stew", ["P001", "P002"]),
            dialogue(2, {"cuisine": "malay", "area": "east", "dietary": "halal"}, "Halal Malay food in the east", ["P012"])]
    path = tmp_path / "gold_test_01.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in gold), encoding="utf-8")
    m = ev.evaluate(ev.load_gold(tmp_path), k=3)
    assert m["hit@K"] == 1.0 and m["violation rate"] == 0.0 and 0 < m["P@K"] <= 1
    assert ev.evaluate(ev.load_gold(path), k=3, use_stub=True)["violation rate"] == 0.0
