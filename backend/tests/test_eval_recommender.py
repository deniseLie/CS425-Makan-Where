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


def test_graded_relevance():
    grades = {"a": 2, "b": 1, "c": 0}
    assert ev.ndcg_at_k(["a", "b", "c"], [], 3, grades) == pytest.approx(1.0)
    assert ev.ndcg_at_k(["b", "a", "c"], [], 3, grades) < 1.0
    assert ev.ndcg_at_k(["c"], [], 1, grades) == 0.0
    d = {"relevance": grades, "relevant_place_ids": ["a", "b"]}
    assert ev.relevant_ids(d) == {"a", "b"} and ev.relevant_ids(d, 2) == {"a"}
    assert ev.relevant_ids({"relevant_place_ids": ["a"]}) == {"a"}
    assert ev.relevant_ids({"relevant_place_ids": ["a"]}, 2) == set()


def dialogue(i, slots, english, relevant):
    state = DialogueState(slots=slots).to_dict()
    return {"dialogue_id": f"d{i}", "annotator": "test", "relevant_place_ids": relevant,
            "turns": [{"raw": english, "english_version": english, "style": "english", "gold_state": state}]}


def test_pool_holds_every_scored_top_k():
    gold = [dialogue(1, {"cuisine": "korean", "area": "downtown core"}, "Korean near Bugis, kimchi stew", []),
            dialogue(2, {"area": "downtown core"}, "Anything near Bugis", [])]
    pooled = ev.pool(gold, k=3)
    for d, p in zip(gold, pooled):
        ids = [x["place_id"] for x in p["places"]]
        assert len(ids) == len(set(ids)) and p["dialogue_id"] == d["dialogue_id"]
        state, query = ev.last_request(d)
        for _, weights, _ in ev.systems(3):
            assert set(ev.recommend(state, query, 3, weights)) <= set(ids)
    json.dumps(pooled, allow_nan=False)  # valid JSON: no NaN fields


def test_evaluate_on_sample(tmp_path):
    gold = [dialogue(1, {"cuisine": "korean", "area": "downtown core"}, "Korean near Bugis, kimchi stew", ["P001", "P002"]),
            dialogue(2, {"cuisine": "malay", "area": "east", "dietary": "halal"}, "Halal Malay food in the east", ["P012"])]
    path = tmp_path / "gold_test_01.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in gold), encoding="utf-8")
    m = ev.evaluate(ev.load_gold(tmp_path), k=3)
    assert m["hit@K"] == 1.0 and m["violation rate"] == 0.0 and 0 < m["P@K"] <= 1
    assert ev.evaluate(ev.load_gold(path), k=3, weights=ev.RATING_ONLY)["violation rate"] == 0.0
