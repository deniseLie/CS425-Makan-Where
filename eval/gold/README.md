# Gold test dialogues

Hand-labelled dialogues for evaluating the normalizer, tracker and recommender.
Store them as **JSONL**: one dialogue per line, files named `gold_<annotator>_<nn>.jsonl`
(e.g. `gold_nichole_01.jsonl`).

## Per dialogue

| field | type | meaning |
|---|---|---|
| `dialogue_id` | str | unique, e.g. `"d001"` |
| `annotator` | str | who labelled it |
| `turns` | list | the user turns, in order (see below) |
| `relevant_place_ids` | list[str] | every `place_id` in `data/master/places.csv` that would be a correct final recommendation for this user (order = most relevant first) |
| `relevance` | object, optional | graded judgments `{place_id: 0 \| 1 \| 2}` for the judged pool (below). When present, `eval_recommender.py` uses it: P@K / hit@K count grade ≥ 1, strict P@K grade 2, nDCG@K uses the grades as gains. `relevant_place_ids` then lists the grade ≥ 1 places, grade 2 first |

## Per user turn

| field | type | meaning |
|---|---|---|
| `raw` | str | exactly what the user typed (Singlish allowed) |
| `english_version` | str | standard-English rewrite - gold target for the normalizer |
| `style` | `"singlish"` \| `"english"` | gold label for the style detector |
| `gold_state` | object | the full `DialogueState` **after** this turn, in `DialogueState.to_dict()` format |

`gold_state` is cumulative: it is the state the tracker + critiques should have produced
after reading every turn up to and including this one. Slots use the same values as
`src/schemas.py`: `"UNKNOWN"` (not mentioned yet), `"ANY"` (no preference), or a value.
Values are **place-list values, after translation** (`src/tracker/values.py`): `"korean"`,
`"downtown core"` for "bugis side", `"<=10"`, `"19:00"`; not the tracker's raw `"Korean"`,
`"Bugis"`, `"below $10"`, `"7pm"`.

You do not label system replies - the policy decides those, and the tracker is evaluated on the
state after each user turn.

## Judging relevance (pooling)

38,000 places can't all be judged, so judge a pool: `python eval/eval_recommender.py --pool pool.jsonl`
writes, per dialogue, the union of every compared ranking's top places (hybrid and rating-only to
depth 5, each ablation to depth 3), shuffled and without system names, with the details a judge
needs. Every top 3 that is scored is inside the pool, so P@3 and nDCG@3 are fully judged. Grade each
place on its own, on an absolute scale:

| grade | meaning |
|---|---|
| 2 | great pick: meets every stated preference and fits the specific request (serves the dish asked for, close to the spot named), nothing makes it a poor choice |
| 1 | acceptable: meets the stated preferences, a reasonable place to send the user, but a weaker fit (generic place when a dish was asked, edge of the radius, ...) |
| 0 | not a good recommendation: breaks a stated preference, is not a place for that meal (caterer, canteen, drinks / dessert only), or doesn't serve what was asked |

An unknown rating, price estimate or opening hours is neutral, not a reason for 0.

## Files

| file | dialogues | made by |
|---|---|---|
| `gold_claude_01.jsonl` | 52 (27 Singlish, 25 English; 1-4 turns) | **Draft by Claude, for the team to review.** Dialogues written by Claude (not real users); `gold_state` built with the backend's own value tables (`aliases.canonical`, `track_landmark`, `parse_budget`); places judged blind by Claude subagents from the `--pool` file with the rubric above (467 places, plus 8 the hybrid ranker put in its top 3 when ranking on the normalizer's output, `--query-from`). Dialogues were adjusted so every final state has at least 3 candidates. Replace `annotator` when a team member re-checks a dialogue. |

## Example (pretty-printed; in the file it is one line)

```json
{
  "dialogue_id": "d001",
  "annotator": "denise",
  "turns": [
    {
      "raw": "I want eat something leh",
      "english_version": "I want to eat something.",
      "style": "singlish",
      "gold_state": {
        "slots": {"cuisine": "UNKNOWN", "area": "UNKNOWN", "budget": "UNKNOWN",
                  "dietary": "UNKNOWN", "setting": "UNKNOWN", "time": "UNKNOWN"},
        "excluded_place_ids": [], "radius_km": 2.0, "budget_max": null
      }
    },
    {
      "raw": "korean lah, bugis side",
      "english_version": "Korean, in the Bugis area.",
      "style": "singlish",
      "gold_state": {
        "slots": {"cuisine": "korean", "area": "downtown core", "budget": "UNKNOWN",
                  "dietary": "UNKNOWN", "setting": "UNKNOWN", "time": "UNKNOWN"},
        "excluded_place_ids": [], "radius_km": 2.0, "budget_max": null
      }
    },
    {
      "raw": "too far leh",
      "english_version": "That is too far.",
      "style": "singlish",
      "gold_state": {
        "slots": {"cuisine": "korean", "area": "downtown core", "budget": "UNKNOWN",
                  "dietary": "UNKNOWN", "setting": "UNKNOWN", "time": "UNKNOWN"},
        "excluded_place_ids": [], "radius_km": 1.0, "budget_max": null
      }
    }
  ],
  "relevant_place_ids": ["P002", "P001", "P003"]
}
```

Load with:

```python
import json
from src.schemas import DialogueState
for line in open("eval/gold/gold_denise_01.jsonl", encoding="utf-8"):
    d = json.loads(line)
    states = [DialogueState.from_dict(t["gold_state"]) for t in d["turns"]]
```
