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

You do not label system replies - the policy decides those, and the tracker is evaluated on the
state after each user turn.

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
        "slots": {"cuisine": "korean", "area": "bugis", "budget": "UNKNOWN",
                  "dietary": "UNKNOWN", "setting": "UNKNOWN", "time": "UNKNOWN"},
        "excluded_place_ids": [], "radius_km": 2.0, "budget_max": null
      }
    },
    {
      "raw": "too far leh",
      "english_version": "That is too far.",
      "style": "singlish",
      "gold_state": {
        "slots": {"cuisine": "korean", "area": "bugis", "budget": "UNKNOWN",
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
