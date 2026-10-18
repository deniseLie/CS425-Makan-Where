# Makan Where? 🍜

A Singlish-aware conversational restaurant recommender for Singapore.
Users chat the way they normally would ("korean lah, bugis side", "too far leh") and the
system asks follow-up questions or recommends places, answering in the same style.

University NLP project, 4 people. This repo is currently a **working stub pipeline**:
every component sits behind a fixed function signature, so each of us can swap in the real
model without touching anyone else's code.

## Pipeline (one user turn)

```
raw text
  → normalize        Flan-T5: Singlish → English                     (src/normalizer)
  → detect_style     SingBERT: "singlish" | "english"                (src/style)
  → track            Qwen2.5-1.5B + LoRA, task [TRACK]: reads the    (src/tracker)
                     whole dialogue, outputs the FULL slot state as JSON
  → apply critiques  code: reject / too_far / too_expensive          (src/policy/critiques.py)
  → policy.decide    rules: ask or recommend                         (src/policy/rules.py)
  → recommender      filter + hybrid ranking over places.csv         (src/recommender)
                     (only when recommending)
  → generate         same Qwen + LoRA, task [REPLY], conditioned on  (src/generator)
                     the policy action + style tag
  → guardrail        reply names a place not in action.places → template reply
  → reply
```

`src/pipeline.py` wires this together in `MakanSession.step(raw_text) -> (reply, debug)` and logs
every turn as one JSON line in `logs/<session_id>.jsonl`.

### Slots

| slot | tier | behaviour |
|---|---|---|
| `cuisine`, `area` | required | asked first (max 3 questions per dialogue) |
| `budget` | hard filter | applied if stated, never asked |
| `dietary`, `setting`, `time` | soft | default ANY; asked only to narrow >10 candidates |

Each slot is `"UNKNOWN"` (not mentioned), `"ANY"` (no preference) or a value string
(`"korean"`, `"bugis"`, `"halal"`, `"<=8"`, ...). All shared types live in
[`src/schemas.py`](src/schemas.py) - that file is the contract.

## Setup & run

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

pytest                 # run tests
python app.py          # Gradio chat UI at http://127.0.0.1:7860
python -m src.pipeline # plain terminal chat
```

The heavy dependencies for training / the real models are listed (commented out) at the bottom
of `requirements.txt`.

## Interfaces and owners

| module | function(s) | owner | stub behaviour now |
|---|---|---|---|
| `src/normalizer` | `normalize(text: str) -> str` | Varidhi | returns text unchanged |
| `src/style` | `detect_style(text: str) -> str` | Varidhi | particle/keyword heuristic |
| `src/tracker` | `track(history: list[Turn], prev: DialogueState) -> TrackerOutput` | Nichole | alias dict + regex, carries over prev slots |
| `src/policy` | `decide(state, n_asked, recommender, query_norm="", user_latlon=None) -> PolicyAction`<br>`apply_tracker_output(prev, out) -> DialogueState` | Nichole | real rules (see `rules.py`) |
| `src/recommender` | `candidates(state, user_latlon=None) -> list[dict]`<br>`rank(state, query_norm, user_latlon, k=3) -> list[RankedPlace]`<br>`load_places()`, `place_names()` | Yuhanna | pandas filter on `data/master/places_sample.csv`, score = rating |
| `src/generator` | `generate(action: PolicyAction, style: str, user_text: str) -> str`<br>`template_reply(action, style)` | Denise | template replies in both styles |
| `src/pipeline.py` | `MakanSession.step(raw_text) -> (reply, debug)` | Denise | - |

Policy rules: required slot missing and `n_asked < 3` → ask it with the top-3 values among
candidates; else if >10 candidates and `n_asked < 3` → ask the unknown soft slot with the highest
value entropy; else if no candidates → relax `setting → time → budget` and return
`fallback_recommend`; else recommend the top 3.

Critiques: `reject` → add to `excluded_place_ids`; `too_far` → `radius_km *= 0.5`;
`too_expensive` → `budget_max *= 0.7` (or 15 if no budget yet). The radius is measured from the
user's location if known, otherwise from the centre of the chosen area.

## How to replace a stub

1. **Keep the signature.** Inputs and outputs are the types in `src/schemas.py`. If you think the
   contract needs to change, raise it with the team first - training data, gold labels and logs
   all use it.
2. Put your implementation in a new file in your module, e.g. `src/tracker/qwen_lora.py`, and
   change the one import line in that module's `__init__.py`:
   ```python
   from .qwen_lora import track   # was: from .stub import track
   ```
   Keep `stub.py` around - it's handy for fast tests and as a fallback.
3. Load models lazily (on first call, cached) so `import src.pipeline` stays fast and the tests
   that don't need your model don't load it.
4. **Add tests** in `tests/test_<module>.py`. The existing tests must still pass - they describe
   behaviour the rest of the pipeline relies on (e.g. "never return a non-halal place when
   dietary=halal"). Mark slow model tests with `@pytest.mark.slow`.
5. Run `pytest` and try a few turns in `python app.py` before pushing.

## Repo layout

```
makan-where/
├── data/{raw,master,train}/   raw + large files gitignored; master/places_sample.csv is fake demo data
├── src/
│   ├── schemas.py             shared data contract
│   ├── normalizer/  style/    Varidhi
│   ├── tracker/     policy/   Nichole
│   ├── recommender/           Yuhanna
│   ├── generator/             Denise
│   └── pipeline.py            Denise
├── eval/gold/                 hand-labelled test dialogues (format in eval/gold/README.md)
├── notebooks/
├── tests/
├── app.py                     Gradio chat UI
└── requirements.txt
```

`data/master/places_sample.csv` contains 21 made-up places (realistic names/locations, not real
data) so the stub pipeline runs end to end.
