# Makan Where? 🍜

A Singlish-aware conversational restaurant recommender for Singapore.
Users chat the way they normally would ("korean lah, bugis side", "too far leh") and the
system asks follow-up questions or recommends places, answering in the same style.

University NLP project, 4 people. This repo is currently a **working stub pipeline**
with a FastAPI backend and a React chat frontend: every model component sits behind a fixed
function signature, and an environment variable switches it from the stub to the trained model
on the Hugging Face Hub. Full project plan: [docs/plan.md](docs/plan.md).

## Pipeline (one user turn)

Paths are under `backend/`.

```
raw text
  → normalize        Flan-T5: Singlish → English                     (src/normalizer)
  → detect_style     SingBERT: "singlish" | "english"                (src/style)
  → track            Qwen2.5-1.5B + LoRA, task [TRACK]: reads the    (src/tracker)
                     whole dialogue, outputs the FULL slot state as JSON
  → translate        code: tracker values → place-list values        (src/tracker/values.py)
  → apply critiques  code: reject / too_far / too_expensive /        (src/policy/critiques.py)
                     not_interested
  → policy.decide    rules: ask or recommend                         (src/policy/rules.py)
  → recommender      filter + hybrid ranking over places.csv         (src/recommender)
                     (only when recommending)
  → generate         same Qwen + LoRA, task [REPLY], conditioned on  (src/generator)
                     the policy action + style tag
  → guardrail        reply names a place not in action.places → template reply
  → reply
```

`backend/src/pipeline.py` wires this together in `MakanSession.step(raw_text) -> (reply, debug)` and logs
every turn as one JSON line in `backend/logs/<session_id>.jsonl`.

### Slots

| slot | tier | behaviour |
|---|---|---|
| `cuisine`, `area` | required | asked first (max 3 questions per dialogue) |
| `budget` | hard filter | applied if stated, never asked |
| `dietary`, `setting`, `time` | soft | default ANY; asked only to narrow >10 candidates |

Each slot is `"UNKNOWN"` (not mentioned), `"ANY"` (no preference) or a value string
(`"korean"`, `"downtown core"`, `"halal"`, `"<=8"`, ...). All shared types live in
[`backend/src/schemas.py`](backend/src/schemas.py) - that file is the contract.

The tracker writes values the way the user said them; `src/tracker/values.py` translates them
into the values the place list uses, right after `track()`:

| slot | tracker value → place-list value |
|---|---|
| `cuisine` | lowercase (`Western` → `western`); a dish → its cuisine (`chicken rice` → `chinese`, `nasi lemak` → `malay`, `roti prata` → `indian`) |
| `area` | planning area → lowercase (`Ang Mo Kio` → `ang mo kio`); landmark → its planning area (`Bugis` → `downtown core`, `NUS` → `queenstown`); a region stays a region (`east`) |
| `budget` | `below $5` → `<=5`, `below $10` → `<=10`, `$10-20` → `<=20`, `above $20` → `ANY` |
| `dietary` | `halal`, `vegetarian`, `no beef` stay; `vegan` → `vegetarian`, `no pork` → `halal` |
| `setting` | `hawker`, `aircon`, `cafe`, `food court` |
| `time` | meal words stay (`supper` → `late night`); exact times → `HH:MM` (`7pm` → `19:00`). `meal_of("19:30")` → `dinner` |

Values already in place-list form are left alone; anything not in the tables is lowercased and
passed through.

## Setup & run

Needs Python 3.10+ and Node 18+ (tested with 3.12 and 22).

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r backend/requirements.txt
cd frontend && npm install && cd ..
```

Development (two terminals):

```bash
cd backend && uvicorn src.api:app --reload --port 8000    # API on :8000
cd frontend && npm run dev                                # UI on http://localhost:5173 (proxies /api)
```

Other commands, from `backend/`:

```bash
pytest                 # run tests
python -m src.pipeline # plain terminal chat, no UI
```

One-process mode (what a deployment runs): `cd frontend && npm run build`, then start uvicorn as
above and open http://127.0.0.1:8000 - the backend serves `frontend/dist` itself.

### API

| method | path | body / result |
|---|---|---|
| GET | `/api/health` | which implementation (stub / hf) and model each component runs |
| POST | `/api/sessions` | → `{session_id}` |
| POST | `/api/sessions/{id}/messages` | `{message}` → `{reply, debug}` (`debug` = state, action, tracker output) |
| POST | `/api/sessions/{id}/reset` | clears the dialogue |

Sessions are kept in memory, so a backend restart starts everyone afresh (the UI recovers by
creating a new session).

## Interfaces and owners

| module | function(s) | owner | stub behaviour now |
|---|---|---|---|
| `backend/src/normalizer` | `normalize(text: str) -> str` | Varidhi | returns text unchanged |
| `backend/src/style` | `detect_style(text: str) -> str` | Varidhi | particle/keyword heuristic |
| `backend/src/tracker` | `track(history: list[Turn], prev: DialogueState) -> TrackerOutput` | Nichole | phrase tables from `values.py` + regex, carries over prev slots |
| `backend/src/policy` | `decide(state, n_asked, recommender, query_norm="", user_latlon=None) -> PolicyAction`<br>`apply_tracker_output(prev, out) -> DialogueState` | Nichole | real rules (see `rules.py`) |
| `backend/src/recommender` | `candidates(state, user_latlon=None) -> list[dict]`<br>`rank(state, query_norm, user_latlon, k=3) -> list[RankedPlace]`<br>`load_places()`, `place_names()` | Yuhanna | pandas filter on `data/master/places.csv`, score = rating |
| `backend/src/generator` | `generate(action: PolicyAction, style: str, user_text: str) -> str`<br>`template_reply(action, style)` | Denise | template replies in both styles |
| `backend/src/pipeline.py` | `MakanSession.step(raw_text) -> (reply, debug)` | Denise | - |

| `backend/src/api.py` | HTTP API for the frontend | Denise | - |

Policy rules: required slot missing and `n_asked < 3` → ask it with the top-3 values among
candidates; else if >10 candidates and `n_asked < 3` → ask the unknown soft slot with the highest
value entropy; else if no candidates → relax `setting → time → budget` and return
`fallback_recommend`; else recommend the top 3.

Critiques: `reject` → add to `excluded_place_ids`; `too_far` → `radius_km *= 0.5`;
`too_expensive` → `budget_max *= 0.7` (or 15 if no budget yet); `not_interested` ("dun want
this one") → the pipeline sets `reject` to the top pick of the last recommendation. The radius is
measured from the user's location if known, otherwise from the centre of the chosen area.

The `[TRACK]` prompt and answer follow the tracker dataset's layout: `User:` / `Normalized:` /
`System:` lines after `[TRACK]`, and an answer of the six slots plus `"feedback"` (`none`,
`too_far`, `too_expensive`, `not_interested`). See `backend/src/tracker/hf.py`.

## Plugging in a trained model (Hugging Face Hub)

Each model module has `stub.py` and `hf.py`. `hf.py` already loads the model from the Hub; you
only choose it with environment variables (see [`backend/src/config.py`](backend/src/config.py)):

| component | switch | model repo variable | expected model |
|---|---|---|---|
| normalizer | `MAKAN_NORMALIZER=hf` | `MAKAN_NORMALIZER_MODEL` | Flan-T5 seq2seq; inputs prefixed with `normalizer.hf.PREFIX` |
| style | `MAKAN_STYLE=hf` | `MAKAN_STYLE_MODEL` | text classifier with `id2label = {0: "english", 1: "singlish"}` |
| tracker | `MAKAN_TRACKER=hf` | `MAKAN_LORA_ADAPTER` | LoRA adapter (+ tokenizer) for `MAKAN_BASE_LLM` |
| generator | `MAKAN_GENERATOR=hf` | `MAKAN_LORA_ADAPTER` | same adapter, `[REPLY]` task |

`MAKAN_BASE_LLM` defaults to `Qwen/Qwen2.5-1.5B-Instruct` and `MAKAN_LORA_ADAPTER` to the team's
adapter, [`makan-where-cs425/makan-qwen-lora`](https://huggingface.co/makan-where-cs425/makan-qwen-lora).
The repos are private, so you need `HF_TOKEN`: a read token from a member of the
`makan-where-cs425` organisation. Install the model packages with
`pip install -r backend/requirements-ml.txt`.

Put the settings in a `.env` file at the repo root (gitignored, never commit it). The backend
reads its `MAKAN_*` and `HF_*` lines on start; a variable set in the shell wins. The tests always
use the stubs. The current adapter was trained on `[REPLY]` only, so keep the tracker on its stub:

```bash
HF_TOKEN=hf_...
MAKAN_GENERATOR=hf
```

Then start the backend as usual (`cd backend && uvicorn src.api:app --port 8000`, or
`python -m src.pipeline`). The first start downloads Qwen (~3 GB) and the adapter.

The UI header shows `stub` / `hf` for each component so you can see what is running.

**Training and serving must use the same prompt format.** Build training files with the
functions in `hf.py`: `tracker.hf.build_prompt` / `build_target` for `[TRACK]`,
`generator.hf.build_prompt` for `[REPLY]`, `normalizer.hf.PREFIX` for Flan-T5. If you change a
format, change it there so serving follows.

Uploading after training (from a notebook):

```python
model.push_to_hub("our-team/makan-normalizer", private=True)
tokenizer.push_to_hub("our-team/makan-normalizer", private=True)
# Qwen: push only the LoRA adapter + tokenizer
peft_model.push_to_hub("our-team/makan-qwen-lora", private=True)
```

Rules when changing a module:

1. **Keep the signature.** Inputs and outputs are the types in `backend/src/schemas.py`. If the
   contract needs to change, raise it with the team first - training data, gold labels and logs
   all use it.
2. Keep `stub.py` working - the tests and anyone without a GPU use it.
3. Load models lazily (on first call, cached) so `import src.pipeline` stays fast.
4. **Add tests** in `backend/tests/`. Existing tests must still pass - they describe behaviour the
   rest of the pipeline relies on (e.g. "never return a non-halal place when dietary=halal").
   Mark tests that load a real model with `@pytest.mark.slow`.
5. Run `pytest` and try a few turns in the UI before committing.

## Repo layout

```
makan-where/
├── backend/
│   ├── src/
│   │   ├── schemas.py         shared data contract
│   │   ├── config.py          env vars: stub/hf switches, model repo ids, data paths
│   │   ├── api.py             FastAPI app (Denise)
│   │   ├── pipeline.py        one turn end to end (Denise)
│   │   ├── llm.py             shared Qwen + LoRA loader for tracker/generator
│   │   ├── normalizer/  style/    stub.py + hf.py (Varidhi)
│   │   ├── tracker/     policy/   (Nichole)
│   │   ├── recommender/           (Yuhanna)
│   │   └── generator/             (Denise)
│   ├── tests/                 data/places_sample.csv: made-up places the tests run on
│   ├── requirements.txt       API + stubs + tests
│   └── requirements-ml.txt    + torch/transformers/peft for MAKAN_*=hf
├── frontend/                  React + Vite chat UI with a pipeline-state panel
├── training/                  notebooks / scripts that train and push models to the HF Hub
├── data/                      [TRACK] datasets + master/places.csv (format in data/README.md)
├── eval/gold/                 hand-labelled test dialogues (format in eval/gold/README.md)
└── docs/plan.md               project plan
```

The recommender reads `data/master/places.csv` (NEA licensed eating establishments; set
`MAKAN_PLACES_CSV` to use another file). It has no `cuisine`, `halal`, `vegetarian`, `price`,
`rating` or `setting` values yet (see [data/README.md](data/README.md)), so for now a request with
a cuisine finds nothing; area and region requests work. The tests run on
`backend/tests/data/places_sample.csv` instead: 21 made-up places in the same columns, set by
`backend/tests/conftest.py`.
