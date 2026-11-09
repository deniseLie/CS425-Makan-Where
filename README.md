# Makan Where? 🍜

A Singlish-aware conversational restaurant recommender for Singapore.
Users chat the way they normally would ("korean lah, bugis side", "too far leh") and the
system asks follow-up questions or recommends places, answering in the same style.

University NLP project, 4 people. A FastAPI backend runs the pipeline on the team's trained
models from the Hugging Face Hub, with a React chat frontend; every model component sits behind
a fixed function signature. Full project plan: [docs/plan.md](docs/plan.md).

## Pipeline (one user turn)

Paths are under `backend/`.

```
raw text
  → normalize        Flan-T5: Singlish → English                     (src/normalizer)
  → detect_style     SingBERT: "singlish" | "english"                (src/style)
  → track            Qwen2.5-1.5B + tracker LoRA, task [TRACK]: the  (src/tracker)
                     whole dialogue in, the FULL slot state as JSON out
  → translate        code: tracker values → place-list values        (src/tracker/values.py)
  → apply critiques  code: reject / too_far / too_expensive /        (src/policy/critiques.py)
                     not_interested
  → policy.decide    rules: ask or recommend                         (src/policy/rules.py)
  → recommender      SQL filter + hybrid ranking over places.csv     (src/recommender)
                     (only when recommending)
  → generate         same Qwen + generator LoRA, task [REPLY], on    (src/generator)
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
pip install -r backend/requirements-ml.txt   # API + the models' packages (torch, transformers, peft)
cd frontend && npm install && cd ..
```

The models need `HF_TOKEN` in a `.env` file at the repo root (see
[Models](#models-hugging-face-hub)). The tests need only `backend/requirements.txt`.

Development (two terminals):

```bash
cd backend && uvicorn src.api:app --reload --port 8000    # API on :8000
cd frontend && npm run dev                                # UI on http://localhost:5173 (proxies /api)
```

Other commands, from `backend/`:

```bash
pytest                 # run tests (offline: no model is loaded)
python -m src.pipeline # plain terminal chat, no UI
```

One-process mode (what a deployment runs): `cd frontend && npm run build`, then start uvicorn as
above and open http://127.0.0.1:8000 - the backend serves `frontend/dist` itself.

### API

| method | path | body / result |
|---|---|---|
| GET | `/api/health` | which model each component runs |
| POST | `/api/sessions` | → `{session_id}` |
| POST | `/api/sessions/{id}/messages` | `{message}` → `{reply, debug}` (`debug` = state, action, tracker output) |
| POST | `/api/sessions/{id}/reset` | clears the dialogue |

Sessions are kept in memory, so a backend restart starts everyone afresh (the UI recovers by
creating a new session).

## Interfaces and owners

| module | function(s) | owner | implementation |
|---|---|---|---|
| `backend/src/normalizer` | `normalize(text: str) -> str` | Varidhi | Flan-T5 (`hf.py`) |
| `backend/src/style` | `detect_style(text: str) -> str` | Varidhi | SingBERT classifier (`hf.py`) |
| `backend/src/tracker` | `track(history: list[Turn], prev: DialogueState) -> TrackerOutput` | Nichole | Qwen + LoRA `[TRACK]` (`hf.py`), values translated by `values.py` |
| `backend/src/policy` | `decide(state, n_asked, recommender, query_norm="", user_latlon=None) -> PolicyAction`<br>`apply_tracker_output(prev, out) -> DialogueState` | Nichole | real rules (see `rules.py`) |
| `backend/src/recommender` | `candidates(state, user_latlon=None) -> list[dict]`<br>`rank(state, query_norm, user_latlon, k=3) -> list[RankedPlace]`<br>`load_places()`, `place_names()` | Yuhanna | SQL filter + hybrid ranking (see below) |
| `backend/src/generator` | `generate(action: PolicyAction, style: str, user_text: str) -> str`<br>`template_reply(action, style)` | Denise | Qwen + LoRA `[REPLY]` (`hf.py`); `template_reply` is the guardrail's fallback |
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

## Models (Hugging Face Hub)

Each model module's `hf.py` loads its model from the Hub, the first time it runs. Environment
variables pick the repo (see [`backend/src/config.py`](backend/src/config.py)):

| component | model repo variable | expected model |
|---|---|---|
| normalizer | `MAKAN_NORMALIZER_MODEL` | Flan-T5 seq2seq; inputs prefixed with `normalizer.hf.PREFIX` |
| style | `MAKAN_STYLE_MODEL` | text classifier with `id2label = {0: "english", 1: "singlish"}` |
| tracker | `MAKAN_TRACKER_ADAPTER` | LoRA adapter (+ tokenizer) for `MAKAN_BASE_LLM`, trained on `[TRACK]` |
| generator | `MAKAN_GENERATOR_ADAPTER` | LoRA adapter (+ tokenizer) for `MAKAN_BASE_LLM`, trained on `[REPLY]` |

The tracker and generator each have their own adapter on one base model: the base is loaded once,
and each adapter the first time its component runs. `MAKAN_BASE_LLM` defaults to
`Qwen/Qwen2.5-1.5B-Instruct`, and the adapters to the team's
[`makan-where-cs425/makan-qwen-track`](https://huggingface.co/makan-where-cs425/makan-qwen-track) and
[`makan-where-cs425/makan-qwen-reply`](https://huggingface.co/makan-where-cs425/makan-qwen-reply).
The normalizer defaults to
[`makan-where-cs425/makan-flan-normalizer`](https://huggingface.co/makan-where-cs425/makan-flan-normalizer)
(Flan-T5-base, ~1 GB, public), and the style classifier to
[`makan-where-cs425/makan-singbert-style-classifier`](https://huggingface.co/makan-where-cs425/makan-singbert-style-classifier)
(SingBERT, ~440 MB, public; labels `English` / `Singlish`). The Qwen repos are private, so you need `HF_TOKEN`: a read token from a member of the
`makan-where-cs425` organisation. Install the model packages with
`pip install -r backend/requirements-ml.txt`.

Put the settings in a `.env` file at the repo root (gitignored, never commit it). The backend
reads its `MAKAN_*` and `HF_*` lines on start; a variable set in the shell wins. With the team's
default repos, all it needs is the token:

```bash
HF_TOKEN=hf_...
```

Then start the backend as usual (`cd backend && uvicorn src.api:app --port 8000`, or
`python -m src.pipeline`). The first start downloads Qwen (~3 GB) and the adapter.

The UI header lists the components; hover over one to see the model it runs.

The tests never load a model: `backend/tests/conftest.py` puts the offline stand-ins from
`backend/tests/fakes.py` in place of `normalize`, `detect_style`, `track` and `generate`.

**Training and serving must use the same prompt format.** Build training files with the
functions in `hf.py`: `tracker.hf.build_prompt` / `build_target` for `[TRACK]`,
`generator.hf.build_prompt` for `[REPLY]`, `normalizer.hf.PREFIX` for Flan-T5. If you change a
format, change it there so serving follows.

Uploading after training (from a notebook):

```python
model.push_to_hub("our-team/makan-normalizer", private=True)
tokenizer.push_to_hub("our-team/makan-normalizer", private=True)
# Qwen: push only the LoRA adapter + tokenizer, one repo per task (training/qwen_denise.ipynb does this)
peft_model.push_to_hub("our-team/makan-qwen-reply", private=True)   # or makan-qwen-track
```

Rules when changing a module:

1. **Keep the signature.** Inputs and outputs are the types in `backend/src/schemas.py`. If the
   contract needs to change, raise it with the team first - training data, gold labels and logs
   all use it.
2. Keep `backend/tests/fakes.py` matching the interface - the tests run on it, not on the models.
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
│   │   ├── config.py          env vars: model repo ids, data paths
│   │   ├── api.py             FastAPI app (Denise)
│   │   ├── pipeline.py        one turn end to end (Denise)
│   │   ├── llm.py             Qwen loader: one base, the tracker's and generator's LoRA adapters
│   │   ├── normalizer/  style/    hf.py (Varidhi)
│   │   ├── tracker/     policy/   (Nichole)
│   │   ├── recommender/           (Yuhanna)
│   │   └── generator/             (Denise)
│   ├── tests/                 data/places_sample.csv: made-up places the tests run on;
│   │                          fakes.py: offline stand-ins for the models
│   ├── requirements.txt       API + tests
│   └── requirements-ml.txt    + torch/transformers/peft to run the models
├── frontend/                  React + Vite chat UI with a pipeline-state panel
├── training/                  notebooks / scripts that train and push models to the HF Hub
├── data/                      [TRACK] datasets, master/places.csv + aliases.csv (format in data/README.md)
├── eval/                      eval_{recommender,normalizer,style,generator,tracker}.py, latest numbers in RESULTS.md;
│                              gold/ = test dialogues with judged places (format in eval/gold/README.md)
└── docs/plan.md               project plan
```

### Recommender

`data/master/places.csv` is built by `data/build_places.py` from the NEA licensed eating
establishments, OpenStreetMap and the MUIS halal list (34,205 places: details and how to add OSM / MUIS / Google data in
[data/README.md](data/README.md)). Set `MAKAN_PLACES_CSV` to use another file.

1. **Canonical values.** Every slot value goes through the alias table `data/master/aliases.csv`
   (`kbbq` → `korean`, `amk` → `ang mo kio`, `bugis side` → `downtown core`, `tapau` → `takeaway`,
   small typos too). It agrees with `values.py` and only adds to it.
2. **Candidates** (`filters.py`): one parameterized SQL query (in-memory SQLite) built only from
   filled slots: cuisine, area / region, radius from the spot the user named ("near Bras Basah", kept in
   `state.landmark`), else the user, else the area's centre, dietary
   (halal, vegetarian, no beef), setting (`aircon` keeps places whose aircon is unknown),
   opening hours for `time` (unknown hours are kept), budget cap (checked
   against the middle of the price range, the "~$N" the reply shows), excluded places.
3. **Ranking** (`ranking.py`): `score = 0.35·content + 0.20·soft + 0.30·quality + 0.15·distance`
   - content: TF-IDF (words + character n-grams) similarity between the request (normalized text
     + slot values + alias expansions) and the place's name / cuisine / tags / description.
     `MAKAN_RECOMMENDER_CONTENT=embed` switches to sentence-transformer embeddings.
   - soft: partial credit for budget (`<=N`: share of the price band under N; `cheap` / `atas`:
     cheaper / pricier is better), aircon and opening hours when known.
   - quality: Bayesian-averaged rating (prior of 50 reviews at the catalogue mean).
   - distance: `exp(-km)` from the user or the area's centre.
   The top 3 never repeats a name or a building (no three stalls of one hawker centre).
4. `recommender.warm_up()` loads everything up front (first fit ~3 s, cached after that).

Evaluate against the gold dialogues (P@K, nDCG@K, hit@K, hard-constraint violation rate, plus
ablations of each score part and a rating-only baseline):

```bash
python eval/eval_recommender.py --k 3 --ablations
```

Data still missing (see data/README.md): ratings / real prices (Google), and cuisine for most
hawker stalls. Cuisine is known for 4,194 places and opening hours for 1,828 (OSM); halal for
1,247 (MUIS list; partial coverage). Prices are per-setting
estimates for now and the reply marks them "(est.)".

The tests run on `backend/tests/data/places_sample.csv` instead: 21 made-up places in the same
columns, set by `backend/tests/conftest.py`. `tests/test_recommender.py` also checks the real
places.csv, including a randomized hard-constraint check.
