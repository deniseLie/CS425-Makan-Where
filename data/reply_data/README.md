# reply_data/

`[REPLY]` training data for the reply generator (Qwen2.5-1.5B + LoRA, task `[REPLY]`). It is fully
**synthetic**: template-made user turns, made-up places, policy actions from our own code, and
replies written by Claude (Claude Code subagents acting as the teacher model, no API).

| file | rows | contents |
|---|---|---|
| `reply_train.jsonl` | 1,363 | training rows: `input` = `generator.hf.build_prompt(action, style, user_text)`, `output` = the reply. Same row format as the `[TRACK]` files (`id`, `task`, `source`, `style`, `input`, `output`), so the two can be mixed for the shared-adapter ablation. |
| `reply_dev.jsonl` | 276 | the same for dev |
| `replies_train.jsonl`, `replies_dev.jsonl` | 1,363 / 276 | the full rows the training rows are built from: `id`, `style`, `user_text`, `action`, `state`, `reply` |

| | ask | recommend | fallback with places | nothing found |
|---|---|---|---|---|
| train, Singlish / English | 313 / 225 | 149 / 149 | 242 / 87 | 141 / 57 |
| dev, Singlish / English | 82 / 33 | 39 / 40 | 26 / 15 | 28 / 13 |

## How it was made

Code is in `backend/src/generator/`. Commands run from `backend/`.

1. **Turns and actions** (`drafts.py`). The synthetic `[TRACK]` dialogues are replayed through the
   real policy: each user turn's gold slots stand in for the tracker, are mapped to backend values
   (`tracker.values.place_value`, the place catalogue), critiques are applied with `apply_tracker_output`,
   and `policy.decide` + the recommender pick the action. Values the catalogue lacks are swapped
   in both the state and the user's text. Turns that change nothing are skipped, and so are
   dialogues with a dietary need the recommender can't filter. Rows are capped per
   (action, slot, style) to keep the mix balanced.
   ```bash
   python -m src.generator.drafts ../data/synthetic_data/dialogues_synth_train.jsonl --out ../data/reply_data/drafts_train.jsonl --max-per-bucket 150
   python -m src.generator.drafts ../data/synthetic_data/dialogues_synth_dev.jsonl --out ../data/reply_data/drafts_dev.jsonl --max-per-bucket 40
   ```
2. **Replies** (`teacher.py`). The drafts were split into 8 chunks. Eight Claude Code subagents each
   wrote one reply per turn following `teacher.SYSTEM` and `teacher.EXAMPLES` (instructions below),
   ran the checks on their chunk, and rewrote failures and over-used phrases.
   ```bash
   python -m src.generator.teacher prepare JOB ../data/reply_data/drafts_train.jsonl ../data/reply_data/drafts_dev.jsonl
   python -m src.generator.teacher check JOB 03      # per chunk, by each worker
   python -m src.generator.teacher merge JOB         # -> replies_train.jsonl, replies_dev.jsonl
   ```
3. **Filter** (`checks.py`). `check_reply` rejects a reply that names a place the action didn't pick
   or leaves one out, skips an option, gives a price / distance / rating / time that isn't in the
   action or the user's text, doesn't say what a fallback loosened, has the wrong style, leaks
   prompt fragments, or runs too long. All 1,639 replies pass.
   ```bash
   python -m src.generator.checks ../data/reply_data/replies_train.jsonl   # -> .kept.jsonl + .rejected.jsonl
   ```
4. **Training rows** (`drafts.py --kept`). Rerun this whenever `build_prompt` changes.
   ```bash
   python -m src.generator.drafts --kept ../data/reply_data/replies_train.kept.jsonl --out ../data/reply_data/reply_train.jsonl
   ```

## Caveats

- **Built from the old synthetic data.** The turns come from `data/synthetic_data/` as of commit
  `60a8c00`, before `678db39` rebuilt it. The `SYN…` ids point at that version
  (`git show 60a8c00:data/synthetic_data/dialogues_synth_train.jsonl`); rerunning step 1 on today's
  files gives different turns.
- **Fake places.** Every action uses the 21 made-up places of the old `places_sample.csv`
  (`git show af5a912^:data/master/places_sample.csv`; lower-case areas such as `bugis`). Rebuild all
  four steps once `data/master/places.csv` has cuisine, halal, price and rating.
- **Templated users.** The user turns come from a handful of templates per slot, so they are far
  less varied than real users. Evaluate the trained generator on hand-written dialogues
  (`eval/gold/`), not on this data.
- **Style check.** These replies were checked with the old keyword heuristic, which misses Singlish
  that relies only on grammar ("also ok", "can!"), so the Singlish replies lean on particles.
  `check_reply` now asks the SingBERT style classifier instead.
- **Policy quirks visible in the data** (left as they are):
  - After "too ex", a fallback that relaxes the budget can offer a pricier place.
  - `relaxed_slot` names only the last slot relaxed, though earlier ones were dropped too.
  - "Too far" has no effect when the area is "anywhere", since there's no reference point.
- 14 no-op turns ("ok can"-style rejections with nothing to reject, revisions to the same value)
  were removed after the replies were written.

## Instructions given to the subagents

Each subagent got its chunk number and these instructions (paths shortened):

```text
You are drafting bot replies that become [REPLY] training targets for "Makan Where?", a
Singlish-aware restaurant chatbot for Singapore. You ARE the teacher model: write every reply
yourself, with your own judgment. Do not call any LLM API, and do not write scripts that generate
or template the replies.

1. Read backend/src/generator/teacher.py first. Its SYSTEM string is your instruction set; follow it
   exactly. Its EXAMPLES show input -> reply (the input text is what describe() produces).
2. Input: JOB/chunk_NN.jsonl, ~207 lines of {"id", "prompt"}. `prompt` describes the turn: Style,
   User's last message, Action, and Options or Places.
3. Output: JSONL lines {"id": "...", "reply": "..."}, one per id, covering every id, written to
   JOB/replies_NN_1.jsonl, replies_NN_2.jsonl, ... (about 50 rows per file). Each line must be valid
   JSON. Touch only your own chunk's files; edit nothing in the repo.
4. Run: python -m src.generator.teacher check JOB NN. It reports bad JSON, missing ids, and replies
   failing the automatic checks with failure codes. Fix each failure by writing the corrected reply
   to replies_NN_fix1.jsonl (later files override earlier ones; use fix2, fix3 for later rounds).
   Repeat until 0 missing and 0 failing.

What the checks enforce, beyond SYSTEM:
- Every listed place name, spelled as listed; no other place from the catalogue.
- Any $ amount, distance, star rating, "N min" or clock time must appear in the listed details or
  the user's message. Never write walking times.
- ask: the reply contains "?" and names every option (an option "none" may be skipped).
- fallback with places: say in words which preference was loosened, outside the place names.
- Style is checked by a keyword heuristic. Singlish replies need at least one marker (lah, leh, lor,
  hor, ah, sia, meh, liao, mah, aiyo, shiok, makan, dabao, walao, wah, "also can", "got ... or not",
  "can or not", "where got", "eat what", "<word> side"); English replies must contain none of them.
- No markdown, braces or place IDs; at most 80 words.

Quality bar (this trains the model, so quality matters more than speed):
- Vary openings and sentence shapes; don't reuse a stock phrase more than a handful of times. Mix
  numbered lists and prose for recommendations.
- Singlish should read like a real Singaporean texting a friend: natural particles and Singlish
  grammar, not in every sentence, never a caricature. English: friendly and concise.
- React to the user's last message where it matters (complaints, a preference they just gave)
  without repeating it word for word.
- For "nothing found", name no places; suggest another area or cuisine (or loosening a dietary
  need if the user stated one).
```
