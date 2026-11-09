# Evaluation results (9 Nov 2026)

The proposal's metrics for the recommender, normalizer, style detector and reply generator, run on
the models the backend serves today (`makan-flan-normalizer`, `makan-singbert-style-classifier`,
`makan-qwen-reply` on Qwen2.5-1.5B-Instruct) and the current `data/master/places.csv`. The
tracker's MultiWOZ and synthetic numbers are in `docs/presentation_draft.md` (from
`training/qwen_track.py`); the tracker on the gold dialogues, with and without the normalizer, is
under Normalizer below.

| Component | Test set | Metric | Ours | Baseline |
|---|---|---|---|---|
| Recommender | 52 gold dialogues | P@3 / nDCG@3 | **0.910 / 0.836** | rating only: 0.833 / 0.710 |
| Recommender | 52 gold dialogues | hit@3 / hard-constraint violations | 1.000 / 0% | 0.923 / 0% |
| Normalizer | SEAT test (held out in training, 161 pairs) | BLEU / chrF | **55.3 / 69.6** | copy input: 30.4 / 55.2 |
| Normalizer | gold Singlish turns (35) | BLEU / chrF | 13.2 / 41.3 | copy input: 10.7 / 43.6 |
| Tracker | gold Singlish turns (35) | joint goal accuracy | with the normalizer: 0.400 | **without it: 0.571** |
| Whole pipeline | 27 gold Singlish dialogues | P@3 / nDCG@3 of the final picks | with the normalizer: 0.642 / 0.583 | **without it: 0.827 / 0.761** |
| Style detector | dev (1,189 turns, 185 Singlish) | macro F1 | 0.904 | keywords: **0.930** |
| Style detector | gold turns (67, 35 Singlish) | macro F1 | **0.895** | keywords: 0.717 |
| Generator | dev (276 replies) | replies passing every check | **96.4%** (266/276) | teacher replies: 100% |

## Recommender

`python eval/eval_recommender.py --k 3 --ablations --by-style`

| ranking | P@3 | strict P@3 | nDCG@3 | hit@3 | violations |
|---|---|---|---|---|---|
| **hybrid** | **0.910** | **0.641** | **0.836** | 1.000 | 0% |
| rating only | 0.833 | 0.506 | 0.710 | 0.923 | 0% |
| without content | 0.821 | 0.532 | 0.715 | 0.942 | 0% |
| without soft | 0.910 | 0.603 | 0.818 | 1.000 | 0% |
| without quality | 0.897 | 0.571 | 0.795 | 1.000 | 0% |
| without distance | 0.917 | 0.615 | 0.820 | 1.000 | 0% |

- P@3 and hit@3 count places judged relevant (grade ≥ 1), strict P@3 only "great" ones (grade 2);
  nDCG@3 uses the grades 0 / 1 / 2 as gains.
- Hybrid beats rating-only by +0.077 P@3 (95% bootstrap CI +0.006 to +0.154) and +0.126 nDCG@3
  (CI +0.037 to +0.221), paired over the 52 dialogues; better nDCG@3 on 22 dialogues, worse on 17, tied on 13.
- Content similarity matters most: without it nDCG@3 falls to 0.715, close to rating-only. Every
  other part helps nDCG@3 a little; without distance P@3 is slightly higher (0.917) but nDCG@3 lower.
- Singlish dialogues 0.864 / 0.814 (rating only 0.765 / 0.626), English 0.960 / 0.860
  (0.907 / 0.801). The recommender reads the gold English rewrite, so this gap is about the
  requests, not the language: 15 of 27 Singlish dialogues ask for a specific dish, 4 of 25 English.
- Misses are mostly dish-level: "chicken rice near Tampines" also returns PUTIEN and a grilled-fish
  place, "char kway teow" a kway chap stall. The place text rarely names dishes, so content
  similarity often can only match the cuisine.
- The ranker reads only the latest message, as the pipeline does. "I want Korean BBQ" → "Is there
  any near Bugis?" ranks on the second message, so the dish is lost and a Korean dessert café
  makes the top 3. Passing the earlier turns' requests along would fix this.
- 0% violations is measured against the catalogue's own labels. The judges found a few catalogue
  errors that pass the filter: a Chinese beef-noodle shop tagged Japanese (returned for "Japanese in
  Bukit Timah"), a polytechnic canteen stall and a union staff canteen.

**Gold set.** None existed, so `eval/gold/gold_claude_01.jsonl` is a draft written by Claude for the
team to review: 52 dialogues (27 Singlish, 25 English; 41 with one turn, 11 with 2-4 turns,
including revisions, "too far" and "too expensive"), states built with the backend's value tables.
Relevance was judged by pooling (union of every ranking's top places, 507 places in all with the
ones added for the normalizer and tracker runs below) and blind grading by Claude subagents that saw
only the request and place details, never which system ranked a place (rubric in
`eval/gold/README.md`). Judging more places can only lower nDCG@3, since its best possible score
counts every good place known; it moved by at most 0.01. The proposal asks for team-judged places, so a
teammate should spot-check the grades before the numbers are final.

## Normalizer

`python eval/eval_normalizer.py` (translation quality), `python eval/eval_tracker.py` (its effect on
the pipeline)

**In short: the normalizer makes the bot worse for Singlish users, and the pipeline does better
without it.**

- The normalizer is meant to turn Singlish into proper English before the rest of the bot reads it.
  It handles the chatty parts fine ("lah", "leh", "got anot"), but it often mistranslates the words
  that matter most, the dish and the place: "want prata sia" becomes "want pizza", "biryani near
  tekka" becomes "Are you near the train?", "nasi lemak" becomes "cheap food".
- The rest of the bot believes the mistranslation. With the normalizer, the tracker gets a Singlish
  request fully right about 4 times in 10. Reading what the user typed, about 6 times in 10.
  Whenever the two disagree, the version without the normalizer is the one that's right.
- What the user sees: with the normalizer, about 1 in 3 places recommended to a Singlish user is a
  bad pick. Without it, about 1 in 6. English users get a bad pick about 1 in 25 times.
- The bot copes without it because Singlish already contains the words the search needs ("laksa",
  "katong", "tekka"), and "lah", "can" and "got" don't throw it off.
- A perfect normalizer (our hand-written English rewrites) does only slightly better than none at
  all. Improving this one would take new in-domain training data for very little gain.
- English users aren't affected either way: the normalizer mostly leaves English alone.
- **Recommendation: take the normalizer out** and give the tracker, the ranker and landmark
  detection the user's own text.
- Caveat: the 52 test dialogues (27 Singlish) are a Claude-written draft. That's enough to show the
  direction, but a teammate should spot-check them before exact numbers are quoted.

### Translation quality

| set | n | BLEU | chrF | exact match | copy input: BLEU | chrF |
|---|---|---|---|---|---|---|
| SEAT test (held out in training) | 161 | **55.3** | **69.6** | 28.0% | 30.4 | 55.2 |
| gold Singlish turns (restaurant requests) | 35 | 13.2 | 41.3 | 0% | 10.7 | 43.6 |

- SEAT ships as one CSV, so `normalizer_training.ipynb` splits it 80 / 10 / 10 (sklearn
  `train_test_split`, `random_state=42`) and trained the served model (`flan_t5_cleaned`) on the
  80%. `eval_normalizer.py` rebuilds that split, so these 161 pairs were never seen in training
  (they are the rows of `model_comparison.csv`, in the same order).
- An earlier version of this eval used its own seeded 10% split: 132 of its 160 pairs were
  training data, and it scored 59.8 / 74.3.
- The fine-tuned column of `model_comparison.csv` scores lower (BLEU 33.0, chrF 51.9) because it
  comes from an earlier checkpoint, made before the training data was cleaned and the model
  retrained.
- On our restaurant requests the normalizer is no better than copying the input (chrF 41.3 vs
  43.6). Part of that is reference style (the gold rewrites are full sentences, "Jurong West is
  fine."), but the real problem is entities: **14 of 35 requests lose or change a dish or place
  name**: "laksa near katong can?" → "Do you have a car near the station?", "biryani got near
  tekka?" → "Are you near the train?", "nasi padang near geylang serai got?" → "Do you have any
  pets near the beach?", "prata" → "pizza", "tampines" → "tomatoes", "bugis side got?" →
  "What's the bug's side?". It was trained on SMS-style chat, not food requests.
- One input loops: "Wahhh lobang Lai Liao" → "wow woooo…" for all 128 tokens (10 s on CPU). A
  `repetition_penalty` or `no_repeat_ngram_size` in `normalizer/hf.py` would stop it.

### Effect on the pipeline

The normalizer's text goes to three places: the tracker's `Normalized:` line, the ranker's content
score (`query_norm`) and landmark detection. Each was run three ways: with the normalizer's output (as
the backend runs today), with the user's raw text (no normalizer), and with the gold English rewrite
(a perfect normalizer).

**Ranker alone** (gold state, only the query text changes; `eval_recommender.py --query-from`):

| Singlish dialogues (27) | P@3 | strict P@3 | nDCG@3 |
|---|---|---|---|
| raw Singlish (no normalizer) | **0.877** | 0.617 | **0.823** |
| gold English rewrite | 0.864 | **0.630** | 0.814 |
| normalizer output | 0.852 | 0.481 | 0.731 |
| rating only | 0.765 | 0.432 | 0.626 |

The hard filters keep the normalizer's picks acceptable (P@3 barely moves), but the dish match is
lost: "great" picks fall from 62-63% to 48%. The raw text does as well as a perfect rewrite, because
the content score matches the dish and place words, which the raw text already has.

**Tracker** (`eval_tracker.py`: each dialogue replayed through `MakanSession.step`, state after every
turn compared with `gold_state`):

| Singlish turns (35) | joint goal acc. | slot acc. | slot F1 | landmark |
|---|---|---|---|---|
| normalizer output (as served) | 0.400 | 0.890 | 0.795 | 0.829 |
| raw text (no normalizer) | 0.571 | 0.924 | 0.854 | **0.943** |
| gold English rewrite | **0.657** | **0.943** | **0.890** | 0.886 |

English turns (32) give the same result all three ways (joint 0.781, slot accuracy 0.953).

- 6 Singlish turns (from 4 dialogues) have every slot right only without the normalizer, and none
  the other way:
  - "prata" → "pizza" sets cuisine to pizza, wrong for all 3 turns of that dialogue.
  - "nasi lemak" → "cheap food" invents a ≤$10 budget.
  - "bak kut teh" → "bak already" makes the cuisine "bak".
  - "korean fried chicken jurong point got?" → "Do you like Korean fried chicken?".
- The normalizer also loses the spot the user named: "near tekka", "near geylang serai",
  "holland v".
- Why: in training the `Normalized:` line was always right (the synthetic data's clean rewrite, a
  copy of the text for MultiWOZ). So when it disagrees with the `User:` line, the tracker trusts it.
- The perfect rewrite gets 3 more turns right than the raw text:
  - "Holland V" read as the Queenstown area (2 turns; an alias entry would also fix this).
  - A "hawker" setting the tracker invented from "i dun eat beef leh".
  - Its landmark score is lower only because it writes "Holland Village" where the gold label says
    "holland v".
- Errors shared by all three come from value translation, not the normalizer:
  - "korean bbq", "dim sum", "pho" and "tom yum" aren't mapped to a cuisine.
  - "NEX", "Old Airport Road", "Tekka" and "Geylang Serai" aren't mapped to an area.

**Whole pipeline** (each dialogue's final tracked state ranked on the same text):

| | P@3 | strict P@3 | nDCG@3 |
|---|---|---|---|
| Singlish (27), with the normalizer (as served) | 0.642 | 0.432 | 0.583 |
| Singlish (27), raw text (no normalizer) | **0.827** | 0.593 | 0.761 |
| Singlish (27), gold English rewrite | 0.815 | **0.617** | **0.777** |
| English (25), any of the three | 0.960-0.973 | 0.653-0.667 | 0.860-0.869 |

- Without the normalizer: P@3 +0.185 (95% bootstrap CI +0.037 to +0.346), nDCG@3 +0.178 (CI +0.027
  to +0.342). That's better on 9 dialogues, worse on 4, tied on 14.
- A perfect rewrite over the raw text: nDCG@3 +0.016 (CI −0.025 to +0.065), no measurable gain.
- How it was run:
  - Replies between turns are the policy's template replies, not the Qwen generator's, and the style
    is the gold label. So only the normalized text differs between the three runs.
  - The tracked states put 32 new places in the top 3. These were graded blind with the same rubric
    (6 great, 4 acceptable, 22 not), so every score is fully judged.
  - On the laptop CPU, 86 of the 150 tracker prompts ran with int8 weights (`--int8`, for lack of
    RAM). On 32 prompts this gave exactly the same output as the bfloat16 weights the backend serves.
- If the team keeps a normalizer, these would address the errors above:
  - keep alias-table words (dishes, places) unchanged in its output;
  - fine-tune it on in-domain pairs (the synthetic `[TRACK]` data has Singlish lines with their
    English);
  - stop the loop with a repetition penalty.

## Style detector

`python eval/eval_style.py`

| set | system | macro F1 | accuracy | Singlish P / R / F1 | English F1 |
|---|---|---|---|---|---|
| dev (1,189) | SingBERT | 0.904 | 0.950 | 0.854 / 0.822 / 0.837 | 0.971 |
| dev (1,189) | keywords | **0.930** | 0.966 | 0.993 / 0.789 / 0.880 | 0.980 |
| gold (67) | SingBERT | **0.895** | 0.896 | 0.889 / 0.914 / 0.901 | 0.889 |
| gold (67) | keywords | 0.717 | 0.731 | 1.000 / 0.486 / 0.654 | 0.780 |

- The dev set is the one `training/style_singbert.py` builds: templated synthetic turns plus MultiWOZ.
  The templates lean on particles, so the keyword list is a strong baseline there and slightly
  beats the model (it is also right on 99.9% of MultiWOZ turns vs 98.5%).
- On the hand-written gold turns the keyword list finds only half the Singlish (recall 0.486):
  "jurong west can", "korean food at nex can?" have no particle. SingBERT finds 91%.
- SingBERT's errors: Singlish without particles ("bro malay got", "cheap one can, Orchard area can")
  read as English, and English naming local food ("nasi lemak", "laksa", "you tiao") or a short
  fragment ("Somewhere in Ang Mo Kio.") reads as Singlish.

## Generator

`python eval/eval_generator.py`

**266 of 276 dev replies (96.4%) pass every check**: Singlish 165/175 (94.3%), English 101/101.

| action | Singlish | English |
|---|---|---|
| ask | 82/82 | 33/33 |
| recommend | 29/39 | 40/40 |
| fallback (with or without places) | 54/54 | 28/28 |

- No reply names a place the action didn't pick, leaves one out, skips an ask option, invents a
  price / distance / rating / time, leaks prompt text or runs too long.
- All 10 failures are `style_mismatch` on Singlish recommendations, and they are light Singlish
  rather than English: "Orchard got! Lucky Plaza Thai Kitchen - around $14, 300 m away, 4.1 stars.
  Want to go?" A list of names and numbers outweighs one "got", and SingBERT misses
  particle-less Singlish (the same weakness as in the style results). Some of these would pass a
  human reader.
- Caveat from `data/reply_data/README.md`: the dev actions use the 21 made-up places and templated
  user turns of the training data, so this measures the format the model learned, not replies on
  the real database.
- Run on a laptop CPU (no GPU): base weights in bfloat16 as served, maths in float32, LoRA merged,
  greedy. On a sample this matched the bfloat16 serving output token for token. About 40 minutes.

## How to rerun

From the repo root with `backend/requirements-ml.txt` installed and `HF_TOKEN` in `.env`:

```bash
python eval/eval_recommender.py --k 3 --ablations --by-style
python eval/eval_normalizer.py
python eval/eval_style.py
python eval/eval_generator.py
python eval/eval_tracker.py            # add --int8 on a CPU with little free RAM; ~30 min on a laptop
```

Per-item outputs go to `outputs/eval/` (gitignored).
