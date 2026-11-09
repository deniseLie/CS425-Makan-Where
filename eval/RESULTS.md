# Evaluation results (9 Nov 2026)

The proposal's metrics for the recommender, normalizer, style detector and reply generator, run on
the models the backend serves today (`makan-flan-normalizer`, `makan-singbert-style-classifier`,
`makan-qwen-reply` on Qwen2.5-1.5B-Instruct) and the current `data/master/places.csv`. Tracker
numbers are in `docs/presentation_draft.md` (from `training/qwen_track.py`).

| Component | Test set | Metric | Ours | Baseline |
|---|---|---|---|---|
| Recommender | 52 gold dialogues | P@3 / nDCG@3 | **0.910 / 0.841** | rating only: 0.833 / 0.716 |
| Recommender | 52 gold dialogues | hit@3 / hard-constraint violations | 1.000 / 0% | 0.923 / 0% |
| Normalizer | SEAT test (held out in training, 161 pairs) | BLEU / chrF | **55.3 / 69.6** | copy input: 30.4 / 55.2 |
| Normalizer | gold Singlish turns (35) | BLEU / chrF | 13.2 / 41.3 | copy input: 10.7 / 43.6 |
| Style detector | dev (1,189 turns, 185 Singlish) | macro F1 | 0.904 | keywords: **0.930** |
| Style detector | gold turns (67, 35 Singlish) | macro F1 | **0.895** | keywords: 0.717 |
| Generator | dev (276 replies) | replies passing every check | **96.4%** (266/276) | teacher replies: 100% |

## Recommender

`python eval/eval_recommender.py --k 3 --ablations --by-style`

| ranking | P@3 | strict P@3 | nDCG@3 | hit@3 | violations |
|---|---|---|---|---|---|
| **hybrid** | **0.910** | **0.641** | **0.841** | 1.000 | 0% |
| rating only | 0.833 | 0.506 | 0.716 | 0.923 | 0% |
| without content | 0.821 | 0.532 | 0.721 | 0.942 | 0% |
| without soft | 0.910 | 0.603 | 0.823 | 1.000 | 0% |
| without quality | 0.897 | 0.571 | 0.799 | 1.000 | 0% |
| without distance | 0.917 | 0.615 | 0.824 | 1.000 | 0% |

- P@3 and hit@3 count places judged relevant (grade ≥ 1), strict P@3 only "great" ones (grade 2);
  nDCG@3 uses the grades 0 / 1 / 2 as gains.
- Hybrid beats rating-only by +0.077 P@3 (95% bootstrap CI +0.006 to +0.154) and +0.125 nDCG@3
  (CI +0.036 to +0.221), paired over the 52 dialogues; better nDCG@3 on 22 dialogues, worse on 17, tied on 13.
- Content similarity matters most: without it nDCG@3 falls to 0.721, close to rating-only. Every
  other part helps nDCG@3 a little; without distance P@3 is slightly higher (0.917) but nDCG@3 lower.
- Singlish dialogues 0.864 / 0.823 (rating only 0.765 / 0.637), English 0.960 / 0.860
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
Relevance was judged by pooling (union of every ranking's top places: 475 places) and blind
grading by four Claude subagents that saw only the request and place details, never which system
ranked a place (rubric in `eval/gold/README.md`). The proposal asks for team-judged places, so a
teammate should spot-check the grades before the numbers are final.

## Normalizer

`python eval/eval_normalizer.py`

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
- This reaches the recommendations: the ranker's content score reads the normalized text.
  Ranking the Singlish dialogues on the normalizer's output instead of the gold rewrite
  (`eval_recommender.py --query-from outputs/eval/normalizer_gold.jsonl`):

  | Singlish dialogues (27) | P@3 | strict P@3 | nDCG@3 |
  |---|---|---|---|
  | hybrid, gold English rewrite | 0.864 | 0.630 | 0.823 |
  | hybrid, normalizer output | 0.852 | **0.481** | **0.740** |
  | rating only | 0.765 | 0.432 | 0.637 |

  The hard filters still keep the picks acceptable (P@3 barely moves), but the dish match is lost:
  "great" picks fall from 63% to 48%. All 8 new top-3 places this produced were judged too.
- One input loops: "Wahhh lobang Lai Liao" → "wow woooo…" for all 128 tokens (10 s on CPU). A
  `repetition_penalty` or `no_repeat_ngram_size` in `normalizer/hf.py` would stop it.
- Possible fixes: keep alias-table words (dishes, places) unchanged in the output, rank on the raw
  and normalized text together, and fine-tune on in-domain pairs (the synthetic `[TRACK]` data has
  Singlish lines with their normalized English).

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
```

Per-item outputs go to `outputs/eval/` (gitignored).
