# Evaluation results (9 Nov 2026; normalizer and tracker rows 17 Nov)

The proposal's metrics for the recommender, normalizer, style detector and reply generator, run on
the models the backend serves today (`makan-flan-normalizer-masked`, `makan-singbert-style-classifier`,
`makan-qwen-reply` on Qwen2.5-1.5B-Instruct) and the current `data/master/places.csv`. Since 17 Nov
the backend normalizes only the turns the style detector calls Singlish (English-skip). The
tracker's MultiWOZ and synthetic numbers are in `docs/presentation_draft.md` (from
`training/qwen_track.py`); the tracker on the gold dialogues, with and without the normalizer, is
under Normalizer below.

| Component | Test set | Metric | Ours | Baseline |
|---|---|---|---|---|
| Recommender | 52 gold dialogues | P@3 / nDCG@3 | **0.910 / 0.836** | rating only: 0.833 / 0.710 |
| Recommender | 52 gold dialogues | hit@3 / hard-constraint violations | 1.000 / 0% | 0.923 / 0% |
| Normalizer | SEAT test (held out in training, 161 pairs) | BLEU / chrF | **53.8 / 68.1** | copy input: 30.4 / 55.2 |
| Normalizer | gold Singlish turns (35) | BLEU / chrF | **22.0 / 54.9** | copy input: 10.7 / 43.6 |
| Tracker | gold Singlish turns (35) | joint goal accuracy | with the normalizer: **0.600** | without it: 0.571 |
| Whole pipeline | 27 gold Singlish dialogues | P@3 / nDCG@3 of the final picks | with the normalizer: 0.827 / **0.783** | without it: 0.827 / 0.761 |
| Whole pipeline | all 52 gold dialogues | P@3 / nDCG@3 of the final picks | English-skip: **0.891 / 0.820** | normalize every turn: 0.885 / 0.815; none: 0.891 / 0.809 |
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
Relevance was judged by pooling (union of every ranking's top places, 509 places in all with the
ones added for the normalizer and tracker runs below) and blind grading by Claude subagents that saw
only the request and place details, never which system ranked a place (rubric in
`eval/gold/README.md`). Judging more places can only lower nDCG@3, since its best possible score
counts every good place known; it moved by at most 0.01. The proposal asks for team-judged places, so a
teammate should spot-check the grades before the numbers are final.

## Normalizer

`python eval/eval_normalizer.py` (translation quality), `python eval/eval_tracker.py` (its effect on
the pipeline)

**In short (17 Nov): the masked normalizer (`makan-flan-normalizer-masked`, trained with typed
placeholders and 1,000 synthetic restaurant requests) is the first one that helps the pipeline:
on Singlish it gets one more turn fully right than the raw text and ranks the final picks better.
On English it only rewords the request and costs a pick, so the backend now skips it for turns
the style detector calls English. That version is best or tied-best on every measure.**

The sections after this one are the 9 Nov analysis of the earlier model (`makan-flan-normalizer`),
kept because they explain why the entity guards exist.

### 17 Nov: the masked model and English-skip

`python eval/eval_tracker.py --variants normalizer raw gated`, run on the Origami GPU cluster.
`gated` is what the backend runs now (`normalize_for` in `backend/src/pipeline.py`): the
normalizer's output when SingBERT says "singlish", the raw text when it says "english". The skip
uses SingBERT's prediction, not the gold label, so its mistakes count.

| | joint goal acc. | slot acc. | P@3 | strict P@3 | nDCG@3 | unjudged in top 3 |
|---|---|---|---|---|---|---|
| **Singlish** (35 turns / 27 dialogues): normalize every turn | **0.600** | **0.929** | 0.827 | **0.617** | **0.783** | 0 |
| Singlish: English-skip (as served) | **0.600** | **0.929** | 0.827 | **0.617** | **0.783** | 0 |
| Singlish: raw text (no normalizer) | 0.571 | 0.924 | 0.827 | 0.593 | 0.761 | 0 |
| **English** (32 turns / 25 dialogues): normalize every turn | 0.781 | 0.953 | 0.947 | 0.640 | 0.849 | 1 |
| English: English-skip (as served) | 0.781 | 0.953 | **0.960** | **0.653** | **0.860** | 0 |
| English: raw text (no normalizer) | 0.781 | 0.953 | **0.960** | **0.653** | **0.860** | 0 |
| **All** (67 turns / 52 dialogues): normalize every turn | **0.687** | **0.940** | 0.885 | 0.628 | 0.815 | 1 |
| All: English-skip (as served) | **0.687** | **0.940** | **0.891** | **0.635** | **0.820** | 0 |
| All: raw text (no normalizer) | 0.672 | 0.938 | **0.891** | 0.622 | 0.809 | 0 |

- Translation quality (`eval_normalizer.py`, from `NORMALIZER_README.md`): on the gold Singlish
  turns BLEU / chrF rose from 12.5 / 44.7 (the earlier model, same typed placeholders) to
  22.0 / 54.9, and the model now rewrites every request instead of returning 46% unchanged. On
  SEAT it holds (54.5 / 68.9 → 53.8 / 68.1).
- Singlish: one turn has all six slots right only with the normalizer, none only without it. In
  the 9 Nov run the earlier model tied the raw text on every Singlish measure.
- English: the tracker reads both texts the same way (joint 0.781 either way), but the ranker's
  content score reads the rewording, which loses a pick and puts one ungraded place in a top 3
  (counted as not relevant). Skipping the normalizer for English gives exactly the raw text's
  scores, and saves about 1 s per English message on a laptop CPU.
- SingBERT got 32 of 35 Singlish and 28 of 32 English gold turns right. None of the 7 misses
  changed a score: a Singlish turn read as English gets the raw text, which tracks as well here,
  and an English turn read as Singlish gets the normalizer, which the tracker reads the same as
  the raw text on English.
- The differences are one turn and a few picks on a small, Claude-drafted test set: "best or
  tied-best everywhere", not a large gain.
- How it was run: the same replay as below (template replies, gold style for the replies). The
  tracker ran on a GPU in float16, with every prompt generated fresh; the raw-text Singlish scores
  match the 9 Nov CPU run exactly, so the two runs compare.

### 9 Nov: the earlier model

- The normalizer is meant to turn Singlish into proper English before the rest of the bot reads it.
  The bare model handles the chatty parts fine ("lah", "leh", "got anot"), but it often
  mistranslates the words that matter most, the dish and the place: "want prata sia" became "want
  pizza", "biryani near tekka" "Are you near the train?", "nasi lemak" "cheap food".
- Since b988c17, `normalize()` protects known dishes, cuisines and dietary words, and returns the
  user's text unchanged when the model drops one, loops or runs long. Its list now also has the
  cuisine, area, time and dietary aliases of `data/master/aliases.csv` (the neighbourhoods people
  type, "bugis", "tekka", "holland v", and "pho", "tom yum", "supper"), and no longer the place
  file's restaurant names. Dish and place words now survive in all 31 requests that have one (15
  with the bare model, 26 with b988c17's list); the fallback fires on 77% of requests (57%).
- The rest of the bot believes whatever the normalizer writes. With the bare model, the tracker got
  a Singlish request fully right about 4 times in 10, 6 in 10 without a normalizer. With the guards
  it gets 6 in 10 either way.
- What the user sees: about 1 in 6 places recommended to a Singlish user is a bad pick, with or
  without the normalizer (1 in 4 with b988c17's list, 1 in 3 with the bare model). English users
  get a bad pick about 1 in 25 times either way.
- The last gap was words b988c17's list didn't have: "pho", "tom yum", "supper" and "holland v"
  were still changed, and the raw text did better on 4 Singlish dialogues. Protecting the alias
  words closed it.
- The bot copes without a normalizer because Singlish already contains the words the search needs
  ("laksa", "katong", "tekka"), and "lah", "can" and "got" don't throw it off. A perfect normalizer
  (our hand-written English rewrites) does only slightly better than none at all.
- Recommendation on 9 Nov: either choice is safe, since the normalizer ties the raw text on every
  measure; keep it only if it gets fine-tuned on restaurant requests. (Done on 17 Nov: the masked
  model above, which now helps on Singlish.) Rerun `eval_tracker.py` after any change to it.
- Caveat: the 52 test dialogues (27 Singlish) are a Claude-written draft. That's enough to show the
  direction, but a teammate should spot-check them before exact numbers are quoted.

### Translation quality (9 Nov, earlier model)

| set | normalizer | n | BLEU | chrF | exact match | returned unchanged | copy input: BLEU | chrF |
|---|---|---|---|---|---|---|---|---|
| SEAT test (held out in training) | as served (guards, alias words protected) | 161 | 53.3 | 68.4 | 27.3% | 1.2% | 30.4 | 55.2 |
| SEAT test (held out in training) | guards with b988c17's list | 161 | 53.2 | 68.1 | 26.7% | 1.2% | 30.4 | 55.2 |
| SEAT test (held out in training) | bare model (`--bare`) | 161 | **55.3** | **69.6** | 28.0% | 0.6% | 30.4 | 55.2 |
| gold Singlish turns (restaurant requests) | as served (guards, alias words protected) | 35 | 11.6 | **44.4** | 0% | 77.1% | 10.7 | 43.6 |
| gold Singlish turns (restaurant requests) | guards with b988c17's list | 35 | 10.0 | 43.1 | 0% | 57.1% | 10.7 | 43.6 |
| gold Singlish turns (restaurant requests) | bare model (`--bare`) | 35 | **13.2** | 41.3 | 0% | 11.4% | 10.7 | 43.6 |

- SEAT ships as one CSV, so `normalizer_training.ipynb` splits it 80 / 10 / 10 (sklearn
  `train_test_split`, `random_state=42`) and trained the served model (`flan_t5_cleaned`) on the
  80%. `eval_normalizer.py` rebuilds that split, so these 161 pairs were never seen in training
  (they are the rows of `model_comparison.csv`, in the same order).
- An earlier version of this eval used its own seeded 10% split: 132 of its 160 pairs were
  training data, and it scored 59.8 / 74.3.
- The fine-tuned column of `model_comparison.csv` scores lower (BLEU 33.0, chrF 51.9) because it
  comes from an earlier checkpoint, made before the training data was cleaned and the model
  retrained.
- On our restaurant requests the bare model is no better than copying the input (chrF 41.3 vs
  43.6). Part of that is reference style (the gold rewrites are full sentences, "Jurong West is
  fine."), but the real problem is entities. It was trained on SMS-style chat, not food requests:
  "laksa near katong can?" → "Do you have a car near the station?", "biryani got near tekka?" →
  "Are you near the train?", "nasi padang near geylang serai got?" → "Do you have any pets near the
  beach?", "prata" → "pizza", "tampines" → "tomatoes", "bugis side got?" → "What's the bug's side?".
- Dish, cuisine, place and dietary words, listed by hand for the 31 requests that have one: the
  bare model keeps all of them in 15 requests (41 of 60 words), b988c17's list in 26 (55 of 60), the
  served normalizer in all 31. Most of the gain is the fallback to the user's text.
- The 5 words b988c17's list lost were missing from it: "bugis" ("What's wrong with the side?"),
  "pho", "tom yum" ("tom yeah"), "tekka" ("the train") and "holland v" ("Holland"). It took the
  place file's names, areas and cuisines plus about 50 dishes and dietary words, and the place file
  stores planning areas ("downtown core", "outram"), not the neighbourhoods people type ("bugis",
  "chinatown", "katong", "tekka", "nex", "holland v"); "geylang serai" matched as "geylang".
- The served list adds the cuisine, area, time and dietary aliases of `data/master/aliases.csv` (3+
  characters: "je", "bb" are too ambiguous). It leaves out the budget and setting aliases ("tapau",
  "atas", "ex"), Singlish the normalizer should translate. It builds in 0.6 s (was 4 s).
- It no longer has restaurant names. Across SEAT's 1,603 chat lines every name that matched was an
  everyday word or phrase ("he" 40 times, "eat" 23, "bojio", "most", "ya lor", "that place"), which
  froze it in 102 lines (6%); "eat" also froze 5 of our 35 requests, and none of them names a
  restaurant. The common words it protects now are meal and time words ("now", "late", "dinner",
  in 163 of the 1,603 lines), which the tracker needs kept.
- On SEAT the guards cost 2 BLEU (55.3 → 53.3): `no_repeat_ngram_size=3` changes some outputs, and 2
  inputs come back unchanged (1 for the bare model).
- The loop is fixed: "Wahhh lobang Lai Liao" used to run all 128 tokens ("wow woooo…", 9 s on CPU).
  It now comes back unchanged in 0.6 s. The guards add about 0.3 s per input (median 1.1 s vs 0.8 s
  on a laptop CPU).

### Effect on the pipeline (9 Nov, earlier model)

The normalizer's text goes to three places: the tracker's `Normalized:` line, the ranker's content
score (`query_norm`) and landmark detection. Each was run with the normalizer's output (as the
backend runs today: b988c17's guards with the alias words protected), with b988c17's list, with
the bare model's output (as served before b988c17), with the user's raw text (no normalizer), and
with the gold English rewrite (a perfect normalizer).

**Ranker alone** (gold state, only the query text changes; `eval_recommender.py --query-from`):

| Singlish dialogues (27) | P@3 | strict P@3 | nDCG@3 |
|---|---|---|---|
| raw Singlish (no normalizer) | **0.877** | 0.617 | **0.823** |
| gold English rewrite | 0.864 | **0.630** | 0.814 |
| normalizer output, as served (alias words protected) | **0.877** | 0.617 | **0.823** |
| normalizer output, b988c17's list | **0.877** | 0.617 | 0.810 |
| bare model output (before b988c17) | 0.852 | 0.481 | 0.731 |
| rating only | 0.765 | 0.432 | 0.626 |

The hard filters kept the bare model's picks acceptable (P@3 barely moves), but the dish match was
lost: "great" picks fell from 62-63% to 48%. With the guards the ranker is back to the raw text's
level, and with the alias words protected it matches it exactly. The raw text does as well as a
perfect rewrite, because the content score matches the dish and place words, which the raw text
already has.

**Tracker** (`eval_tracker.py`: each dialogue replayed through `MakanSession.step`, state after every
turn compared with `gold_state`):

| Singlish turns (35) | joint goal acc. | slot acc. | slot F1 | landmark |
|---|---|---|---|---|
| normalizer output, as served (alias words protected) | 0.571 | 0.924 | 0.854 | **0.943** |
| normalizer output, b988c17's list | 0.571 | 0.924 | 0.859 | 0.886 |
| bare model output (before b988c17) | 0.400 | 0.890 | 0.795 | 0.829 |
| raw text (no normalizer) | 0.571 | 0.924 | 0.854 | **0.943** |
| gold English rewrite | **0.657** | **0.943** | **0.890** | 0.886 |

English turns (32) give the same result every way (joint 0.781, slot accuracy 0.953).

- With the bare model, 6 Singlish turns (from 4 dialogues) had every slot right only without the
  normalizer, and none the other way:
  - "prata" → "pizza" sets cuisine to pizza, wrong for all 3 turns of that dialogue.
  - "nasi lemak" → "cheap food" invents a ≤$10 budget.
  - "bak kut teh" → "bak already" makes the cuisine "bak".
  - "korean fried chicken jurong point got?" → "Do you like Korean fried chicken?".
- The bare model also lost the spot the user named: "near tekka", "near geylang serai",
  "holland v".
- With the guards, all of these now keep the user's words, and the tracker scores the same as on
  the raw text: no turn is fully right with one and not the other. The one landmark b988c17's list
  still lost, "holland v" ("thai food Holland have you got?", 2 turns), is kept with the alias
  words protected: every turn's state is now the same as on the raw text.
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
| Singlish (27), with the normalizer, as served (alias words protected) | **0.827** | 0.593 | 0.761 |
| Singlish (27), with the normalizer, b988c17's list | 0.765 | 0.568 | 0.705 |
| Singlish (27), with the bare model (before b988c17) | 0.642 | 0.432 | 0.583 |
| Singlish (27), raw text (no normalizer) | **0.827** | 0.593 | 0.761 |
| Singlish (27), gold English rewrite | 0.815 | **0.617** | **0.777** |
| English (25), any of these | 0.960-0.973 | 0.653-0.667 | 0.860-0.869 |

- With the alias words protected, the normalizer and the raw text give the same top 3 in all 27
  Singlish dialogues.
- Against b988c17's list the raw text was ahead: P@3 +0.062 (95% bootstrap CI +0.012 to +0.123),
  nDCG@3 +0.056 (CI +0.000 to +0.134), better on 4 dialogues, worse on 1, tied on 22. The 4:
  "pho" → "Do you have a question?" (the cuisine is lost), "tom yum" → "tom yeah", "holland v" →
  "Holland" (the landmark is lost), and "supper" dropped. All four words are protected now.
- The guards over the bare model: P@3 +0.123 (CI −0.012 to +0.284), nDCG@3 +0.122 (CI −0.022 to
  +0.281); better on 7 dialogues, worse on 5. In 3 of those 5 the guarded normalizer gives the
  user's own text, and the bare model's mistranslation happened to rank better ("char kway teow" →
  "coffee"). The other two are "tom yum" → "tom yeah" (the bare model kept "tom yum") and "I dun
  eat beef", where the tracker invents a hawker setting.
- Before the guards, raw text beat the normalizer by P@3 +0.185 (CI +0.037 to +0.346).
- A perfect rewrite over the raw text: nDCG@3 +0.016 (CI −0.025 to +0.065), no measurable gain.
- How it was run:
  - Replies between turns are the policy's template replies, not the Qwen generator's, and the style
    is the gold label. So only the normalized text differs between the three runs.
  - The tracked states put 32 new places in the top 3. These were graded blind with the same rubric
    (6 great, 4 acceptable, 22 not), so every score is fully judged. The guarded run added 2 more
    (1 great, 1 not), graded blind the same way. Shown 8 already-graded places as calibration, that
    judge matched 6 grades and was one grade higher on the other 2.
  - On the laptop CPU, 86 of the 150 tracker prompts ran with int8 weights (`--int8`, for lack of
    RAM). On 32 prompts this gave exactly the same output as the bfloat16 weights the backend serves.
    The 19 prompts new in the guarded run ran without `--int8`; so did the 6 new in the run with the
    alias words protected, which put no new place in a top 3.
- Done: the alias words are protected and restaurant names are not (`normalizer/entities.py`),
  with unit tests in `backend/tests/test_normalizer_protect.py`.
- Done on 17 Nov: fine-tuned on in-domain pairs (1,000 synthetic restaurant requests with typed
  placeholders, `training/build_normalizer_data.py`); results in the 17 Nov section above.
- Still open:
  - unit tests for the guards in `normalizer/hf.py` (dropped placeholder, repetition, length): the
    backend tests replace the normalizer with a fake, so that code doesn't run in `pytest`.

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
python eval/eval_normalizer.py         # add --bare for the model without entity protection
python eval/eval_style.py
python eval/eval_generator.py
python eval/eval_tracker.py            # add --int8 on a CPU with little free RAM; ~30 min on a laptop
```

Per-item outputs go to `outputs/eval/` (gitignored).
