# Training

Notebooks/scripts that build training files from `data/master/` and fine-tune the models
(Colab or Kaggle T4 is enough). Name them `<model>_<owner>.ipynb`, e.g. `normalizer_varidhi.ipynb`.

| model | owner | base | push to (private HF repo) | backend variable (defaults to the team repo) |
|---|---|---|---|---|
| normalizer | Varidhi | `google/flan-t5-base` | `<team>/makan-normalizer` | `MAKAN_NORMALIZER_MODEL` |
| style | Varidhi | `zanelim/singbert` (full fine-tune) | `makan-where-cs425/makan-singbert-style-classifier` (public; the backend's default) | `MAKAN_STYLE_MODEL` |
| tracker | Nichole | `Qwen/Qwen2.5-1.5B-Instruct` (QLoRA) | `makan-where-cs425/makan-qwen-track` (adapter + tokenizer) | `MAKAN_TRACKER_ADAPTER` |
| generator | Denise | `Qwen/Qwen2.5-1.5B-Instruct` (QLoRA) | `makan-where-cs425/makan-qwen-reply` (adapter + tokenizer) | `MAKAN_GENERATOR_ADAPTER` |

Build the inputs with the backend's own prompt functions, so serving matches training:

```python
import sys; sys.path.append("../backend")
from src.tracker.hf import build_prompt as track_prompt, build_target
from src.generator.hf import build_prompt as reply_prompt
from src.normalizer.hf import PREFIX
```

The tracker and generator each have their own LoRA adapter on the same base model, so each can be
retrained without the other. A single adapter trained on both (`makan-qwen-shared`) is kept as an
ablation. Never commit weights or `HF_TOKEN`; push to the Hub instead (see the main README).

## Notebooks

| notebook | trains | notes |
|---|---|---|
| `qwen_track.ipynb` | runs `qwen_track.py` in Colab / Jupyter | Colab T4 (secrets `HF_TOKEN`, `GITHUB_TOKEN`), checkpoints on Drive. `QUICK_TRIAL = True` for a few-minute check first; separate cells for the test scores and the Hub push. |
| `qwen_track.py` | Qwen LoRA for the tracker (`[TRACK]`), as a script | `python training/qwen_track.py [--push]` from a checkout on a GPU; `--eval-only <adapter> --split test` for the final MultiWOZ numbers. Reports joint / slot / per-slot accuracy and slot F1 per source. |
| `qwen_denise.ipynb` | Qwen LoRA: `TASK = "reply"` (generator), `"track"` (tracker) or `"mix"` (shared-adapter ablation) | GPU cluster (run from `training/`, `HF_TOKEN` exported) or Colab (secrets `HF_TOKEN`, `GITHUB_TOKEN`). Evaluates replies with `generator.checks.check_reply` and tracking with joint / slot accuracy, then pushes adapter + tokenizer. |
| `style_singbert.ipynb` | runs `style_singbert.py` in Colab / Jupyter | Colab T4 (secrets `HF_TOKEN`, `GITHUB_TOKEN`), checkpoints on Drive; a full run takes minutes. `QUICK_TRIAL = True` for a check first; a cell to read the dev errors, and one for the Hub push. |
| `style_singbert.py` | SingBERT style classifier (`english` / `singlish`), as a script | `python training/style_singbert.py [--push]`. Builds the data from the user turns of the `[TRACK]` files (synthetic Singlish lines vs. their normalized English lines and MultiWOZ turns); `--extra` adds real Singlish (CoSEM, NUS SMS). Reports macro F1 on dev through `src.style.hf.detect_style`, with a keyword heuristic as a baseline. |
