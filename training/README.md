# Training

Notebooks/scripts that build training files from `data/master/` and fine-tune the models
(Colab or Kaggle T4 is enough). Name them `<model>_<owner>.ipynb`, e.g. `normalizer_varidhi.ipynb`.

| model | owner | base | push to (private HF repo) | backend switch |
|---|---|---|---|---|
| normalizer | Varidhi | `google/flan-t5-base` | `<team>/makan-normalizer` | `MAKAN_NORMALIZER=hf` |
| style | Varidhi | SingBERT | `<team>/makan-style` | `MAKAN_STYLE=hf` |
| tracker | Nichole | `Qwen/Qwen2.5-1.5B-Instruct` (QLoRA) | `makan-where-cs425/makan-qwen-track` (adapter + tokenizer) | `MAKAN_TRACKER=hf` |
| generator | Denise | `Qwen/Qwen2.5-1.5B-Instruct` (QLoRA) | `makan-where-cs425/makan-qwen-reply` (adapter + tokenizer) | `MAKAN_GENERATOR=hf` |

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
| `qwen_denise.ipynb` | Qwen LoRA: `TASK = "reply"` (generator), `"track"` (tracker) or `"mix"` (shared-adapter ablation) | GPU cluster (run from `training/`, `HF_TOKEN` exported) or Colab (secrets `HF_TOKEN`, `GITHUB_TOKEN`). Evaluates replies with `generator.checks.check_reply` and tracking with joint / slot accuracy, then pushes adapter + tokenizer. |
