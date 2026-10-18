# Training

Notebooks/scripts that build training files from `data/master/` and fine-tune the models
(Colab or Kaggle T4 is enough). Name them `<model>_<owner>.ipynb`, e.g. `normalizer_varidhi.ipynb`.

| model | owner | base | push to (private HF repo) | backend switch |
|---|---|---|---|---|
| normalizer | Varidhi | `google/flan-t5-base` | `<team>/makan-normalizer` | `MAKAN_NORMALIZER=hf` |
| style | Varidhi | SingBERT | `<team>/makan-style` | `MAKAN_STYLE=hf` |
| tracker + generator | Nichole, Denise | `Qwen/Qwen2.5-1.5B-Instruct` (QLoRA) | `<team>/makan-qwen-lora` (adapter + tokenizer) | `MAKAN_TRACKER=hf`, `MAKAN_GENERATOR=hf` |

Build the inputs with the backend's own prompt functions, so serving matches training:

```python
import sys; sys.path.append("../backend")
from src.tracker.hf import build_prompt as track_prompt, build_target
from src.generator.hf import build_prompt as reply_prompt
from src.normalizer.hf import PREFIX
```

The tracker and generator share one adapter: mix `[TRACK]` and `[REPLY]` examples in one
training file. Never commit weights or `HF_TOKEN`; push to the Hub instead (see the main README).
