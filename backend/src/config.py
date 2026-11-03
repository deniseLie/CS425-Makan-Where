"""Runtime settings, all from environment variables (nothing secret lives in the repo).

Each model component runs either the local stub or the trained model from the Hugging Face Hub:

    MAKAN_NORMALIZER=stub|hf   MAKAN_NORMALIZER_MODEL=<hf repo id>    Flan-T5 (Varidhi)
    MAKAN_STYLE=stub|hf        MAKAN_STYLE_MODEL=<hf repo id>         SingBERT (Varidhi)
    MAKAN_TRACKER=stub|hf      MAKAN_LORA_ADAPTER=<hf repo id>        Qwen + LoRA [TRACK] (Nichole)
    MAKAN_GENERATOR=stub|hf    MAKAN_LORA_ADAPTER=<hf repo id>        Qwen + LoRA [REPLY] (Denise)
                               MAKAN_BASE_LLM (default Qwen/Qwen2.5-1.5B-Instruct)

MAKAN_LORA_ADAPTER defaults to the team's adapter, makan-where-cs425/makan-qwen-lora. Private
repos also need HF_TOKEN (read by huggingface_hub itself). Any of these can go in the repo's
.env file (gitignored) instead of the environment; a real environment variable wins.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root


def _load_dotenv(path: Path) -> None:
    """MAKAN_* and HF_* lines (KEY=value) from .env as defaults for os.environ; other keys are ignored."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        key, sep, value = line.strip().removeprefix("export ").partition("=")
        key = key.strip()
        if sep and key.startswith(("MAKAN_", "HF_")):
            os.environ.setdefault(key, value.strip().strip("'\""))


_load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("MAKAN_DATA_DIR", ROOT / "data"))
PLACES_CSV = Path(os.getenv("MAKAN_PLACES_CSV", DATA_DIR / "master" / "places.csv"))
FRONTEND_DIST = ROOT / "frontend" / "dist"
CORS_ORIGINS = [o.strip() for o in os.getenv("MAKAN_CORS_ORIGINS", "").split(",") if o.strip()]

COMPONENTS = ("normalizer", "style", "tracker", "generator")
MODEL_ENV = {
    "normalizer": "MAKAN_NORMALIZER_MODEL",
    "style": "MAKAN_STYLE_MODEL",
    "tracker": "MAKAN_LORA_ADAPTER",
    "generator": "MAKAN_LORA_ADAPTER",
}
QWEN_LORA = "makan-where-cs425/makan-qwen-lora"  # private: HF_TOKEN must be a member's token
DEFAULT_MODELS = {"tracker": QWEN_LORA, "generator": QWEN_LORA}
BASE_LLM = os.getenv("MAKAN_BASE_LLM", "Qwen/Qwen2.5-1.5B-Instruct")


def impl(component: str) -> str:
    """'stub' or 'hf' for one component."""
    value = os.getenv(f"MAKAN_{component.upper()}", "stub").strip().lower()
    if value not in ("stub", "hf"):
        raise ValueError(f"MAKAN_{component.upper()} must be 'stub' or 'hf', got {value!r}")
    return value


def model_id(component: str) -> str:
    """HF repo id for a component; fails loudly if it is switched to 'hf' without one."""
    var = MODEL_ENV[component]
    repo = os.getenv(var, "").strip() or DEFAULT_MODELS.get(component, "")
    if not repo:
        raise RuntimeError(f"MAKAN_{component.upper()}=hf needs {var} set to a Hugging Face repo id")
    return repo


def summary() -> dict:
    """What each component is running, for /api/health and the UI."""
    out = {}
    for c in COMPONENTS:
        mode = impl(c)
        repo = os.getenv(MODEL_ENV[c], "").strip() or DEFAULT_MODELS.get(c, "")
        out[c] = {"impl": mode, "model": repo if mode == "hf" else None}
    if "hf" in (impl("tracker"), impl("generator")):
        out["base_llm"] = BASE_LLM
    return out
