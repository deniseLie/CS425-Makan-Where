"""Tests run on tests/data/places_sample.csv: 21 made-up places in places.csv's columns, and on
the offline stand-ins in fakes.py instead of the four trained models.

The real data/master/places.csv has no cuisine, halal, price or rating values yet, so the
filter and policy rules can't be checked against it. Set before src.config is imported.
"""

import os
from pathlib import Path

import pytest

os.environ["MAKAN_PLACES_CSV"] = str(Path(__file__).parent / "data" / "places_sample.csv")

import fakes  # noqa: E402


@pytest.fixture(autouse=True)
def offline_models(monkeypatch):
    """No test downloads or runs a model: the pipeline and the reply checks use fakes.py."""
    import src.generator.checks
    import src.pipeline

    for name in ("normalize", "detect_style", "track", "generate"):
        monkeypatch.setattr(src.pipeline, name, getattr(fakes, name))
    monkeypatch.setattr(src.generator.checks, "detect_style", fakes.detect_style)
