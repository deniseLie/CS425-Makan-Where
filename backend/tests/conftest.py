"""Tests run on tests/data/places_sample.csv: 21 made-up places in places.csv's columns.

The real data/master/places.csv has no cuisine, halal, price or rating values yet, so the
filter and policy rules can't be checked against it. Set before src.config is imported.
"""

import os
from pathlib import Path

os.environ["MAKAN_PLACES_CSV"] = str(Path(__file__).parent / "data" / "places_sample.csv")
