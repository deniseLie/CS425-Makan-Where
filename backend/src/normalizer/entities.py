"""Protect restaurant-domain entities from being changed by FLAN-T5.

Protected: the dishes and dietary words below, the place list's areas, regions and cuisines, and
the cuisine, area, time and dietary aliases of data/master/aliases.csv: the neighbourhoods,
dishes and meal words people type, which the place list doesn't hold ("bugis", "tekka",
"holland v", "pho", "tom yum", "supper").

Restaurant names are not protected: across SEAT's 1,603 chat lines every name that matched was
an everyday word or phrase ("he", "eat", "bojio", "ya lor", "that place"), which froze it in 6%
of inputs, and none of our restaurant requests names a restaurant.
"""

import re
from functools import lru_cache

import pandas as pd
from src import config
from src.recommender.aliases import load_aliases

PLACE_COLUMNS = ("area", "region", "cuisine")
# budget and setting aliases ("tapau", "atas", "ex") are Singlish the normalizer should translate
ALIAS_SLOTS = ("cuisine", "area", "time", "dietary")
MIN_ALIAS_LEN = 3  # "je", "bb", "ne", "rn" are too ambiguous to match in free text


FOOD_TERMS = {
    "chicken rice", "nasi lemak", "nasi goreng", "mee goreng",
    "mee rebus", "mee siam", "mee pok", "mee kia", "bak chor mee",
    "fishball noodles", "wanton mee", "char kway teow", "hokkien mee",
    "fried carrot cake", "carrot cake", "roti prata", "prata",
    "murtabak", "thosai", "dosa", "idli", "vadai", "biryani",
    "laksa", "curry puff", "kaya toast", "soft boiled eggs",
    "satay", "sambal", "rendang", "rojak", "popiah",
    "bak kut teh", "chilli crab", "black pepper crab",
    "dim sum", "congee", "porridge", "ban mian", "yong tau foo",
    "ice kacang", "chendol", "tau huay", "kopi", "teh tarik",
}

DIETARY_TERMS = {
    "vegetarian", "vegan", "halal", "no pork", "no lard",
    "gluten free", "dairy free", "nut free",
}


def alias_terms() -> set[str]:
    """The cuisine, area, time and dietary aliases of aliases.csv, at least MIN_ALIAS_LEN long."""
    if not config.ALIASES_CSV.is_file():
        return set()
    df = load_aliases(config.ALIASES_CSV)
    return {a for a in df.loc[df["slot"].isin(ALIAS_SLOTS), "alias"].dropna() if len(a) >= MIN_ALIAS_LEN}


@lru_cache(maxsize=1)
def _compiled_pattern():
    terms = set(FOOD_TERMS) | set(DIETARY_TERMS) | alias_terms()

    path = config.PLACES_CSV
    if path.is_file():
        df = pd.read_csv(path, usecols=lambda c: c in PLACE_COLUMNS, low_memory=False)

        for column in PLACE_COLUMNS:
            if column not in df.columns:
                continue

            for value in df[column].dropna().astype(str).str.strip():
                value = re.sub(r"\s+", " ", value)
                if value and len(value) <= 100:
                    terms.add(value)

    # Prefer longer phrases when multiple terms overlap.
    alternatives = sorted(terms, key=len, reverse=True)
    if not alternatives:
        return None

    pattern = "|".join(re.escape(term) for term in alternatives)
    return re.compile(rf"(?<!\w)(?:{pattern})(?!\w)", re.IGNORECASE)


def protect_entities(text: str):
    """Replace known entities with placeholders and return their originals."""
    mapping = {}
    pattern = _compiled_pattern()

    if pattern is None:
        return text, mapping

    counter = 0

    def replace(match):
        nonlocal counter
        placeholder = f"ZXQENTITY{counter:05d}QXZ"
        mapping[placeholder] = match.group(0)
        counter += 1
        return placeholder

    return pattern.sub(replace, text), mapping


def restore_entities(text: str, mapping: dict) -> str:
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text


def all_entities_preserved(text: str, mapping: dict) -> bool:
    return all(placeholder in text for placeholder in mapping)
