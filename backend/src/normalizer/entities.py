"""Protect restaurant-domain entities from being changed by FLAN-T5."""

import re
from functools import lru_cache

import pandas as pd
from src import config


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


@lru_cache(maxsize=1)
def _compiled_pattern():
    terms = set(FOOD_TERMS) | set(DIETARY_TERMS)

    path = config.PLACES_CSV
    if path.is_file():
        df = pd.read_csv(path, low_memory=False)

        for column in ("name", "area", "region", "cuisine"):
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
