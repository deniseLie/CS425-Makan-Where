"""Protect restaurant-domain entities from being changed by FLAN-T5.

Each protected word is swapped for a typed, numbered placeholder ("Place1", "Food1", "Diet1")
before the model runs and swapped back after it. The same functions mask the normalizer's
training data (training/build_normalizer_data.py), so the model learns to copy exactly these
placeholders.

Protected, by type:
  Place  the place list's areas and regions and the area aliases of data/master/aliases.csv
         ("tampines", "east", "bugis", "holland v")
  Food   the dishes below, the place list's cuisines and the cuisine aliases ("laksa", "korean",
         "pho", "tom yum")
  Diet   the dietary words below and the dietary aliases ("halal", "no pork", "vegetarian")

Time words ("now", "late", "dinner", "supper") are not protected: they are everyday English the
model already keeps, and in SEAT "now" and "late" are mostly not about meal times.

Restaurant names are not protected: across SEAT's 1,603 chat lines every name that matched was
an everyday word or phrase ("he", "eat", "bojio", "ya lor", "that place"), which froze it in 6%
of inputs, and none of our restaurant requests names a restaurant.
"""

import re
from functools import lru_cache

import pandas as pd
from src import config
from src.recommender.aliases import load_aliases

# budget, setting and time aliases ("tapau", "atas", "ex", "supper") are left to the model
ALIAS_TYPES = {"dietary": "Diet", "cuisine": "Food", "area": "Place"}
PLACE_COLUMN_TYPES = {"cuisine": "Food", "area": "Place", "region": "Place"}
MIN_ALIAS_LEN = 3  # "je", "bb", "ne", "rn" are too ambiguous to match in free text
TYPES = ("Place", "Food", "Diet")


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


def _alias_types() -> dict[str, str]:
    """{alias: type} for the dietary, cuisine and area aliases of aliases.csv."""
    if not config.ALIASES_CSV.is_file():
        return {}
    df = load_aliases(config.ALIASES_CSV)
    df = df[df["slot"].isin(ALIAS_TYPES) & df["alias"].notna()]
    return {a: ALIAS_TYPES[slot] for slot, a in zip(df["slot"], df["alias"]) if len(a) >= MIN_ALIAS_LEN}


def _place_list_types() -> dict[str, str]:
    """{value: type} for the place list's cuisines, areas and regions."""
    path = config.PLACES_CSV
    if not path.is_file():
        return {}
    df = pd.read_csv(path, usecols=lambda c: c in PLACE_COLUMN_TYPES, low_memory=False)
    out = {}
    for column, kind in PLACE_COLUMN_TYPES.items():
        if column not in df.columns:
            continue
        for value in df[column].dropna().astype(str).str.strip():
            value = re.sub(r"\s+", " ", value)
            if value and len(value) <= 100:
                out.setdefault(value, kind)
    return out


@lru_cache(maxsize=1)
def _entity_index():
    """(pattern matching any protected term, {lowercased term: type}).

    A term listed under two types keeps the first of: Diet, Food, Place (the word lists
    below, then aliases.csv, then the place list), so "halal" is always Diet.
    """
    types: dict[str, str] = {}
    sources = [{t: "Diet" for t in DIETARY_TERMS}, {t: "Food" for t in FOOD_TERMS}]
    aliases, places = _alias_types(), _place_list_types()
    for kind in TYPES[::-1]:  # Diet, Food, Place
        sources.append({t: k for t, k in aliases.items() if k == kind})
        sources.append({t: k for t, k in places.items() if k == kind})
    for source in sources:
        for term, kind in source.items():
            types.setdefault(term.lower(), kind)

    if not types:
        return None, {}
    # Prefer longer phrases when multiple terms overlap.
    alternatives = sorted(types, key=len, reverse=True)
    pattern = "|".join(re.escape(term) for term in alternatives)
    return re.compile(rf"(?<!\w)(?:{pattern})(?!\w)", re.IGNORECASE), types


# Placeholders are a type word and a number: "Place1", "Food2", "Diet1". Flan-T5's tokenizer
# splits each into 2-3 pieces of ordinary vocabulary ("Place" "1"), which the model can copy,
# where "ZXQENTITY00000QXZ" was ~11 pieces it garbled or dropped. The type tells the model what
# the word was ("in Place1", "Diet1 food"); the number makes swapping back exact.
PLACEHOLDER = re.compile(rf"(?<!\w)(?:{'|'.join(TYPES)})\d+(?!\w)", re.IGNORECASE)


def _word(term: str) -> re.Pattern:
    return re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)


def _mask(text: str, keep=lambda term: True):
    """Mask the protected terms of text for which keep(term) is true.

    The same term (ignoring case) gets the same placeholder each time it appears.
    Returns (masked text, {placeholder: original}, {lowercased term: placeholder}).
    """
    pattern, types = _entity_index()
    mapping: dict[str, str] = {}
    by_term: dict[str, str] = {}
    if pattern is None:
        return text, mapping, by_term
    counts = dict.fromkeys(TYPES, 0)

    def replace(match):
        term = match.group(0)
        key = term.lower()
        if not keep(key):
            return term
        if key not in by_term:
            kind = types[key]
            counts[kind] += 1
            by_term[key] = f"{kind}{counts[kind]}"
            mapping[by_term[key]] = term
        return by_term[key]

    return pattern.sub(replace, text), mapping, by_term


def protect_entities(text: str):
    """Replace known entities with placeholders and return (masked text, {placeholder: original}).

    The masked text is None when the text already holds something shaped like a placeholder
    ("Place1"), which couldn't be told apart from one, so the caller keeps the text as it is.
    """
    if PLACEHOLDER.search(text):
        return None, {}
    masked, mapping, _ = _mask(text)
    return masked, mapping


def mask_pair(singlish: str, english: str):
    """Mask a training pair: (masked singlish, masked english, {placeholder: original}).

    A protected term is masked only when it appears in both sides (ignoring case), with the same
    placeholder in each, so the target teaches the model to copy the placeholder through. A term
    only the Singlish has ("kopi" -> "coffee") stays as it is: the model is learning to translate it.
    Pairs that already hold a placeholder-shaped word come back unmasked.
    """
    if PLACEHOLDER.search(singlish) or PLACEHOLDER.search(english):
        return singlish, english, {}
    masked, mapping, by_term = _mask(singlish, keep=lambda term: bool(_word(term).search(english)))
    for term, placeholder in by_term.items():
        english = _word(term).sub(placeholder, english)
    return masked, english, mapping


def restore_entities(text: str, mapping: dict) -> str:
    for placeholder, original in mapping.items():
        text = _word(placeholder).sub(lambda _: original, text)
    return text


def all_entities_preserved(text: str, mapping: dict) -> bool:
    return all(_word(placeholder).search(text) for placeholder in mapping)
