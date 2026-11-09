"""Entity protection for the Flan-T5 normalizer (src/normalizer/entities.py), without the model."""

import pytest

from src import config
from src.normalizer import entities


@pytest.fixture
def places(tmp_path, monkeypatch):
    """A place list with restaurants named after everyday words."""
    path = tmp_path / "places.csv"
    path.write_text("name,area,region,cuisine\n"
                    "Eat,bedok,east,korean\n"
                    "He,outram,central,chinese\n"
                    "The Bay,downtown core,central,thai\n", encoding="utf-8")
    monkeypatch.setattr(config, "PLACES_CSV", path)
    entities._compiled_pattern.cache_clear()
    yield
    entities._compiled_pattern.cache_clear()


def protected(text: str) -> list[str]:
    _, mapping = entities.protect_entities(text)
    return sorted(v.lower() for v in mapping.values())


def test_neighbourhoods_dishes_and_meals_from_the_alias_table(places):
    assert protected("biryani got near tekka?") == ["biryani", "tekka"]
    assert protected("pho or tom yum near holland v for supper") == ["holland v", "pho", "supper", "tom yum"]
    assert protected("bugis side got halal?") == ["bugis", "halal"]
    assert protected("nasi padang near geylang serai got?") == ["geylang serai", "nasi padang"]


def test_place_list_areas_and_cuisines(places):
    assert protected("korean in bedok") == ["bedok", "korean"]


def test_restaurant_names_are_not_protected(places):
    assert protected("he say eat first then go the bay") == []


def test_short_and_translatable_aliases_are_left_out(places):
    # "je" (Jurong East) is too short to match safely; "tapau" and "atas" should be translated
    assert protected("je can, tapau one, not atas") == []


def test_round_trip(places):
    text = "Laksa near Katong can?"
    masked, mapping = entities.protect_entities(text)
    assert "Laksa" not in masked and "Katong" not in masked
    assert entities.restore_entities(masked, mapping) == text
