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
    entities._entity_index.cache_clear()
    yield
    entities._entity_index.cache_clear()


def protected(text: str) -> list[str]:
    _, mapping = entities.protect_entities(text)
    return sorted(v.lower() for v in mapping.values())


def test_neighbourhoods_and_dishes_from_the_alias_table(places):
    assert protected("biryani got near tekka?") == ["biryani", "tekka"]
    assert protected("pho or tom yum near holland v for supper") == ["holland v", "pho", "tom yum"]
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


def test_time_words_are_left_to_the_model(places):
    assert protected("supper now can? or late night lunch") == []


def test_placeholders_are_typed_and_numbered(places):
    # The old "ZXQENTITY00000QXZ" codes split into ~11 tokens the model garbled, so the guard
    # threw away every normalized food question.
    masked, mapping = entities.protect_entities("in tampines got halal food or not?")
    assert masked == "in Place1 got Diet1 food or not?"
    assert mapping == {"Place1": "tampines", "Diet1": "halal"}

    masked, mapping = entities.protect_entities("laksa or korean near bugis, bugis better")
    assert masked == "Food1 or Food2 near Place1, Place1 better"
    assert mapping == {"Food1": "laksa", "Food2": "korean", "Place1": "bugis"}


def test_restore_after_the_model_reorders_and_recases(places):
    _, mapping = entities.protect_entities("in tampines got halal food or not?")
    output = "Do you have diet1 food in Place1?"
    assert entities.all_entities_preserved(output, mapping)
    assert entities.restore_entities(output, mapping) == "Do you have halal food in tampines?"


def test_a_dropped_placeholder_is_caught(places):
    _, mapping = entities.protect_entities("in tampines got halal food or not?")
    assert not entities.all_entities_preserved("Do you have food in Place1?", mapping)
    # "Place10" is not "Place1"
    assert not entities.all_entities_preserved("Place10 Diet1 food", mapping)


def test_text_holding_a_placeholder_shaped_word_is_left_alone(places, monkeypatch):
    assert entities.protect_entities("food1 near bedok") == (None, {})

    from src.normalizer import hf
    monkeypatch.setattr(hf, "_load", lambda: pytest.fail("the model should not run"))
    assert hf.normalize("food1 near bedok") == "food1 near bedok"


def test_training_pairs_mask_terms_found_on_both_sides(places):
    sg, en, mapping = entities.mask_pair("kopi at tampines got halal or not", "Is there halal coffee at Tampines?")
    # "kopi" became "coffee": a translation for the model to learn, so it stays
    assert sg == "kopi at Place1 got Diet1 or not"
    assert en == "Is there Diet1 coffee at Place1?"
    assert mapping == {"Place1": "tampines", "Diet1": "halal"}


def test_training_pairs_mask_like_the_pipeline(places):
    text = "want laksa near bedok"
    sg, _, _ = entities.mask_pair(text, "I want laksa near Bedok.")
    assert sg == entities.protect_entities(text)[0]
