from pandas import DataFrame

from sonorous.pronunciationdata import (
    SYLLABIC_PHONEMES,
    augment_pronunciations_df,
    count_phonemes,
    count_primary_stressed_syllables,
    count_syllables,
    load_pronunciations,
)


def test_load_data():
    """This is partially just an integregation test to make sure the CMU Dict data
    doesn't change from under my feet.
    """
    pronunciations = load_pronunciations()
    assert len(pronunciations) == 124567
    cat = pronunciations.loc["cat"]

    assert cat.pronunciation == "ˈkæt"
    assert cat.num_phonemes == 3
    assert cat.num_syllables == 1
    assert cat.num_primary_stressed_syllables == 1


def test_count_syllables():
    assert count_syllables("ˈdɔgi") == 2
    assert count_syllables("rɪˈpiːtɝ") == 3
    assert count_syllables("əˈbaʊt") == 2


def test_count_phonemes():
    assert count_phonemes("ˈfɝːi") == 3
    assert count_phonemes("əˈbaʊt") == 4


def test_count_primary_stressed_syllables():
    assert count_primary_stressed_syllables("ˈkæt") == 1
    assert count_primary_stressed_syllables("ˈeɪˈaɪ") == 2


def test_augment_pronunciations_df():
    pronunciations = DataFrame({"pronunciation": ["ˈrɛdʒi"]})
    augment_pronunciations_df(pronunciations)

    reggie = pronunciations.iloc[0]
    assert reggie["num_phonemes"] == 5
    assert reggie["num_syllables"] == 2

