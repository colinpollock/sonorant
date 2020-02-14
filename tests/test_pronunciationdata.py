from pandas import DataFrame

from sonorous.pronunciationdata import (
    SYLLABIC_PHONEMES,
    augment_pronunciations_df,
    count_primary_stressed_syllables,
    count_syllables,
    load_pronunciations,
)


def test_load_data():
    """This is partially just an integregation test to make sure the CMU Dict data
    doesn't change from under my feet.
    """
    df = load_pronunciations()
    assert len(df) == 125801
    cat = df.loc["cat"]
    assert cat.pronunciation == ("K", "AE1", "T")
    assert cat.num_phonemes == 3
    assert cat.num_syllables == 1
    assert cat.num_primary_stressed_syllables == 1


def test_get_all_syllabic_phonemes():
    """The tested function mostly depends on the cmudict, so rather than testing its contents
    thoroughly I'll do a few sanity checks.
    """
    syllabic_phonemes = ("AH0", "AH1", "AH2", "ER0", "ER1", "ER2")
    non_syllabic_phonemes = ("K", "SH", "M")

    for syllabic in syllabic_phonemes:
        assert syllabic in SYLLABIC_PHONEMES

    for non_syllabic in non_syllabic_phonemes:
        assert non_syllabic not in SYLLABIC_PHONEMES


def test_count_syllables():
    assert count_syllables(("K", "AH1", "L", "IH0", "N")) == 2
    assert count_syllables(("F", "ER1")) == 1


def test_count_primary_stressed_syllables():
    assert count_primary_stressed_syllables(("K", "AE1", "F", "ER0")) == 1
    assert count_primary_stressed_syllables(("EY1", "AY1")) == 2


def test_augment_pronunciations_df():
    pronunciations = DataFrame({"pronunciation": [("R", "EH1", "DJ", "IY0")]})
    augment_pronunciations_df(pronunciations)

    reggie = pronunciations.iloc[0]
    assert reggie["num_phonemes"] == 4
    assert reggie["num_syllables"] == 2

