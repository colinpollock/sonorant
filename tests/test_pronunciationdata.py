from pandas import DataFrame

from sonorous.pronunciationdata import (
    SYLLABIC_PHONEMES,
    augment_pronunciations_df,
    count_phonemes,
    count_primary_stressed_syllables,
    count_syllables,
    load_pronunciations,
    tokenize_pronunciation_string,
)


def test_load_pronunciations():
    """This is partially just an integregation test to make sure the CMU Dict data
    doesn't change from under my feet.
    """
    pronunciations = load_pronunciations()
    assert len(pronunciations) == 131964

    # `load_pronunciations` drops all pronunciations without one primary stressed syllable,
    # so each one here hsould have one primary stressed syllable.
    assert (
        pronunciations.pronunciation.apply(count_primary_stressed_syllables).unique()
        == 1
    )

    cat = pronunciations.loc["cat"]
    assert cat.pronunciation == ("ˈ", "k", "æ", "t")
    assert cat.as_string == "ˈkæt"
    assert cat.num_phonemes == 3
    assert cat.num_syllables == 1


def test_count_syllables():
    assert count_syllables(("ˈ", "d", "ɔ", "g", "i")) == 2
    assert count_syllables(("r", "ɪ", "ˈ", "p", "i", "ː", "t", "ɝ")) == 3
    assert count_syllables(("ə", "ˈ", "b", "aʊ", "t")) == 2
    assert count_syllables(("ɪ", "k", "s", "ˈ", "k", "j", "uː", "z")) == 2


def test_count_phonemes():
    assert count_phonemes(("ˈ", "f", "ɝ", "ː", "i")) == 3
    assert count_phonemes(("ə", "ˈ", "b", "aʊ", "t")) == 4


def test_count_primary_stressed_syllables():
    assert count_primary_stressed_syllables(("ˈ", "k", "æ", "t")) == 1
    assert count_primary_stressed_syllables(("ˈ", "e", "ɪ", "ˈ", "a", "ɪ")) == 2


def test_augment_pronunciations_df():
    pronunciations = DataFrame({"pronunciation": [("ˈ", "r", "ɛ", "dʒ", "i")]})
    augment_pronunciations_df(pronunciations)

    reggie = pronunciations.iloc[0]
    assert reggie["num_phonemes"] == 4
    assert reggie["num_syllables"] == 2


def test_tokenize_pronunciation_string():
    # No groupings
    assert tokenize_pronunciation_string("ˈkæt") == ("ˈ", "k", "æ", "t")
    assert tokenize_pronunciation_string("ˈbætˌmæn") == (
        "ˈ",
        "b",
        "æ",
        "t",
        "ˌ",
        "m",
        "æ",
        "n",
    )

    # Length symbol grouped
    assert tokenize_pronunciation_string("ˈfɝːi") == ("ˈ", "f", "ɝː", "i")
    assert tokenize_pronunciation_string("ˈfuːd") == ("ˈ", "f", "uː", "d")

    # # Affricates
    assert tokenize_pronunciation_string("ˈtʃæt") == ("ˈ", "tʃ", "æ", "t")
    assert tokenize_pronunciation_string("ˈrɛdʒi") == ("ˈ", "r", "ɛ", "dʒ", "i")

    # # Diphthongs
    assert tokenize_pronunciation_string("ˈʃaɪ") == ("ˈ", "ʃ", "aɪ")
    assert tokenize_pronunciation_string("ˈbaʊd") == ("ˈ", "b", "aʊ", "d")
    assert tokenize_pronunciation_string("ˈtreɪ") == ("ˈ", "t", "r", "eɪ")
    assert tokenize_pronunciation_string("ˈbɔɪ") == ("ˈ", "b", "ɔɪ")
    assert tokenize_pronunciation_string("ˈkɔrd") == ("ˈ", "k", "ɔ", "r", "d")
    assert tokenize_pronunciation_string("ˈloʊ") == ("ˈ", "l", "oʊ")
