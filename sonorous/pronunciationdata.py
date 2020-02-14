"""Interface for loading pronunciation data."""


from collections import Counter
from string import ascii_lowercase
from typing import Set, Tuple

import cmudict
from pandas import DataFrame


ALLOWABLE_LETTERS = set(ascii_lowercase)
CMU = cmudict.dict()


def load_pronunciations() -> DataFrame:
    """Return a DataFrame with columns for `word` and `pronunciation`."""
    records = []
    for word, pronunciations in CMU.items():
        if not all(letter in ALLOWABLE_LETTERS for letter in word):
            continue

        for pronunciation in pronunciations:
            records.append({"word": word, "pronunciation": tuple(pronunciation)})

    pronunciations = DataFrame(records).set_index("word")
    augment_pronunciations_df(pronunciations)

    return pronunciations


def augment_pronunciations_df(pronunciations: DataFrame) -> None:
    """Adds new fields to the input DataFrame, in place.

    Columns that are added:
    - num_phonemes
    - num_syllables
    """
    pronunciations["num_phonemes"] = pronunciations.pronunciation.apply(len)
    pronunciations["num_syllables"] = pronunciations.pronunciation.apply(
        count_syllables
    )

    pronunciations[
        "num_primary_stressed_syllables"
    ] = pronunciations.pronunciation.apply(count_primary_stressed_syllables)


def get_syllabic_phonemes() -> Set[str]:
    """Return a set of all phonemes that are syllabic.

    This is just vowels and the syllabic /R/, /ER/. CMU Dict doesn't seem to have other syllabic consonants.
    """
    stresses = ("0", "1", "2")
    vowels = set()
    for phoneme, features in cmudict.phones():
        if "vowel" in features:
            for stress in stresses:
                vowels.add(phoneme + stress)

    return vowels


SYLLABIC_PHONEMES = get_syllabic_phonemes()


def count_syllables(pronunciation: Tuple[str, ...]) -> int:
    """Return the number of syllables in a single pronunciation.

    This is approximate as I'm assuming the number of syllabic phonemes is the
    same as the number of vowels. It should be right almost all the time though.
    """
    counts = Counter(pronunciation)
    return sum(counts[vowel] for vowel in SYLLABIC_PHONEMES)


def count_primary_stressed_syllables(pronunciation: Tuple[str, ...]) -> int:
    """Return the number of syllables with primary stress.

    Typically words only have one syllable with primary stress, but compound words or acronyms will
    break that rule. For example, "ai" is /EY1 AY1/ because it's said as two words.
    """
    return sum(1 for phoneme in pronunciation if phoneme.endswith("1"))
