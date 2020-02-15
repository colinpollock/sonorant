"""Interface for loading pronunciation data."""


from collections import Counter
from string import ascii_uppercase
from typing import Dict, List, Set, Tuple

from pandas import DataFrame


CMUDICT_FILEPATH = "data/cmudict-0.7b-ipa.txt"

SYLLABIC_PHONEMES = {
    "i",
    "e",
    "ʊ",
    "o",
    "u",
    "ɑ",
    "ɔ",
    "ə",
    "ɛ",
    "ɪ",
    "a",
    "ɝ",
    "æ",
    "ʌ",
}

PRIMARY_STRESS = "ˈ"
SECONDARY_STRESS = "ˌ"
LONG_VOWEl = "ː"
NON_PHONEME_SYMBOLS = {PRIMARY_STRESS, SECONDARY_STRESS, LONG_VOWEl}

DIPHTHONGS = {
    "oʊ",
    "aʊ",
    "aɪ",
    "eɪ",
    "ɔɪ",
}


def load_pronunciations(num_rows: int = None) -> DataFrame:
    """Return a DataFrame with columns for `word` and `pronunciation`."""
    records: List[Dict[str, str]] = []

    with open("t", "r") as fh:
        for line in fh:
            word, pronunciations = line.strip().split("\t")

            if not all(letter in ascii_uppercase for letter in word):
                continue

            for pronunciation in pronunciations.split(","):
                if num_rows is not None and len(records) > num_rows:
                    break
                records.append(
                    {"word": word.lower(), "pronunciation": pronunciation.strip()}
                )
    pronunciations_df = DataFrame.from_records(records).set_index("word")
    augment_pronunciations_df(pronunciations_df)

    return pronunciations_df


def augment_pronunciations_df(pronunciations: DataFrame) -> None:
    """Adds new fields to the input DataFrame, in place.

    Columns that are added:
    - num_phonemes
    - num_syllables
    """
    pronunciations["num_phonemes"] = pronunciations.pronunciation.apply(count_phonemes)
    pronunciations["num_syllables"] = pronunciations.pronunciation.apply(
        count_syllables
    )

    pronunciations[
        "num_primary_stressed_syllables"
    ] = pronunciations.pronunciation.apply(count_primary_stressed_syllables)


def count_phonemes(pronunciation: str) -> int:
    pronunciation = _remap_diphthongs(pronunciation)
    return sum(1 for phoneme in pronunciation if phoneme not in NON_PHONEME_SYMBOLS)


def count_syllables(pronunciation: str) -> int:
    """Return the number of syllables in a single pronunciation.

    This is approximate as I'm assuming the number of syllabic phonemes is the
    same as the number of vowels. It should be right almost all the time though.
    """
    counts = Counter(_remap_diphthongs(pronunciation))
    return sum(counts[phoneme] for phoneme in SYLLABIC_PHONEMES)


def count_primary_stressed_syllables(pronunciation: str) -> int:
    """Return the number of syllables with primary stress.

    Typically words only have one syllable with primary stress, but compound words or acronyms will
    break that rule. For example, "ai" is /EY1 AY1/ because it's said as two words.
    """
    return sum(1 for phoneme in pronunciation if phoneme == PRIMARY_STRESS)


def _remap_diphthongs(pronunciation: str) -> str:
    """Map each pronunciation to its first vowel."""
    for diphthong in DIPHTHONGS:
        pronunciation = pronunciation.replace(diphthong, diphthong[0])

    return pronunciation
