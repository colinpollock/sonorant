"""Interface for loading pronunciation data."""


from collections import Counter
from string import ascii_uppercase
from typing import Any, Dict, List, Set, Tuple

from pandas import DataFrame

Pronunciation = Tuple[str, ...]

CMUDICT_FILEPATH = "sonorous/data/cmudict-0.7b-ipa.txt"

ACCEPTABLE_WORD_CHARACTERS = set(ascii_uppercase) | {"'", "_", ".", "-"}


PRIMARY_STRESS = "ˈ"
SECONDARY_STRESS = "ˌ"
LENGTH_SYMBOL = "ː"
NON_PHONEME_SYMBOLS = {PRIMARY_STRESS, SECONDARY_STRESS, LENGTH_SYMBOL}


DIPHTHONGS = {
    "oʊ",
    "aʊ",
    "aɪ",
    "eɪ",
    "ɔɪ",
}


def _get_syllabic_phonemes():
    syllabic_phonemes = {
        "i",
        "e",
        "ʊ",
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
    } | DIPHTHONGS
    for _phoneme in list(syllabic_phonemes):
        syllabic_phonemes.add(_phoneme + LENGTH_SYMBOL)

    return syllabic_phonemes


SYLLABIC_PHONEMES = _get_syllabic_phonemes()


def load_pronunciations(num_rows: int = None) -> DataFrame:
    """Return a DataFrame with columns for `word` and `pronunciation`."""
    records: List[Dict[str, Any]] = []

    with open(CMUDICT_FILEPATH, "r") as fh:
        for line in fh:
            word, pronunciations = line.strip().split("\t")

            if not all(letter in ACCEPTABLE_WORD_CHARACTERS for letter in word):
                continue

            for pronunciation in pronunciations.split(","):
                if num_rows is not None and len(records) > num_rows:
                    break
                records.append(
                    {
                        "word": word.lower(),
                        "pronunciation": tokenize_pronunciation_string(
                            pronunciation.strip()
                        ),
                    }
                )
    pronunciations_df = DataFrame.from_records(records).set_index("word")
    augment_pronunciations_df(pronunciations_df)

    # I really only want pronunciations that are for single words. By definition, single words
    # should have only one primary stress so I'm dropping words that don't have just one. This
    # eliminates compound words ("solid-state") and acronyms ("ai"), which have more than one
    # primary stress. It also eliminates words with no primary stress, which don't make sense to me.
    # These tend to be function words, so I'm guessing the idea is that a neutral vowel can be
    # unstressed.
    return pronunciations_df[pronunciations_df.num_primary_stressed_syllables == 1]


def tokenize_pronunciation_string(pronunciation_string: str) -> Pronunciation:
    """Turn the string into a tuple of symbols.

    Most elements in the tuple are individual phonemes, but some are groups:
    - symbols followed by "ː" are grouped
    - affricates are grouped

    """
    symbols = tuple(pronunciation_string)
    new_symbols = []

    pairs = [
        ("d", "ʒ"),
        ("t", "ʃ"),
        ("a", "ɪ"),
        ("a", "ʊ"),
        ("e", "ɪ"),
        ("ɔ", "ɪ"),
        ("o", "ʊ"),
    ]

    idx = 0
    while idx < len(symbols):
        symbol = symbols[idx]
        if idx == len(symbols) - 1:
            new_symbols.append(symbol)
            break

        next_symbol = symbols[idx + 1]
        if next_symbol == LENGTH_SYMBOL:
            new_symbols.append(symbol + LENGTH_SYMBOL)
            idx += 2
            continue
        else:
            if any(
                symbol == first and next_symbol == second for (first, second) in pairs
            ):
                new_symbols.append(symbol + next_symbol)
                idx += 2
                continue
            else:
                new_symbols.append(symbol)
                idx += 1
    return tuple(new_symbols)


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


def count_phonemes(pronunciation: Pronunciation) -> int:
    return sum(1 for symbol in pronunciation if symbol not in NON_PHONEME_SYMBOLS)


def count_syllables(pronunciation: Pronunciation) -> int:
    """Return the number of syllables in a single pronunciation.

    This is approximate as I'm assuming the number of syllabic phonemes is the
    same as the number of vowels. It should be right almost all the time though.
    """
    counts = Counter(pronunciation)
    return sum(counts[phoneme] for phoneme in SYLLABIC_PHONEMES)


def count_primary_stressed_syllables(pronunciation: Pronunciation) -> int:
    """Return the number of syllables with primary stress.

    Typically words only have one syllable with primary stress, but compound words or acronyms will
    break that rule. For example, "ai" is /EY1 AY1/ because it's said as two words.
    """
    return sum(1 for phoneme in pronunciation if phoneme == PRIMARY_STRESS)
