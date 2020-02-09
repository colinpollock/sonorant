"""Interface for loading pronunciation data."""


from string import ascii_lowercase

import nltk
import pandas as pd


ALLOWABLE_LETTERS = set(ascii_lowercase)
CMU = nltk.corpus.cmudict.dict()


def load_data(include_stress=True):
    """Return a DataFrame with columns for `word` and `pronunciation`."""
    records = []
    for word, pronunciations in CMU.items():
        if not all(letter in ALLOWABLE_LETTERS for letter in word):
            continue

        for pronunciation in pronunciations:
            if not include_stress:
                pronunciation = _strip_stress(pronunciation)
            records.append({'word': word, 'pronunciation': tuple(pronunciation)})

    df = pd.DataFrame(records).set_index('word')
    df['pronunciation_string'] = df.pronunciation.apply(' '.join)
    df['length'] = df.pronunciation.apply(len)
    return df

def _strip_stress(pronunciation):
    new_pronunciation = []
    for phoneme in pronunciation:
        for stress in '012':
            phoneme = phoneme.replace(stress, '')
        new_pronunciation.append(phoneme)

    return new_pronunciation