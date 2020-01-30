from collections import defaultdict

import numpy as np
import pandas as pd

from utils import load_data

def ngrams(tokens, n):
    for i in range(0, len(tokens) - n + 1):
        yield ''.join(map(str, tokens[i:i+n]))


def count_phoneme_and_letter(df, num_letters=1):
    counter = defaultdict(lambda: defaultdict(int))

    for idx, row in df.iterrows():
        for letter_ngram in ngrams(row['word'], num_letters):
            for phoneme in row['pronunciation']:
                counter[letter_ngram][phoneme] += 1

    return pd.DataFrame(counter)


def compute_ooe(counts_df):
    total = counts_df.sum().sum()
    observed_joint_probs = counts_df / total
    P_letter = observed_joint_probs.sum(axis=0)
    P_phoneme = observed_joint_probs.sum(axis=1)

    expected_joint_probs = pd.DataFrame(
        np.outer(P_phoneme, P_letter),
        index=observed_joint_probs.index,
        columns=observed_joint_probs.columns
    )
    
    return observed_joint_probs / expected_joint_probs


def show_best(ooe_df, counts_df, phoneme, min_support=100):
    mask = counts_df >= min_support
    for_phoneme = ooe_df[mask].loc[phoneme]
    good_ortho = for_phoneme[for_phoneme > 1]
    return good_ortho.sort_values(ascending=False)