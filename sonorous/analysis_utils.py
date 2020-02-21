"""Utility functions for use in the analysis Jupyter notebooks."""


import sys
from operator import itemgetter
from typing import Optional, Set, Tuple

from pandas import Series
from matplotlib import pyplot as plt

from sonorous.pronunciationdata import Pronunciation


def plot_next_phoneme_distribution(
    language_model,
    pronunciation: Tuple[str, ...],
    min_probability: float = 0.01,
    phonemes_to_include: Optional[Set[str]] = None,
) -> None:
    """Plot the distribution over the vocabulary of the model's predictions for the next phoneme.

    If `phonemes_to_include` is not None then only those phonemes will be plotted and the
    `min_probability` argument is ignored.
    """
    phoneme_to_probability = Series(language_model.next_probabilities(pronunciation))

    if phonemes_to_include is not None:
        row_mask = phoneme_to_probability.index.map(
            lambda phoneme: phoneme in phonemes_to_include
        )
        phoneme_to_probability = phoneme_to_probability[row_mask]
    else:
        phoneme_to_probability = phoneme_to_probability[
            phoneme_to_probability >= min_probability
        ]
        if len(phoneme_to_probability) == 0:
            print(
                f"No phonemes had probability above {min_probability}", file=sys.stderr
            )
            return

    phoneme_to_probability.sort_values(ascending=False).plot.bar()
    plt.xlabel("Phoneme")
    plt.title(
        "Probability of each phoneme coming after /{}/".format(" ".join(pronunciation))
    )


def plot_pronunciation_probability(
    language_model, pronunciation: Tuple[str, ...]
) -> None:
    """For each phoneme in `pronunciation`, plot P(phoneme | pronunciation so far).

    For example, for the pronunciation /K AE1 T/ the bars in the chart will be:
    * P(K | <START>)
    * P(AE1 | <START> K)
    * P(T | <START> K AE1)
    * P(<END> | <START> K AE1 T)
    """
    probabilities = language_model.conditional_probabilities_of_text(pronunciation)
    phoneme_to_probability = Series(
        probabilities, pronunciation + (language_model.vocab.END,)
    )

    phoneme_to_probability.plot.bar()
    plt.xlabel("Phoneme")
    plt.ylabel("P(i | 0 ... i - n)")
    plt.title("Probability of Each Phoneme in Word")


def interactive_generation(language_model, min_prob: float = 0.01) -> None:
    """Generate a pronunciation with input from the user."""
    pronunciation: Pronunciation = ()
    phoneme = language_model.vocab.START
    while True:
        print("pron:", pronunciation)
        phoneme_to_prob = Series(language_model.next_probabilities(pronunciation))
        phoneme_to_prob = phoneme_to_prob[phoneme_to_prob >= min_prob].sort_values(
            ascending=False
        )
        phoneme_to_prob.plot.bar()
        plt.show()
        print("Choices:", ", ".join(phoneme_to_prob.index))

        phoneme = input("Choose next phoneme:")
        if phoneme == language_model.vocab.END:
            print(pronunciation)
            break

        pronunciation = pronunciation + (phoneme,)


def most_similar_phonemes(language_model, embedding, topn=10):
    """Print the most similar phonemes to the input embedding."""
    other_to_sim = {
        phoneme: cosine_similarity(
            language_model.embedding_for(phoneme), embedding
        ).item()
        for phoneme in language_model.vocab.tokens
        if phoneme not in language_model.vocab.DUMMY_TOKENS
    }

    for other_phoneme, similarity in sorted(
        other_to_sim.items(), key=itemgetter(1), reverse=True
    )[:topn]:
        print(f"[{similarity:.3f}]\t{other_phoneme}")


def analogy(language_model, a, b, c):
    """Identify the phoneme w such that a:b ~ c:w

    argmax(w) over sim(w, c - a + b)
    """
    emb_a = language_model.embedding_for(a)
    emb_b = language_model.embedding_for(b)
    emb_c = language_model.embedding_for(c)

    all_phonemes = language_model.vocab.tokens - set(language_model.vocab.DUMMY_TOKENS)

    phoneme_to_sim = {}
    for phoneme in all_phonemes:
        if phoneme in {a, b, c}:
            continue

        emb_p = language_model.embedding_for(phoneme)
        sim = cosine_similarity(emb_p, emb_c - emb_a + emb_b)
        phoneme_to_sim[phoneme] = sim.item()

    for phoneme, sim in sorted(phoneme_to_sim.items(), key=itemgetter(1), reverse=True)[
        :10
    ]:
        print(f"[{sim:.4f}] {phoneme}")

