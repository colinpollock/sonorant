"""General utilities """
from typing import Optional, Sequence, Tuple, Type

import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch import nn


def split_data(
    data: DataFrame,
    dev_proportion: float,
    test_proportion: float,
    random_state: int = 47,
):
    """Return three DataFrames (train, dev, test)."""
    train, dev = train_test_split(
        data, test_size=dev_proportion + test_proportion, random_state=random_state
    )

    dev, test = train_test_split(
        dev,
        test_size=test_proportion / (dev_proportion + test_proportion),
        random_state=random_state,
    )

    return train, dev, test


def perplexity(probability: float, length: int) -> float:
    """Return the perplexity of a sequence with specified probability and length."""
    return probability ** -(1 / length)


def has_decreased(scores, in_last):
    """Return True iff the score in the last `in_last` descended."""
    if in_last >= len(scores):
        return True

    last = scores[-(in_last + 1)]
    for score in scores[-in_last:]:
        if score < last:
            return True
        last = score

    return False


def count_origins(
    generated_texts: Sequence[Tuple[str, ...]],
    train_texts: Sequence[Tuple[str, ...]],
    dev_texts: Sequence[Tuple[str, ...]],
) -> Tuple[int, int, int]:
    """Count the proportion of generated texts that are in the train or dev
    sets, or are novel texts.

    This can be useful when training a language model to get a sense of
    whether the model has just memorized the training set. This wouldn't be
    useful for a large domain where texts are unlikely to repeat themselves
    entirely (e.g. generated paragraphs of text) but could be useful in a
    much more constrained domain like words of less than 15 characters.

    Args:
    - generated_texts: a Sequence of texts, each of which is a tuple of tokens.
    - train_texts: same format as `generated_texts`.
    - dev_texts: same format as `generated_texts`.

    Returns a triple, which is the proportion of generated texts that are:
    1. in the train set
    2. in the dev set
    3. novel
    """
    train = dev = novel = 0
    for text in generated_texts:
        if text in train_texts:
            train += 1
        elif text in dev_texts:
            dev += 1
        else:
            novel += 1

    total = len(generated_texts)
    return (int(train / total * 100), int(dev / total * 100), int(novel / total * 100))


def get_rnn_model_by_name(rnn_name: str) -> Type[nn.modules.rnn.RNNBase]:
    """Return a nn.RNN, nn.LSTM, or nn.GRU by name."""
    name_to_model = {
        "rnn": nn.RNN,
        "lstm": nn.LSTM,
        "gru": nn.GRU,
    }

    if rnn_name not in name_to_model:
        valid_names = ", ".join(sorted(name_to_model))
        raise ValueError(
            f"RNN name '{rnn_name}' is invalid. Must be one of ({valid_names})"
        )

    return name_to_model[rnn_name]


def get_torch_device_by_name(device_name: Optional[str] = None) -> torch.device:
    """Return a `torch.device` for the specified device name.

    If `device_name` is None then CUDA will be used if it's available and CPU if not.
    """
    cuda_is_available = torch.cuda.is_available()

    # pylint: disable=no-else-return
    if device_name is None:
        if cuda_is_available:
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        if device_name == "cuda" and not cuda_is_available:
            raise ValueError("cuda is not available")

        return torch.device(device_name)
