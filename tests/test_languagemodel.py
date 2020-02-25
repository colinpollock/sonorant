import os
import tempfile

import numpy as np
import pytest
import torch

from sonorant.languagemodel import (
    LanguageModel,
    ModelParams,
    Vocabulary,
    build_data_loader,
)


def test_build_data_loader():
    texts, vocab = _dummy_texts_and_vocab()

    loader = build_data_loader(texts, vocab, batch_size=2)
    assert loader.batch_size == 2
    input_tensors, target_tensors = loader.dataset.tensors

    # Both tensors should be (num texts, max text length). There are four texts,
    # and the max length is 3 plus 2 for START and END.
    assert input_tensors.shape == (4, 5)
    assert target_tensors.shape == (4, 5)

    def f(token):
        return vocab[token]

    assert input_tensors.tolist() == [
        [vocab.START_IDX, f("a"), f("b"), vocab.END_IDX, vocab.PAD_IDX],
        [vocab.START_IDX, f("b"), f("c"), f("d"), vocab.END_IDX],
        [vocab.START_IDX, f("a"), vocab.END_IDX, vocab.PAD_IDX, vocab.PAD_IDX],
        [vocab.START_IDX, f("a"), f("c"), vocab.END_IDX, vocab.PAD_IDX],
    ]

    assert target_tensors.tolist() == [
        [f("a"), f("b"), vocab.END_IDX, vocab.PAD_IDX, vocab.PAD_IDX],
        [f("b"), f("c"), f("d"), vocab.END_IDX, vocab.PAD_IDX,],
        [f("a"), vocab.END_IDX, vocab.PAD_IDX, vocab.PAD_IDX, vocab.PAD_IDX],
        [f("a"), f("c"), vocab.END_IDX, vocab.PAD_IDX, vocab.PAD_IDX],
    ]

    # Make sure that the first batch in the loader looks right.
    for inputs, targets in loader:
        assert inputs.shape == (2, 5)
        assert targets.shape == (2, 5)
        break


class TestLanguageModel:
    """Some sanity checks of an instantiated but unfit LanguageModel."""

    def setup(self):
        self.texts, self.vocab = _dummy_texts_and_vocab()

        # Using weird numbers to minimize the chance that these are defaults.
        self.model_params = ModelParams(
            rnn_type="lstm",
            embedding_dimension=13,
            hidden_dimension=17,
            num_layers=4,
            max_epochs=471,
            early_stopping_rounds=57,
        )
        self.lm = LanguageModel(self.vocab, self.model_params, torch.device("cpu"))

    def test_layer_dimensions(self):
        """Sanity check that parameters are passed through.
        """
        encoder = self.lm._encoder
        assert encoder.num_embeddings == len(self.vocab)
        assert encoder.embedding_dim == self.model_params.embedding_dimension

        rnn = self.lm._rnn
        assert rnn.input_size == self.model_params.embedding_dimension
        assert rnn.hidden_size == self.model_params.hidden_dimension
        assert rnn.num_layers == self.model_params.num_layers
        assert isinstance(rnn, torch.nn.modules.rnn.LSTM)

        decoder = self.lm._decoder
        assert decoder.in_features == self.model_params.hidden_dimension
        assert decoder.out_features == len(self.vocab)

    def test_embeddings(self):
        embeddings = self.lm.embeddings
        assert embeddings.shape == (
            len(self.vocab),
            self.model_params.embedding_dimension,
        )
        assert isinstance(embeddings, np.ndarray)

    def test_embedding_for(self):
        assert "a" in self.vocab
        embedding = self.lm.embedding_for("a")
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, self.model_params.embedding_dimension)

    def test_probabilities(self):
        text = ("a", "b", "c")
        probabilities = self.lm.conditional_probabilities_of_text(text)

        # There should be four probabilities: P(a|start), P(b|start a), P(c|start a b), P(end|start a b c)
        assert len(probabilities) == 4
        assert all(0 <= prob <= 1 for prob in probabilities)

        total_probability = self.lm.probability_of_text(text)
        assert np.prod(probabilities) == pytest.approx(total_probability)

        perplexity = self.lm.perplexity_of_text(text)
        assert perplexity == pytest.approx(total_probability ** -(1 / 3))

    def test_next_probabilities(self):
        text = ("a", "b")
        next_probabilities = self.lm.next_probabilities(text)
        assert set(next_probabilities) == self.vocab.tokens
        total_prob = sum(next_probabilities.values())
        assert total_prob == pytest.approx(1)

    def test_generate(self):
        generated_text = self.lm.generate(3)
        assert isinstance(generated_text, tuple)
        assert all(token in self.vocab for token in generated_text)

    def test_evaluate(self):
        _, vocab = _dummy_texts_and_vocab()
        dev_texts = [("a", "b", "c"), ("b", "a", "b", "b"), ("d", "a")]

        dev_loader = build_data_loader(dev_texts, vocab)
        dev_loss = self.lm.evaluate(dev_loader)
        assert dev_loss >= 0

    def test_save_and_load(self):
        fd, path = tempfile.mkstemp()

        with open(path, "wb") as fh:
            self.lm.save(fh)

        with open(path, "rb") as fh:
            LanguageModel.load(fh, "cpu")

        os.close(fd)

    def test_save_and_load_missing_keys(self):
        fd, path = tempfile.mkstemp()

        with open(path, "wb") as fh:
            self.lm.save(fh)

        # Loading the saved data. Deleting a single key.
        with open(path, "rb") as fh:
            data = torch.load(fh)
        del data["state_dict"]["_encoder.weight"]
        with open(path, "wb") as fh:
            torch.save(data, fh)

        with open(path, "rb") as fh:
            with pytest.raises(RuntimeError):
                LanguageModel.load(fh, "cpu")

        os.close(fd)

    def test_save_and_load_unexpected_keys(self):
        fd, path = tempfile.mkstemp()

        with open(path, "wb") as fh:
            self.lm.save(fh)

        # Loading the saved data. Adding one unexpected key.
        with open(path, "rb") as fh:
            data = torch.load(fh)
        data["state_dict"]["UNEXPECTED_KEY.weight"] = data["state_dict"][
            "_encoder.weight"
        ]
        with open(path, "wb") as fh:
            torch.save(data, fh)

        with open(path, "rb") as fh:
            with pytest.raises(RuntimeError):
                LanguageModel.load(fh, "cpu")

        os.close(fd)


def test_fit_language_model():
    """Simple test to make sure that after one epoch every parameter tensor has at least some change."""
    texts, vocab = _dummy_texts_and_vocab()

    model_params = ModelParams(
        rnn_type="lstm",
        embedding_dimension=10,
        hidden_dimension=10,
        num_layers=1,
        max_epochs=1,
        early_stopping_rounds=1,
    )

    lm = LanguageModel(vocab, model_params, torch.device("cpu"))
    params_before = [param.detach().tolist() for param in lm.parameters()]
    lm.fit(texts)
    params_after = [param.detach().tolist() for param in lm.parameters()]
    assert len(params_before) == len(params_after)

    for before, after in zip(params_before, params_after):
        assert before != after


class TestVocab:
    def setup(self):
        self.texts = ["a b c".split(), "a b d".split(), "d e f g".split()]
        self.vocab = Vocabulary.from_texts(self.texts)

    def test_encode_text(self):
        text = ("a", "b", "e")
        encoded = self.vocab.encode_text(text)

        # Number of tokens plus START and END.
        assert len(encoded) == 5

        assert encoded[0] == self.vocab.START_IDX
        assert encoded[1] == self.vocab["a"]
        assert encoded[2] == self.vocab["b"]
        assert encoded[3] == self.vocab["e"]
        assert encoded[4] == self.vocab.END_IDX

    def test_encode_text_as_target(self):
        text = ("a", "b", "e")
        encoded = self.vocab.encode_text(text, is_target=True)

        # Number of tokens plus END and PAD
        assert len(encoded) == 5

        assert encoded[0] == self.vocab["a"]
        assert encoded[1] == self.vocab["b"]
        assert encoded[2] == self.vocab["e"]
        assert encoded[3] == self.vocab.END_IDX
        assert encoded[4] == self.vocab.PAD_IDX

    def test_len(self):
        all_tokens = {token for text in self.texts for token in text}
        assert len(all_tokens) == 7
        assert len(self.vocab.DUMMY_TOKENS) == 3
        assert len(self.vocab) == 7 + 3

    def test_getitem(self):
        assert isinstance(self.vocab["a"], int)

        with pytest.raises(KeyError):
            self.vocab["zzz"]

    def test_contains(self):
        assert "a" in self.vocab
        assert "zz" not in self.vocab

    def test_token_from_idx(self):
        a_idx = self.vocab["a"]
        assert self.vocab.token_from_idx(a_idx) == "a"

    def test_cannot_use_dummy_token(self):
        texts = [("hello", Vocabulary.START)]

        with pytest.raises(ValueError):
            Vocabulary.from_texts(texts)


def _dummy_texts_and_vocab():
    # TODO: make fixture
    texts = [("a", "b"), ("b", "c", "d"), ("a",), ("a", "c")]

    return texts, Vocabulary.from_texts(texts)
