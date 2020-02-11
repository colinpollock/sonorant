"""Language model implemented in Torch.

The main class of interest is `LanguageModel`. The model is an RNN (either a
vanilla RNN, a LSTM, or a GRU). There are a number of options and
hyperparameters that can be set on the model.

To train the model you need to pass in a list of train texts, each of which is 
a list of tokens.
"""

import math
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from sonorous.utils import get_rnn_model_by_name, get_torch_device_by_name, has_decreased, count_origins, perplexity


class ModelParams(NamedTuple):
    """Holder of hyperparameters for a model.

    - rnn_type: a string indicating the type of RNN ('rnn', 'lstm', 'gru').
    - embedding_dimension: the length of each phoneme's embedding vector.
    - hidden_dimension: the size of the RNN/LSTM/GRU's hidden layer.
    - num_layers: number of layers in the RNN. Defaults to 1.
    - max_epochs: the maximum number of epochs to train for. Note that this an
      argument to the model rather than the `fit` method so that it's easier to
      automate group all the hyperparameters in one place.
    - early_stopping_rounds: The model will train until the train score stops
      improving. Train error needs to decrease at least every
      early_stopping_rounds to continue training.
    - learning_rate: defaults to 1--3
    - dropout: defaults to 0
    - l2_strength: L2 regularization strength. Default of 0 is no regularization.
    - batch_size: defaults to 1024
    """
    rnn_type: str
    embedding_dimension: int
    hidden_dimension: int
    num_layers: int
    max_epochs: int
    early_stopping_rounds: int
    learning_rate: float=1e-3
    dropout: float=0
    l2_strength: float=0
    batch_size: int=1024


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab: 'Vocabulary',
        model_params: ModelParams,
        device_name: Optional[str]
    ):
        super(LanguageModel, self).__init__()

        # Short name since this'll be referenced a lot.
        self.vocab = vocab
        self.mp = model_params
        self.device = get_torch_device_by_name(device_name)

        # Layers
        self._encoder = nn.Embedding(len(self.vocab), model_params.embedding_dimension)
    
        rnn_model = get_rnn_model_by_name(model_params.rnn_type)
        self._rnn = rnn_model(
            input_size=model_params.embedding_dimension,
            hidden_size=model_params.hidden_dimension,
            num_layers=model_params.num_layers,
            batch_first=True
        )

        self._decoder = nn.Linear(model_params.hidden_dimension, len(self.vocab))

        self._dropout = nn.Dropout(model_params.dropout)
        self._l2_strength = model_params.l2_strength

    def forward(self, inputs, hidden_state=None):
        inputs = inputs.to(self.device)

        embedded = self._encoder(inputs)
        rnn_output, new_hidden_state = self._rnn(embedded, hidden_state)
        rnn_output = self._dropout(rnn_output)
        return self._decoder(rnn_output), new_hidden_state
    
    def fit(
        self,
        train_texts: Sequence[Tuple[str]],
        dev_texts: Sequence[Tuple[str]]=None,
        learning_rate: float=None,
        max_epochs: int=None,
        early_stopping_rounds: int=None,
        batch_size: int=None,
        show_generated: bool=True,
    ) -> Tuple[List[float], List[float]]:
        """Fit the model to the training data.

        Args:
        - train_texts: list of lists of tokens. If the vocabulary is words then
          it would look like this: [["a", "cat"], ["the", "dog", ...]].
        - dev_texts: same format as `train_texts`, but used for printing out
          dev set errors during testing.
        - learning_rate: learning rate. Defaults to self.mp.learning_rate if None.
        - max_epochs: the maximum number of epochs to train for. Defaults to 
          self.max_epochs.
        - early_stopping_rounds: The model will train until the dev score stops
          improving. Train error needs to decrease at least every
          early_stopping_rounds to continue training.
        - batch_size: batch size for both train and assess. Defaults to self.batch_size.
        - show_generated: whether to print out generated pronunciations after each
          epoch.

        Returns a pair (train_losses: float, dev_losses: float), which are the
            losses at each epoch.
        """
        # Set None parameters passed in to the their default values in `self`.
        learning_rate = learning_rate if learning_rate is not None else self.mp.learning_rate
        max_epochs = max_epochs if max_epochs is not None else self.mp.max_epochs
        early_stopping_rounds = early_stopping_rounds if early_stopping_rounds is not None else self.mp.early_stopping_rounds
        batch_size = batch_size if batch_size is not None else self.mp.batch_size

        self.to(self.device)

        optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=self.mp.l2_strength)
        criterion = nn.CrossEntropyLoss()

        train_loader = build_data_loader(train_texts, self.vocab, batch_size)
        if dev_texts is not None:
            dev_loader = build_data_loader(dev_texts, self.vocab, batch_size)

        train_losses: List[float] = []
        dev_losses = []
        for epoch in range(1, max_epochs + 1):
            if not has_decreased(train_losses, early_stopping_rounds):
                print(
                    f'Early stopping because of no decrease in {early_stopping_rounds} epochs.',
                    file=sys.stderr
                )
                break

            self.train()
            train_epoch_loss = 0.0
            for batch_num, (inputs, targets) in enumerate(train_loader, start=1):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs, hidden_states = self(inputs, hidden_state=None)
                loss = criterion(outputs.permute(0, 2, 1), targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
                print(
                    'Epoch {}; Batch {} of {}; loss: {:.4f}'.format(
                        epoch,
                        batch_num,
                        len(train_loader),
                        loss.item()
                    ),
                    end='\r'
                )

            train_loss = self.evaluate(train_loader)
            train_losses.append(train_loss)
            status = f'Epoch {epoch}: train loss: {train_loss:.4f}'

            if dev_texts is not None:
                dev_loss = self.evaluate(dev_loader)
                dev_losses.append(dev_loss)
                status += f'\tdev loss: {dev_loss:.4f}'

            print(status)

            generated_texts = [self.generate(1000) for _ in range(100)]

            percent_train_origin, percent_dev_origin, percent_novel_origin = count_origins(generated_texts, train_texts, dev_texts or [])
            print(f'\tGenerated: in train: {percent_train_origin}%, assess: {percent_dev_origin}%, novel: {percent_novel_origin}%')

            if show_generated:
                for text in generated_texts[:5]:
                    print('\t', ' '.join(text))

        return train_losses, dev_losses

    def evaluate(self, loader: DataLoader) -> float:
        """Compute the average entropy per symbol on the input loader.

        Loss for every single predicted token is summed and that result is
        divided by the total number of tokens seen. Note that the base for
        entropy is e rather than 2.

        Per-symbol is useful since the return value for different size
        loaders or different batch sizes are the same.
        """
        self.eval()

        criterion = nn.CrossEntropyLoss(reduction='sum')
        loss = 0.0
        total_tokens = 0

        for inputs, targets in loader:
            total_tokens += inputs.numel()
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                outputs, _ = self(inputs, hidden_state=None)
                loss += criterion(outputs.permute(0, 2, 1), targets).item()

        return (loss / total_tokens)

    def generate(self, max_length: int) -> Tuple[str, ...]:
        """Generate a new text.

        Args:
        - max_length: the maximum number of phonemes to generate.

        Returns: a tuple of tokens.
        """
        self.eval()

        generated = []

        token_idx = self.vocab.START_IDX
        hidden_state = None
        for _ in range(max_length):
            input_ = torch.LongTensor([token_idx]).unsqueeze(0)
            output, hidden_state = self(input_, hidden_state)
            probabilities = F.softmax(output.squeeze(), dim=0)

            token_idx = int(torch.multinomial(probabilities, 1).item())
            token = self.vocab.token_from_idx(token_idx)
            
            if token in self.vocab.DUMMY_TOKENS:
                break

            generated.append(token)

        return tuple(generated)

    def next_probabilities(self, text: Tuple[str]) -> Dict[str, float]:
        """Return the probability of each token coming next.

        Args:
        - text: a sequence of tokens.

        Returns: a dict mapping each phoneme in the vocabulary to a probability.
        """
        # Dropping the final token, which is the END token.
        encoded = self.vocab.encode_text(text)[:-1]
        input_ = torch.LongTensor(encoded).unsqueeze(0)
        output, _ = self(input_)
        probabilities = F.softmax(output, dim=-1)
        next_token_probabilities = probabilities[0, -1, :]

        return {
            self.vocab.token_from_idx(idx): probability.item()
            for idx, probability in enumerate(next_token_probabilities)
        }


    def conditional_probabilities_of_text(self, text: Tuple[str, ...]) -> Tuple[float, ...]:
        """Returns the probability of each token in `text`.

        If `text` has two tokens (t1 and t2) then the returned tuple will be of length 3:
        1. P(t1|START)
        2. P(t2|START t1)
        3. P(END|START t1 t2)
        """
        encoded_text = self.vocab.encode_text(text)
        output, _ = self(torch.LongTensor(encoded_text).unsqueeze(0))
        output = F.softmax(output, dim=-1).squeeze()

        # At each step a distribution over all tokens is output. This represents
        # the model's predictions for what the next token will be. We pull out
        # the probability for whatever the next phoneme actually is, and end up
        # with the probabilities for each of the actual tokens. Through the
        # chain rule we can get the overall probability for the full text.
        probabilities = []
        for step, (input_token__idx, next_token_idx) in enumerate(zip(encoded_text, encoded_text[1:])):
            probabilities.append(output[step, next_token_idx].item())

        return tuple(probabilities)

    def probability_of_text(self, text: Tuple[str, ...]) -> float:
        """Calculate the probability of the given text.

        Args:
        - text: a sequence of tokens.

        Returns: the probability.
        """
        total_logprob = 0.0
        for probability in self.conditional_probabilities_of_text(text):
            total_logprob += math.log(probability)
        
        return math.exp(total_logprob)

    def perplexity_of_text(self, text: Tuple[str, ...]) -> float:
        """Calculate the perplexity of the given text.

        Note that this calls `probability_of_text`. That then calls
        `conditional_probabilities_of_text`, which is fairly expensive.
        """
        probability = self.probability_of_text(text)
        return perplexity(probability, len(text))


    def embedding_for(self, token: str):
        """Return the embedding for the specified phoneme.

        Args:
        - phoneme: an ARPABET phoneme

        Returns: a 1x`embedding_dimension` NumPy array.
        """
        self.eval()
        with torch.no_grad():
            token_idx = self.vocab[token]
            return self._encoder(torch.LongTensor([token_idx]).to(self.device)).cpu().numpy()

    @property
    def embeddings(self):
        """Return the embeddings as a NumPy array."""
        return self._encoder.weight.cpu().detach().numpy()
    
    def save(self, file_handle):
        """Save a file to disk."""
        data = {
            'token_to_idx': self.vocab.token_to_idx,
            'model_params': self.mp._asdict(),
            'state_dict': self.state_dict()
        }

        torch.save(data, file_handle)
    
    @staticmethod
    def load(file_handle, device_name: str) -> 'LanguageModel':
        """Load a model from disk that has been saved using the `save` method."""
        data = torch.load(file_handle)
        vocab = Vocabulary(data['token_to_idx'])
        model_params = ModelParams(**data['model_params'])
        state_dict = data['state_dict']

        model = LanguageModel(vocab, model_params, device_name)
        model.load_state_dict(state_dict)

        return model


class Vocabulary:
    """A vocabulary over tokens and operations on it.

    A Vocabulary is initialized by passing in a list of texts, where a text is
    a list of tokens. It is immutable and cannot be udpated after being
    initialized.

    >>> vocab = Vocabulary.from_texts([['a', 'b', 'c'], ['c', 'd']])
    >>> 'a' in vocab
    True
    >>> len(vocab)
    4
    >>> vocab.token_from_idx(vocab['b'])
    'b'
    """
    PAD = '<PAD>'
    START = '<START>'
    END = '<END>'
    DUMMY_TOKENS = (PAD, START, END)
    PAD_IDX = 0
    START_IDX = 1
    END_IDX = 2

    def __init__(self, token_to_idx: Dict[str, int]):
        self.token_to_idx = token_to_idx

        self._idx_to_token = {
            idx: token for (token, idx) in token_to_idx.items()
        }

        # Note that this only works because a Vocabulary is immutable
        # and no tokens can be added outside of __init__.
        self.tokens = set(token_to_idx)
        self.indices = set(self._idx_to_token)
    
    @classmethod
    def from_texts(cls, texts: List[List[str]]) -> 'Vocabulary':
        token_to_idx = cls._build_token_to_idx(texts)
        return Vocabulary(token_to_idx)

    def encode_text(self, text: Tuple[str, ...], is_target: bool=False):
        """Encode a text as an array of int indices.

        Args:
        - text: a tuple of tokens.
        - is_target: bool indicating how to wrap the output in START and END
        indicators. If False then the result is [START, ...tokens..., END]. If
        True then the result is [...tokens..., END, END]. This is useful for
        the target since it offsets each target token by one and lines up the
        RNN's prediction for the next token with the target.

        Returns: a NumPy array of ints representing each phoneme.
        """
        if is_target is True:
            with_boundaries = text + (self.END, self.PAD)
        else:
            with_boundaries = (self.START,) + text + (self.END,)
        return np.array([self[token] for token in with_boundaries])

    def __getitem__(self, token: str) -> int:
        idx = self.token_to_idx.get(token)
        if idx is None:
            # Note that this could also just return an UNK value, which would be
            # another dummy like PAD, but I haven't needed it yet.
            raise KeyError(f"Token '{token}' is not in the vocabulary")

        return idx
    
    def __contains__(self, token: str) -> bool:
        return token in self.token_to_idx

    def token_from_idx(self, idx: int) -> str:
        token = self._idx_to_token[idx]
        if token is None:
            raise KeyError(f"Token index '{idx}' is not in the vocabulary")
        
        return token
    
    def __eq__(self, other):
        if not isinstance(other, Vocabulary):
            return False
        
        return self.token_to_idx == other.token_to_idx

    def __len__(self):
        return len(self.token_to_idx)
    
    @classmethod
    def _build_token_to_idx(cls, texts: List[List[str]]) -> Dict[str, int]:
        """Build a token-to-index dictionary.
        
        Args:
        - texts: a collection of texts, each of which is a list of tokens.
        
        Returns: a dict mapping tokens to int indices.
        """
        tokens = {token for text in texts for token in text}

        if any(dummy in tokens for dummy in cls.DUMMY_TOKENS):
            raise ValueError(f"Input text contains a reserved dummy token")
        tokens.update(cls.DUMMY_TOKENS)

        token_to_idx = {cls.PAD: cls.PAD_IDX, cls.START: cls.START_IDX, cls.END: cls.END_IDX}
        
        for text in texts:
            for token in text:
                if token not in token_to_idx:
                    token_to_idx[token] = len(token_to_idx)

        return token_to_idx
    

def build_data_loader(texts: Sequence[Tuple[str]], vocab: Vocabulary, batch_size=128) -> DataLoader:
    """Convert a list of texts into a LongTensor.

    Args:
    - texts: list of text, each of which is a list of tokens.
    - vocab
    - batch_size: the batch side for the resulting data loader.

    Returns: a DataLoader wrapping a dataset that is a pair (inputs, targets). Each
      of those in the pair is a LongTensor of dimension (num texts, max text length).
      Since we're training a language model to predict the next token, the target for
      a given input text is each of the tokens shifted one to the right.
      input: <START> K AE T <END>
      target: K AE T <END> <PAD>
    """
    input_tensors = [
        torch.LongTensor(vocab.encode_text(text, is_target=False))
        for text in texts
    ]
    
    target_tensors = [
        torch.LongTensor(vocab.encode_text(text, is_target=True))
        for text in texts
    ]

    dataset = TensorDataset(
        pad_sequence(input_tensors, batch_first=True, padding_value=vocab.PAD_IDX),
        pad_sequence(target_tensors, batch_first=True, padding_value=vocab.PAD_IDX)
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)