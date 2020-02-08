"""A language model over phonemic representations in [1] ARPABET.

The language model in this module is trained on ~125k ARPABET representations of
English words from the [2] CMU Pronouncing Dictionary.

Language models over phonemes can do a few things:
1. Assign a probability to how likely a pronunciation (i.e.phoneme) string is.
2. Generate pronunciations.

Additionally, the embeddings for the phonemes seem to encode some phonetic
properties. For example, adding a Voicing vector to /p/ results in /b/.

[1] https://en.wikipedia.org/wiki/ARPABET
[2] https://en.wikipedia.org/wiki/CMU_Pronouncing_Dictionary
"""

import math
import sys
from typing import Dict, NamedTuple, Optional, Tuple

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import decreased


PAD = 'PAD'
START = 'START'
END = 'END'
PAD_VALUE = 0
START_VALUE = 1
END_VALUE = 2


class ModelParams(NamedTuple):
    """Holder of hyperparameters for a model.

    - rnn_type: a string indicating the type of RNN ('rnn', 'lstm', 'gru').
    - embedding_dimension: the length of each phoneme's embedding vector.
    - hidden_dimension: the size of the RNN/LSTM/GRU's hidden layer.
    - num_layers: number of layers in the RNN. Defaults to 1.
    - learning_rate: defaults to 1--3
    - max_epochs: the maximum number of epochs to train for. Note that this an
      argument to the model rather than the `fit` method so that it's easier to
      automate group all the hyperparameters in one place.
    - early_stopping_rounds: The model will train until the train score stops
      improving. Train error needs to decrease at least every
      early_stopping_rounds to continue training.
    - dropout: defaults to 0
    - l2_strength: L2 regularization strength. Default of 0 is no regularization.
    - batch_size: defaults to 1024
    """
    rnn_type: str
    embedding_dimension: int
    hidden_dimension: int
    num_layers: int
    learning_rate: float=1e-3
    max_epochs: int
    early_stopping_rounds: int
    dropout: float=0
    l2_strength: float=0
    batch_size: int=1024


class PhonemeLM:
    """A language model over phonemes.

    There's not much about this class specific to phonemes and it could be used
    for any vocabulary (words, characters, etc.).

    Arguments to __init__:
    - phoneme_to_idx: a dict mapping phonemes (each a string in ARPABET) to ints.
    - params: a ModelParams.
    - device: 'cuda' or 'cpu'. Defaults to 'gpu' if available.
    """
    def __init__(
        self,
        phoneme_to_idx: Dict[str, int],
        model_params: ModelParams,
        device_name: Optional[str]=None,
    ):
    self.model_params = model_params
    self.device = _get_torch_device_by_name(device_name)
    self.lm = LanguageModel(phoneme_to_idx, model_params, device)


# TODO: move to utils
def _get_torch_device_by_name(device_name: Optional[str]) -> torch.Device:
    # woof test
    if device_name is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    else:
        return torch.device(devic_name )

# TODO: move to utils
def _get_rnn_model_by_name(rnn_name):
    # woof test
    name_to_model = {
        'rnn': nn.RNN,
        'lstm': nn.LSTM,
        'gru': nn.GRU,
    }

    if rnn_name not in name_to_model:
        valid_names = ', '.join(sorted(name_to_model))
        raise ValueError(f"RNN name '{rnn_name}' is invalid. Must be one of ({valid_name})") 
    
    return name_to_model[rnn_name]

class LanguageModel(nn.Module):
    def __init__(
        self,
        token_to_idx: typing.Dict[str, int],
        model_params: ModelParams,
        device: torch.Device,
    ):
        super(LanguageModel, self).__init__()

        # Short name since this'll be referenced a lot.
        self.mp = model_params
        self.token_to_idx = token_to_idx
        self.vocab = sorted(token_to_idx, key=token_to_idx.get)
        self.vocab_size = len(vocab)

        self.embedding = nn.Embedding(self.vocab_size, embedding_dimension)
    
        rnn_model = _get_rnn_model_by_name(model_params.rnn_type)
        self.rnn = rnn_model(
            input_size=model_params.embedding_dimension,
            hidden_size=model_params.hidden_dimension,
            num_layers=model_params.num_layers,
            batch_first=True
        )
    
        self.linear = nn.Linear(model_params.hidden_dimension, self.vocab_size)

        self.dropout = nn.Dropout(model_params.dropout)
        self.l2_strength = model_params.l2_strength

    def forward(self, inputs, hidden_state=None):
        inputs = inputs.to(self.device)

        embedded = self.embedding(inputs)
        rnn_output, new_hidden_state = self.rnn(embedded, hidden_state)
        rnn_output = self.dropout(rnn_output)
        return self.linear(rnn_output), new_hidden_state
    
    def fit(
        self,
        train_texts: List[List[str]],
        dev_texts: Optional[List[List[str]]]=None,
        learning_rate: float=None,
        max_epochs: int=None,
        early_stopping_rounds: int=None,
        batch_size: int=None,
        show_generated: bool=True,
    ):
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
        """
        # Set None parameters passed in to the their default values in `self`.
        lr = learning_rate if learning_rate is not None else self.mp.learning_rate
        max_epochs = max_epochs if max_epochs is not None else self.mp.max_epochs
        early_stopping_rounds = early_stopping_rounds if early_stopping_rounds is not None else self.mp.early_stopping_rounds
        batch_size = batch_size if batch_size is not None else self.mp.batch_size

        self.to(self.device)

        optimizer = Adam(self.parameters(), lr=learning_rate, weight_decay=self.mp.l2_strength)
        criterion = nn.CrossEntropyLoss()

        train_loader = build_data_loader(train_texts, self.token_to_idx, batch_size)
        if dev_texts is not None:
            dev_loader = build_data_loader(dev_texts, self.token_to_idx, batch_size)

        train_losses = []
        dev_losses = []
        for epoch in range(1, max_epochs + 1):
            if not decreased(train_losses, early_stopping_rounds):
                print(
                    f'Early stopping because of no decrease in {early_stopping_rounds} epochs.',
                    file=sys.stderr
                )
                break

            self.train()
            train_epoch_loss = 0
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

            generated_texts = [self.generate(1000, 1) for _ in range(100)]

            num_train_origin, num_assess_origin, num_novel_origin =  \
                self._count_origins(generated_pronunciations, train_pronunciations, assess_pronunciations or [])
            print('\tGenerated: in train: {:.0f}%, assess: {:.0f}%, novel: {:.0f}%'.format(
                num_train_origin, num_assess_origin, num_novel_origin
            ))

            if show_generated:
                for text in generated_texts[:5]:
                    print('\t', ' '.join(text))

        return train_losses, dev_losses

    # TODO: move to utils as something more generic. type parameterize str?
    @staticmethod
    def _count_origins(generated_texts: List[str], train_texts: List[str], dev_texts: List[Str]) -> Tuple[float, float, float]:
        """Count the proportion of generated texts that are in the train or dev
        sets, or are novel texts.

        This can be useful when training a language model to get a sense of
        whether the model has just memorized the training set. This wouldn't be
        useful for a large domain where texts are unlikely to repeat themselves
        entirely (e.g. generated paragraphs of text) but could be useful in a
        much more constrained domain like words of less than 15 characters.

        Returns a triple, which is the proportion of generated texts that are:
        1. in the train set
        2. in the dev set
        3. novel
        """
        train = dev = novel = 0
        for text in generated_texts:
            if text in train-texts:
                train += 1
            elif text in dev_texts:
                assess += 1
            else:
                novel += 1

        total = len(generated_texts)
        return train / total * 100, assess / total * 100, novel / total * 100


    def evaluate(self, loader):
        """Compute the average entropy per symbol on the input loader.

        Loss for every single predicted token is summed and that result is
        divided by the total number of tokens seen. Note that the base for
        entropy is e rather than 2.

        Per-symbol is useful since the return value for different size
        loaders or different batch sizes are the same.
        """
        self.eval()

        criterion = nn.CrossEntropyLoss(reduction='sum')
        loss = 0
        total_tokens = 0

        for inputs, targets in loader:
            total_tokens += inputs.numel()
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                outputs, hidden_states = self(inputs, hidden_state=None)
                loss += criterion(outputs.permute(0, 2, 1), targets)

        return (loss / total_tokens).item()

    def generate(self, max_length: int) -> Tuple[str, ...]:
        """Generate a new text.

        Args:
        - max_length: the maximum number of phonemes to generate.

        Returns: a tuple of tokens.
        """
        self.eval()

        generated = []

        token_idx = self.token_to_idx[START]
        hidden_state = None
        for _ in range(max_length):
            input_ = torch.LongTensor([token_idx]).unsqueeze(0)
            output, hidden_state = self(input_, hidden_state)
            probabilities = F.softmax(output.squeeze(), dim=0)

            token_idx = torch.multinomial(probabilities, 1).item()
            token = self.idx_to_token[token_idx]
            
            if phoneme in (PAD, START, END):
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
        encoded = encode_text(pronunciation, self.phoneme_to_idx)[:-1]
        input_ = torch.LongTensor(encoded).unsqueeze(0)
        output, _ = self(input_)
        probabilities = F.softmax(output, dim=-1)
        next_token_probabilities = probabilities[0, -1, :]

        return {
            self.idx_to_token[idx]: probability.item()
            for idx, probability in enumerate(next_token_probabilities)
        }


    def calculate_probability(self, text: Tuple[str, ...]) -> float:
        """Calculate the probability of the given text.

        Args:
        - text: a sequence of tokens.

        Returns: the probability.
        """
        # TODO: change func call
        encoded_text = encode_text(
            text,
            self.token_to_idx
        )
        output, _ = self(torch.LongTensor(encoded_text).unsqueeze(0))
        output = F.softmax(output, dim=-1).squeeze()

        # At each step a distribution over all tokens is output. This represents
        # the model's predictions for what the next token will be. We pull out
        # the probability for whatever the next phoneme actually is, and end up
        # with the probabilities for each of the actual tokens. Through the
        # chain rule we can get the overall probability for the full text.
        total_logprob = 0
        for step, (input_token__idx, next_token_idx) in enumerate(zip(encoded_text, encoded_text[1:])):
            prob = output[step, next_token_idx].item()
            total_logprob += math.log(prob)

        return math.exp(total_logprob)


    def embedding_for(self, token: str):
        """Return the embedding for the specified phoneme.

        Args:
        - phoneme: an ARPABET phoneme

        Returns: a 1x`embedding_dimension` NumPy array.
        """
        self.eval()
        with torch.no_grad():
            token_idx = self.token_to_idx[token]
            return self.embedding(torch.LongTensor([token_idx]).to(self.device)).cpu().numpy()

    @property
    def embeddings(self):
        """Return the embeddings as a NumPy array."""
        return self.embedding.weight.cpu().detach().numpy()


def build_data_loader(pronunciations, phoneme_to_idx, batch_size=128):
    """Convert the pronunciations into a LongTensor.

    Args:
    - pronunciations: list of pronunciation, each of which is a list of phonemes.
    - phoneme_to_idx: a dict mapping ARPABET phonemes to ints.
    - batch_size: the batch side for the resulting data loader.

    Returns: a DataLoader wrapping a dataset that is a pair (pronunciations,
        targets). Each of those in the pair is a LongTensor of dimension
        (|pronunciations|, max_length). Since we're training a language model to
        predict the next letter, the target for a given pronunciation is all of
        the phonemes shifted one to the right. For example:
          pronunciation: <START> K AE T <END>
          target: K AE T <END>
    """
    pronunciations_as_token_ids = [
        torch.LongTensor(encode_text(pronunciation, phoneme_to_idx, target=False))
        for pronunciation in pronunciations
    ]
    
    target_pronunciations_as_token_ids = [
        torch.LongTensor(encode_text(pronunciation, phoneme_to_idx, target=True))
        for pronunciation in pronunciations
    ]

    dataset = TensorDataset(
        pad_sequence(pronunciations_as_token_ids, batch_first=True, padding_value=PAD_VALUE),
        pad_sequence(target_pronunciations_as_token_ids, batch_first=True, padding_value=PAD_VALUE)
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)



# TODO: test
def encode_text(text: Tuple[str, ...], token_to_idx: [str, int], target: bool=False):
    """Encode a text as an array of int indices.

    Args:
    - text: a tuple of tokens.
    - token_to_idx: a dict mapping tokens to their indices
    - target: bool indicating how to wrap the output in START and END
      indicators. If False then the result is [START, ...tokens..., END]. If
      True then the result is [...tokens..., END, END]. This is useful for
      the target since it offsets each target token by one and lines up the
      RNN's prediction for the next token with the target.

    Returns: a NumPy array of ints representing each phoneme.
    """
    if target is True:
        with_boundaries = text + (END, PAD)
    else:
        with_boundaries = (START,) + text + (END,)
    return np.array([token_to_idx[phoneme] for token in with_boundaries])


class Vocabulary:
    def __init__(self, texts):
        self._token_to_idx = self._build_token_to_idx(texts)
        self._idx_to_token = {
            idx: token for (token, idx) in self._token_to_idx.items()
        }

    # TODO test
    def __len__(self):
        return len(self._token_to_idx)

    def __getitem__(self, token):
        idx = self._token_to_idx.get(token)
        if idx is None:
            raise KeyError(f"Token '{token}' is not in the vocabulary")

        return idx

    # TODO test
    def token_from_idx(self, idx: int) -> str:
        token = self._idx_to_token[idx]
        if token is None:
            raise KeyError(f"Token index '{idx}' is not in the vocabulary")

    @staticmethod
    def _build_token_to_idx(texts: List[List[str]]) -> Dict[str, int]:
        """Build a token-to-index dictionary.
        
        Args:
        - texts: a collection of texts, each of which is a list of tokens.
        
        Returns: a dict mapping tokens to int indices.
        """
        tokens = {token for text in texts for token in text}

        if any(dummy in tokens for dummy in START, END, PAD):
            raise ValueError(f"Input text contains a reserved dummy token")
        tokens.update([START, END, PAD])

        token_to_idx = {PAD: PAD_VALUE, START: START_VALUE, END: END_VALUE}
        
        for text in texts:
            for token in text:
                if token not in token_to_idx:
                    token_to_idx[token] = len(token_to_idx)

        return token_to_idx