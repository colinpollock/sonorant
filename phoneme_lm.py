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

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import decreased


PAD = '<PAD>'
START = '<W>'
END = '</W>'
PAD_VALUE = 0
START_VALUE = 1
END_VALUE = 2


class PhonemeLM(nn.Module):
    """A language model over phonemes.

    There's not much about this class specific to phonemes and it could be used
    for any vocabulary (words, characters, etc.).

    Arguments to __init__:
    - phoneme_to_idx: a dict mapping phonemes (each a string in ARPABET) to ints.
    - idx_to_phoneme: a dict mapping ints to phonemes (each a string in ARPABET).
    - rnn_type: a string indicating the type of RNN ('rnn', 'lstm', 'gru').
    - embedding_dimension: the length of each phoneme's embedding vector.
    - hidden_dimension: the size of the RNN/LSTM/GRU's hidden layer.
    - num_layers: number of layers in the RNN. Defaults to 1.
    - device: 'cuda' or 'cpu'. Defaults to 'gpu' if available.
    - lr: learning rate.
    - max_epochs: the maximum number of epochs to train for. Note that this an
      argument to the model rather than the `fit` method so that it's easier to
      automate group all the hyperparameters in one place.
    - early_stopping_rounds: The model will train until the dev score stops
      improving. Dev error needs to decrease at least every
      early_stopping_rounds to continue training.
    - dropout
    - batch_size
    """
    def __init__(
        self,
        phoneme_to_idx,
        rnn_type,
        embedding_dimension,
        hidden_dimension,
        num_layers=1,
        device=None,
        lr=1e-3,
        max_epochs=100,
        early_stopping_rounds=5,
        dropout=0,
        batch_size=256,
    ):
        super(PhonemeLM, self).__init__()
        
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_rounds = early_stopping_rounds
        
        self.phoneme_to_idx = phoneme_to_idx
        self.idx_to_phoneme = {idx: phoneme for (phoneme, idx) in phoneme_to_idx.items()}
        self.vocab = sorted(phoneme_to_idx, key=phoneme_to_idx.get)
        self.vocab_size = len(phoneme_to_idx)

        self.embedding = nn.Embedding(self.vocab_size, embedding_dimension)

        rnn_model = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'gru': nn.GRU,
        }[rnn_type]

        self.rnn = rnn_model(
            input_size=embedding_dimension,
            hidden_size=hidden_dimension,
            num_layers=num_layers,
            batch_first=True
        )
    
        self.linear = nn.Linear(hidden_dimension, self.vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden_state=None):
        inputs = inputs.to(self.device)

        embedded = self.embedding(inputs)
        rnn_output, new_hidden_state = self.rnn(embedded, hidden_state)
        rnn_output = self.dropout(rnn_output)
        return self.linear(rnn_output), new_hidden_state
    
    def fit(
        self,
        train_pronunciations,
        assess_pronunciations=None,
        lr=None,
        max_epochs=None,
        early_stopping_rounds=None,
        batch_size=None
    ):
        """Fit on the pronunciations.
        Args:
        - train_pronunciations: list of pronunciations, each of which is a
          sequence of ARPABET, that the model is trained on.
        - assess_pronunciations: same format as above. Used as a holdout set to
          evaluate the model after each epoch.
        - lr: learning rate. Defaults to self.lr if None.
        - max_epochs: the maximum number of epochs to train for. Defaults to 
          self.max_epochs.
        - early_stopping_rounds: The model will train until the dev score stops
          improving. Dev error needs to decrease at least every
          early_stopping_rounds to continue training.
        - batch_size: batch size for both train and assess. Defaults to self.batch_size.
        """
        # Set None parameters passed in to the their default values in `self`.
        lr = lr if lr is not None else self.lr
        max_epochs = max_epochs if max_epochs is not None else self.max_epochs
        early_stopping_rounds = early_stopping_rounds if early_stopping_rounds is not None else self.early_stopping_rounds
        batch_size = batch_size if batch_size is not None else self.batch_size

        self.to(self.device)

        optimizer = Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_loader = build_data_loader(train_pronunciations, self.phoneme_to_idx, batch_size)
        if assess_pronunciations is not None:
            assess_loader = build_data_loader(assess_pronunciations, self.phoneme_to_idx, batch_size)

        train_losses = []
        assess_losses = []
        for epoch in range(1, max_epochs + 1):
            if not decreased(assess_losses, early_stopping_rounds):
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

            if assess_pronunciations is not None:
                assess_loss = self.evaluate(assess_loader)
                assess_losses.append(assess_loss)
                status += f'\tassess loss: {assess_loss:.4f}'

            print(status)

            for _ in range(5):
                generated_pronunciation = ' '.join(self.generate(10, 1))
                print('\t', generated_pronunciation)

        return train_losses, assess_losses

    def evaluate(self, loader):
        self.eval()

        criterion = nn.CrossEntropyLoss()
        loss = 0
        for inputs, targets in loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                outputs, hidden_states = self(inputs, hidden_state=None)
                loss += criterion(outputs.permute(0, 2, 1), targets)

        return (loss / len(loader)).item()

    def generate(self, max_length, temperature):
        """Generate a pronunciation.

        Args:
        - max_length: the maximum number of phonemes to generate.
        - temperature: the amount of diversity in the generated pronunciation.

        Returns: a list of ARPABET phonemes.
        """
        self.eval()

        generated = []

        phoneme_idx = self.phoneme_to_idx[START]
        hidden_state = None
        for _ in range(max_length):
            input_ = torch.LongTensor([phoneme_idx]).unsqueeze(0)
            output, hidden_state = self(input_, hidden_state)
            probabilities = F.softmax(output.squeeze(), dim=0)

            # TODO: use temperature
            phoneme_idx = torch.multinomial(probabilities, 1).item()
            phoneme = self.idx_to_phoneme[phoneme_idx]
            
            if phoneme in (PAD, END):
                break

            generated.append(phoneme)

        return generated


    def calculate_probability(self, pronunciation):
        """Calculate the probability of the given pronunciation.

        Args:
        - pronunciation: a sequence of ARPABET phonemes, each as a string.

        Returns: the probability as a float.
        """

        # TODO: fix for single phonemes
        # TODO: some kind of length normalization

        encoded_pronunciation = encode_pronunciation(
            pronunciation,
            self.phoneme_to_idx
        )
        output, _ = self(torch.LongTensor(encoded_pronunciation).unsqueeze(0))
        output = F.softmax(output, dim=-1).squeeze()
        
        # At each step (each phoneme) in the output there is a distribution over
        # all phonemes. This represents the model's predictions for what the
        # next phoneme will be.` We pull out the probability for whatever the
        # next phoneme actually is, and end up with the probabilities for each
        # of the actual phonemes. Through the chain rule we can get the overall
        # probability for the pronunciation.
        total_logprob = 0
        for step, (input_phoneme_idx, next_phoneme_idx) in enumerate(zip(encoded_pronunciation, encoded_pronunciation[1:])):
            # next_phoneme_idx = self.phoneme_to_idx[next_phoneme]
            prob = output[step, next_phoneme_idx].item()
            total_logprob += math.log(prob)

        return math.exp(total_logprob)


    def embedding_for(self, phoneme):
        """Return the embedding for the specified phoneme.

        Args:
        - phoneme: an ARPABET phoneme

        Returns: a 1x`embedding_dimension` NumPy array.
        """
        self.eval()
        with torch.no_grad():
            phoneme_idx = self.phoneme_to_idx[phoneme]
            return self.embedding(torch.LongTensor([phoneme_idx]).to(self.device)).cpu().numpy()

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
        torch.LongTensor(encode_pronunciation(pronunciation, phoneme_to_idx, target=False))
        for pronunciation in pronunciations
    ]
    
    target_pronunciations_as_token_ids = [
        torch.LongTensor(encode_pronunciation(pronunciation, phoneme_to_idx, target=True))
        for pronunciation in pronunciations
    ]

    data = pad_sequence(pronunciations_as_token_ids, batch_first=True)
    
    
    dataset = TensorDataset(
        pad_sequence(pronunciations_as_token_ids, batch_first=True, padding_value=PAD_VALUE),
        pad_sequence(target_pronunciations_as_token_ids, batch_first=True, padding_value=PAD_VALUE)
    )
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

def decode(phoneme_idxs, id_to_phoneme):
    """Return the phonemes as a string for the given int representation."""
    return ' '.join(id_to_phoneme[phoneme_idx] for phoneme_idx in phoneme_idxs)


def build_vocab(pronunciations):
    """Build a phoneme-to-index dictionary.
    
    Args:
    - pronunciations: a collection of lists of phonemes.
    
    Returns: a dict mapping phonemes to int indices.
    """
    phonemes = {phoneme for pronunciation in pronunciations for phoneme in pronunciation}
    phonemes.update([START, END, PAD])

    phoneme_to_idx = {PAD: PAD_VALUE, START: START_VALUE, END: END_VALUE}
    
    for pronunciation in pronunciations:
        for phoneme in pronunciation:
            if phoneme not in phoneme_to_idx:
                phoneme_to_idx[phoneme] = len(phoneme_to_idx)

    idx_to_phoneme = {idx: phoneme for (phoneme, idx) in phoneme_to_idx.items()}

    return phoneme_to_idx, idx_to_phoneme


def encode_pronunciation(pronunciation, phoneme_to_idx, target=False):
    """Encode a pronunciation

    Args:
    - pronuncation: a sequence of phonemes, each of which is a string.
    - phoneme_to_idx: a dict mapping phoneme strings to their indices.
    - target: bool indicating how to wrap the output in START and END
      indicators. If False then the result is [START, ...phonemes..., END]. If
      True then the result is [...phonemes..., END, END]. This is useful for
      the target since it offsets each target phoneme by one and lines up the
      RNN's prediction for the next phoneme with the target.

    Returns: a NumPy array of ints representing each phoneme.
    """
    if target is True:
        with_boundaries = pronunciation + [END, PAD]
    else:
        with_boundaries = [START] + pronunciation + [END]    
    return np.array([phoneme_to_idx[phoneme] for phoneme in with_boundaries])

