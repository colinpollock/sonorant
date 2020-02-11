Sonorous
========
This is a language model for English sounds. It's an RNN trained in PyTorch on words from the [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict), a corpus of phonemic transcriptions of words. Each word is a sequence of [ARPABET](https://en.wikipedia.org/wiki/ARPABET) characters. So for example the word "fish" is /F IH1 SH/.

There are a few interesting uses for this model:
1. Calculate the probability of a given word. For example, the model thinks "bar" is very likely (it seems like English) whereas "pshew" is very unlikely.
2. Generate novel words. For example, it generated /AO1 R P ER0 AH0 T S/, which might be written as "orperuts".
3. Construct representations of phonetic properties of sounds. For example, the sounds /B/ and /P/ are both made by stopping air by closing the lips and then opening the lips. /B/ is "voiced", meaning that your vocal chords are vibrating when you make that sound. It's possible to aumatically learn representations for things like "voiced" and then add them to voiceless phonemes. For example, VOICED + /P/ is /B/, VOICED + /F/ is /V/, etc.


## Setup ##
1. Create and source a virtualenv. `python -m venv venv && source venv/bin/activate`
2. Install dependencies. `pip install -r requirements.txt`
3. Download the CMU dictionary into NLTK. `python -c "import nltk; nltk.download('cmudict')"`

## Related Work ##
* https://github.com/aparrish/pronouncingpy
* https://github.com/Kyubyong/g2p
* https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
* https://www.stonybrook.edu/commcms/amp2019/_pdf/_Poster2/2.13Nelson.pdf


## Language Model ##
The majority of code in this repo isn't specific to pronunciations. `sonorous/languagemodel.py` contains a class `LanguageModel`, which is a neural language model that can be trained on any type of texts. In this section I'll outline its functionalities.

Import these three classes.
>>> from sonorous.languagemodel import LanguageModel, ModelParams, Vocabulary

Define the train and dev texts. Build a `Vocabulary` from the texts, which handles the mapping of tokens like "a" to integer indices.
>>> train_texts = [("a", "cat", "ate"), ("some", "dogs", "slept")]
>>> dev_texts = [("some", "cat", "ate"), ("dogs", "slept")]
>>> vocab = Vocabulary.from_texts(train_texts + dev_texts)
>>> len(vocab)
Out[1]: 9
>>> vocab['a']
Out[2]: 3

Define ModelParams, which encapsulate the hyperparameters for a model. This is a useful abstraction that allows parameters to be passed around as a group rather than one by one and aids serialization.
>>> model_params = ModelParams(
    rnn_type="gru",
    embedding_dimension=50,
    hidden_dimension=30,
    num_layers=1,
    max_epochs=2,
    early_stopping_rounds=5,
)

A model is defined by a vocabulary, model parameters, and the name of the device on which it'll run. Any Torch devices will work, but you probably want "cuda" if you're running on a GPU and "cpu"
otherwise.
>>> model = LanguageModel(vocab, model_params, device_name="cpu")

To train a model pass a sequence of train texts and dev texts to the `fit` function. At the end of every epoch the model prints out the loss for the dev set. Note that `max_epochs` and a few other parameters set in `model_params` can be overriddedn by passing them to `fit`.
>>> train_errors, dev_errors = model.fit(train_texts, dev_texts, max_epochs=10)

Now I'll run through some basic operations over a trained model.

You can calculate the perplexity of any text. Perplexity is basically the length normalized, inverse probability. length normalized.
>>> model.perplexity_of_text(dev_texts[0])
Out[3]: 14.466998156671036

You can pass in a tuple of tokens and the model will return a probability distribution over the vocabulary for its predictions of which token will come next.
>>> model.next_probabilities(("a", "cat"))
Out[4]:
{'<PAD>': 0.0008051212062127888,
 '<START>': 0.004382132086902857,
 '<END>': 0.010958804748952389,
 'a': 0.00777098536491394,
 'cat': 0.005946762394160032,
 'ate': 0.944864809513092,
 'some': 0.00814706552773714,
 'dogs': 0.0017555770464241505,
 'slept': 0.015368768945336342}

You can generate novel texts.
>>> model.generate(max_length=1000)
Out[5]: ('dogs', 'cat')

You can save the model to disk and then load it.
>>> with open('model.pt', 'wb') as fh:
    model.save(fh)

>>> with open('model.pt', 'rb') as fh:
    the_same_model = LanguageModel.load(fh, device_name='cpu')
