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
