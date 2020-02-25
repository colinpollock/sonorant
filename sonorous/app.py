"""The Flask app that exposes the Sonorous interactive website."""


import math
from operator import itemgetter

from flask import Flask, jsonify, render_template, request

from flask_cors import CORS

from sonorous.languagemodel import LanguageModel


# TODO: put this into a config
PORT = 5555

# TODO: load this elsewhere
# TODO: put model name in config
MODEL = LanguageModel.load("models/gru_20_20_1.pt", device_name="cpu")


def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def interactive_app():
        return render_template("interactive_app.html", port=PORT)

    @app.route("/next_probs")
    def next_probs():
        """Return a probability distribution over the vocabulary for the next phoneme.

        Query parameters:
        - so_far: a space-separated string of phonemes, each of which must be in the model's vocab.
        - min_probability: phonemes at or below this threshold won't be returned.

        Returns: a JSON descended sorted list of [phoneme, probability] pairs.

        Note that the returned probabilities are ints between 0 and 100 since those are easier to
        display in a chart.
        """
        so_far = request.args.get("so_far")

        # This can't be zero, or else things like <PAD> will show up.
        default_min_prob = 0.001
        min_prob = float(request.args.get("min_prob", default_min_prob))

        # Pronunciation is a tuple, so if the input string is empty it's an empty tuple
        pronunciation = tuple(so_far.split(" ")) if so_far else ()

        try:
            next_probabilities = MODEL.next_probabilities(pronunciation)
        except KeyError as e:
            return str(e), 400

        next_probabilities = {
            phoneme: math.floor(prob * 100)
            for (phoneme, prob) in next_probabilities.items()
            if prob > min_prob
        }

        sorted_probs = sorted(
            next_probabilities.items(), key=itemgetter(1), reverse=True
        )

        return jsonify(sorted_probs)

    return app
