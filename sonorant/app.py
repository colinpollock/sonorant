"""The Flask app that exposes the Sonorant interactive website."""

from operator import itemgetter

from flask import Flask, jsonify, render_template, request

from flask_cors import CORS

from sonorant.languagemodel import LanguageModel
from sonorant.utils import truncate


MODEL = LanguageModel.load("models/gru_20_20_1.pt", device_name="cpu")

# Minimum probability for returning a  phoneme. This can't be zero, or else things like <PAD> will
# show up.
DEFAULT_MIN_PROB = 0.001


def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def interactive_app():
        next_probs_endpoint = f"next_probs?so_far="
        return render_template(
            "interactive_app.html", next_probs_endpoint=next_probs_endpoint
        )

    @app.route("/next_probs")
    def next_probs():
        """Return a probability distribution over the vocabulary for the next phoneme.

        Query parameters:
        - so_far: a space-separated string of phonemes, each of which must be in the model's vocab.
        - min_probability: phonemes at or below this threshold won't be returned. Must be greater
          than zero (to avoid showing *everything*) and less than 1.

        Returns: a JSON descended sorted list of [phoneme, probability] pairs.

        Note that the returned probabilities are ints between 0 and 100 since those are easier to
        display in a chart.
        """
        so_far = request.args.get("so_far")

        try:
            min_prob = float(request.args.get("min_prob", DEFAULT_MIN_PROB))
        except ValueError as e:
            return e.message, 400

        if not 0 < min_prob <= 1:
            return "min_prob must be greater than 0 and less than or equal to 1", 400

        # Pronunciation is a tuple, so if the input string is empty it's an empty tuple
        pronunciation = tuple(so_far.split(" ")) if so_far else ()

        try:
            next_probabilities = MODEL.next_probabilities(pronunciation)
        except KeyError as e:
            return str(e), 400

        next_probabilities = {
            phoneme: truncate(prob * 100, 3)
            for (phoneme, prob) in next_probabilities.items()
            if prob > min_prob
        }

        sorted_probs = sorted(
            next_probabilities.items(), key=itemgetter(1), reverse=True
        )

        return jsonify(sorted_probs)

    return app
