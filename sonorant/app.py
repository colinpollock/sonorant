"""The Flask app that exposes the Sonorant interactive website."""

from operator import itemgetter

from flask import Flask, jsonify, render_template, request
from flask_caching import Cache
from flask_cors import CORS

from sonorant.languagemodel import LanguageModel
from sonorant.polly import get_audio
from sonorant.utils import truncate


MODEL = LanguageModel.load("models/gru_20_20_1.pt", device_name="cpu")

# Minimum probability for returning a  phoneme. This can't be zero, or else things like <PAD> will
# show up.
DEFAULT_MIN_PROB = 0.001


def create_app():
    app = Flask(__name__)
    CORS(app)
    cache = Cache(app, config={"CACHE_TYPE": "simple", "CACHE_THRESHOLD": int(1e6)})

    @app.route("/")
    def interactive_app():
        server_sync_endpoint = f"server_sync?pronunciation="
        return render_template(
            "interactive_app.html", server_sync_endpoint=server_sync_endpoint
        )

    @app.route("/server_sync")
    @cache.cached(timeout=60 * 60 * 24 * 30, query_string=True)
    def server_sync():
        """Return a probability distribution over the vocabulary for the next phoneme.

        Query parameters:
        - pronunciation_string: a space-separated string of phonemes, each of which must be in the
          model's vocab.
        - min_probability: phonemes at or below this threshold won't be returned. Must be greater
          than zero (to avoid showing *everything*) and less than 1.

        Returns: a JSON object with two keys:
        - next_probabilities: descended sorted list of [phoneme, probability] pairs.
        - audio: a base64 encoded string of the audio of the current pronunciation

        Note that the returned probabilities are ints between 0 and 100 since those are easier to
        display in a chart.
        """
        pronunciation_string = request.args.get("pronunciation")

        try:
            min_prob = float(request.args.get("min_prob", DEFAULT_MIN_PROB))
        except ValueError as e:
            return e.message, 400

        if not 0 < min_prob <= 1:
            return "min_prob must be greater than 0 and less than or equal to 1", 400

        # Pronunciation is a tuple, so if the input string is empty it's an empty tuple
        pronunciation = (
            tuple(pronunciation_string.split(" ")) if pronunciation_string else ()
        )

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

        audio_string = get_audio(pronunciation)

        return jsonify({"next_probabilities": sorted_probs, "audio": audio_string})

    return app
