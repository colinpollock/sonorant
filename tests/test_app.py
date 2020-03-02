from sonorant.app import create_app


# TODO: use pytest fixture instead
class TestApp:
    def setup(self):
        self.app = create_app()
        self.app_context = self.app.app_context()
        self.app_context.push()

        self.client = self.app.test_client()

    def teardown(self):
        self.app_context.pop()

    def test_valid(self):
        response = self.client.get("/server_sync?pronunciation=t r")
        assert response.status_code == 200

        data = response.json
        assert len(data) == 2

        next_probabilities = data["next_probabilities"]
        audio = data["audio"]

        assert isinstance(next_probabilities, list)
        for phoneme, probability in next_probabilities:
            assert isinstance(phoneme, str)
            assert 0 < probability <= 100

        assert isinstance(audio, str)

    def test_valid_with_min_prob(self):
        response = self.client.get("/server_sync?pronunciation=t r&min_prob=.3")
        assert response.status_code == 200

        data = response.json
        assert len(data) == 2

        next_probabilities = data["next_probabilities"]
        audio = data["audio"]

        assert isinstance(next_probabilities, list)
        for phoneme, probability in next_probabilities:
            assert isinstance(phoneme, str)
            assert 0 < probability <= 100

        assert isinstance(audio, str)

    def test_invalid_min_prob_value(self):
        response = self.client.get("/server_sync?pronunciation=t r&min_prob=0")
        assert response.status_code == 400

    def test_out_of_vocab(self):
        """Tests that a `pronunciation` string with unknown tokens will fail."""
        response = self.client.get("/server_sync?pronunciation=this_is_not_a_token")
        assert response.status_code == 400

        error_message = response.data.decode("utf-8")
        assert "Token 'this_is_not_a_token' is not in the vocabulary" in error_message
