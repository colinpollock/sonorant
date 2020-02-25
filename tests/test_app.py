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
        response = self.client.get("/next_probs?so_far=t r")
        assert response.status_code == 200

        phoneme_distribution = response.json

        assert isinstance(phoneme_distribution, list)
        for phoneme, probability in phoneme_distribution:
            assert isinstance(phoneme, str)
            assert 1 <= probability <= 100

    def test_valid_with_min_prob(self):
        response = self.client.get("/next_probs?so_far=t r&min_prob=.3")
        assert response.status_code == 200

        phoneme_distribution = response.json

        assert isinstance(phoneme_distribution, list)
        for phoneme, probability in phoneme_distribution:
            assert isinstance(phoneme, str)
            assert 30 <= probability <= 100

    def test_out_of_vocab(self):
        """Tests that a so_far string with unknown tokens will fail."""
        response = self.client.get("/next_probs?so_far=this_is_not_a_token")
        assert response.status_code == 400

        error_message = response.data.decode("utf-8")
        assert "Token 'this_is_not_a_token' is not in the vocabulary" in error_message
