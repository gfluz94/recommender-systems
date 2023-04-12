from typing import Dict, Tuple
import random
import pytest
import pandas as pd
import numpy as np
import tensorflow as tf

from recsys.autoencoder.base import UserVectorAutoEncoder
from recsys.autoencoder.data import UserItemMatrix
from recsys.utils.errors import ModelNotFittedYet


class TestUserVectorAutoEncoder:

    _LATENT_SPACE_DIM = 5
    _HIDDEN_LAYERS = [256, 64, 16]
    _USER_ID_FIELD = "userId"
    _ITEM_ID_FIELD = "movieId"
    _BATCH_SIZE = 5

    def test___init___RunsCorrectly(
        self,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
    ):
        # OUTPUT
        _, item_mapping = movielens_mappings
        autoencoder = UserVectorAutoEncoder(
            latent_space_dimension=self._LATENT_SPACE_DIM,
            item_mapping=item_mapping,
            encoder_hidden_layers=self._HIDDEN_LAYERS,
        )

        # EXPECTED
        input_dim = len(item_mapping.keys())

        # ASSERT
        assert input_dim == autoencoder._input_dim
        assert not autoencoder.fitted

    def test_predict_raisesModelNotFittedYet(
        self,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
    ):
        _, item_mapping = movielens_mappings
        autoencoder = UserVectorAutoEncoder(
            latent_space_dimension=self._LATENT_SPACE_DIM,
            item_mapping=item_mapping,
            encoder_hidden_layers=self._HIDDEN_LAYERS,
        )
        with pytest.raises(ModelNotFittedYet):
            _ = autoencoder.predict(np.array([[1.0, 0.0]]))

    def test_recommend_items_raisesModelNotFittedYet(
        self,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
    ):
        _, item_mapping = movielens_mappings
        autoencoder = UserVectorAutoEncoder(
            latent_space_dimension=self._LATENT_SPACE_DIM,
            item_mapping=item_mapping,
            encoder_hidden_layers=self._HIDDEN_LAYERS,
        )
        with pytest.raises(ModelNotFittedYet):
            _ = autoencoder.recommend_items(np.array([[1.0, 0.0]]))

    def test_get_user_embedding_raisesModelNotFittedYet(
        self,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
    ):
        _, item_mapping = movielens_mappings
        autoencoder = UserVectorAutoEncoder(
            latent_space_dimension=self._LATENT_SPACE_DIM,
            item_mapping=item_mapping,
            encoder_hidden_layers=self._HIDDEN_LAYERS,
        )
        with pytest.raises(ModelNotFittedYet):
            _ = autoencoder.get_user_embedding(np.array([[1.0, 0.0]]))

    def test__get_encoder_runsCorrectly(
        self,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
    ):
        # OUTPUT
        _, item_mapping = movielens_mappings
        autoencoder = UserVectorAutoEncoder(
            latent_space_dimension=self._LATENT_SPACE_DIM,
            item_mapping=item_mapping,
            encoder_hidden_layers=self._HIDDEN_LAYERS,
        )
        encoder = autoencoder._get_encoder()

        # ASSERT
        assert isinstance(encoder, tf.keras.models.Model)
        assert len(encoder.layers) == len(self._HIDDEN_LAYERS) + 2

    def test__get_decoder_runsCorrectly(
        self,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
    ):
        # OUTPUT
        _, item_mapping = movielens_mappings
        autoencoder = UserVectorAutoEncoder(
            latent_space_dimension=self._LATENT_SPACE_DIM,
            item_mapping=item_mapping,
            encoder_hidden_layers=self._HIDDEN_LAYERS,
        )
        decoder = autoencoder._get_decoder()

        # ASSERT
        assert isinstance(decoder, tf.keras.models.Model)
        assert len(decoder.layers) == len(self._HIDDEN_LAYERS) + 2

    def test__get_autoencoder_runsCorrectly(
        self,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
    ):
        # OUTPUT
        _, item_mapping = movielens_mappings
        autoencoder = UserVectorAutoEncoder(
            latent_space_dimension=self._LATENT_SPACE_DIM,
            item_mapping=item_mapping,
            encoder_hidden_layers=self._HIDDEN_LAYERS,
        )
        model = autoencoder._get_autoencoder()

        # ASSERT
        assert isinstance(model, tf.keras.models.Model)
        assert len(model.layers) == 2

    def test_fit_runsCorrectly(
        self,
        movielens_sample: pd.DataFrame,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
    ):
        # OUTPUT
        user_mapping, item_mapping = movielens_mappings
        sequence = UserItemMatrix(
            data=movielens_sample,
            user_id_field_name=self._USER_ID_FIELD,
            item_id_field_name=self._ITEM_ID_FIELD,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            batch_size=self._BATCH_SIZE,
        )
        autoencoder = UserVectorAutoEncoder(
            latent_space_dimension=self._LATENT_SPACE_DIM,
            item_mapping=item_mapping,
            encoder_hidden_layers=self._HIDDEN_LAYERS,
        )
        autoencoder.fit(sequence, epochs=1)

        # EXPECTED
        is_fitted = True

        # ASSERT
        assert autoencoder.fitted == is_fitted

    def test_fit_and_predict_runsCorrectly(
        self,
        movielens_sample: pd.DataFrame,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
        output_reconstruction: np.ndarray,
    ):
        # OUTPUT
        user_mapping, item_mapping = movielens_mappings
        np.random.seed(99)
        tf.random.set_seed(99)
        random.seed(99)
        sequence = UserItemMatrix(
            data=movielens_sample,
            user_id_field_name=self._USER_ID_FIELD,
            item_id_field_name=self._ITEM_ID_FIELD,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            batch_size=self._BATCH_SIZE,
        )
        autoencoder = UserVectorAutoEncoder(
            latent_space_dimension=self._LATENT_SPACE_DIM,
            item_mapping=item_mapping,
            encoder_hidden_layers=self._HIDDEN_LAYERS,
        )
        autoencoder.fit(sequence, epochs=1)
        reconstruction = autoencoder.predict(sequence[0][0][0, :])

        # EXPECTED
        expected_reconstruction = output_reconstruction

        # ASSERT
        np.testing.assert_almost_equal(reconstruction, expected_reconstruction)

    def test_fit_and_get_user_embedding_runsCorrectly(
        self,
        movielens_sample: pd.DataFrame,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
    ):
        # OUTPUT
        user_mapping, item_mapping = movielens_mappings
        np.random.seed(99)
        tf.random.set_seed(99)
        random.seed(99)
        sequence = UserItemMatrix(
            data=movielens_sample,
            user_id_field_name=self._USER_ID_FIELD,
            item_id_field_name=self._ITEM_ID_FIELD,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            batch_size=self._BATCH_SIZE,
        )
        autoencoder = UserVectorAutoEncoder(
            latent_space_dimension=self._LATENT_SPACE_DIM,
            item_mapping=item_mapping,
            encoder_hidden_layers=self._HIDDEN_LAYERS,
        )
        autoencoder.fit(sequence, epochs=1)
        embedding = autoencoder.get_user_embedding(sequence[0][0][0, :])
        print(embedding.tolist())

        # EXPECTED
        expected_embedding = np.array([0.0, 0.0, 0.0, 0.0, 0.38587772846221924])

        # ASSERT
        np.testing.assert_almost_equal(embedding, expected_embedding)

    def test_fit_and_recommend_items_runsCorrectly(
        self,
        movielens_sample: pd.DataFrame,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
    ):
        # OUTPUT
        user_mapping, item_mapping = movielens_mappings
        np.random.seed(99)
        tf.random.set_seed(99)
        random.seed(99)
        sequence = UserItemMatrix(
            data=movielens_sample,
            user_id_field_name=self._USER_ID_FIELD,
            item_id_field_name=self._ITEM_ID_FIELD,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            batch_size=self._BATCH_SIZE,
        )
        autoencoder = UserVectorAutoEncoder(
            latent_space_dimension=self._LATENT_SPACE_DIM,
            item_mapping=item_mapping,
            encoder_hidden_layers=self._HIDDEN_LAYERS,
        )
        autoencoder.fit(sequence, epochs=1)
        recommendations = autoencoder.recommend_items(
            sequence[0][0][0, :], top_K_items=3
        )

        # EXPECTED
        expected_recommendations = [
            {
                1242: 0.5097495317459106,
                2746: 0.5092957615852356,
                1201: 0.5088930726051331,
            }
        ]

        # ASSERT
        assert recommendations == expected_recommendations
