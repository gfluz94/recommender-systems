from typing import Dict, Tuple
import random
import pytest
import pandas as pd
import numpy as np
import tensorflow as tf

from recsys.autoencoder.variational import UserVectorVAE
from recsys.autoencoder.data import UserItemMatrix
from recsys.utils.errors import ModelNotFittedYet


class TestUserVectorVAE:

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
        autoencoder = UserVectorVAE(
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
        autoencoder = UserVectorVAE(
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
        autoencoder = UserVectorVAE(
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
        autoencoder = UserVectorVAE(
            latent_space_dimension=self._LATENT_SPACE_DIM,
            item_mapping=item_mapping,
            encoder_hidden_layers=self._HIDDEN_LAYERS,
        )
        with pytest.raises(ModelNotFittedYet):
            _ = autoencoder.get_user_embedding(np.array([[1.0, 0.0]]))

    def test__get_autoencoder_runsCorrectly(
        self,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
    ):
        # OUTPUT
        _, item_mapping = movielens_mappings
        autoencoder = UserVectorVAE(
            latent_space_dimension=self._LATENT_SPACE_DIM,
            item_mapping=item_mapping,
            encoder_hidden_layers=self._HIDDEN_LAYERS,
        )
        model = autoencoder._get_autoencoder()
        print(model.layers)

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
        autoencoder = UserVectorVAE(
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
        output_reconstruction_vae: np.ndarray,
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
        autoencoder = UserVectorVAE(
            latent_space_dimension=self._LATENT_SPACE_DIM,
            item_mapping=item_mapping,
            encoder_hidden_layers=self._HIDDEN_LAYERS,
        )
        autoencoder.fit(sequence, epochs=1)
        reconstruction = autoencoder.predict(sequence[0][0][0, :])

        # EXPECTED
        expected_reconstruction = output_reconstruction_vae

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
        autoencoder = UserVectorVAE(
            latent_space_dimension=self._LATENT_SPACE_DIM,
            item_mapping=item_mapping,
            encoder_hidden_layers=self._HIDDEN_LAYERS,
        )
        autoencoder.fit(sequence, epochs=1)
        embedding = autoencoder.get_user_embedding(sequence[0][0][0, :])

        # EXPECTED
        expected_embedding = np.array(
            [
                0.5036712288856506,
                -0.09158383309841156,
                -0.6548135280609131,
                -0.4904910922050476,
                0.25821346044540405,
            ]
        )

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
        autoencoder = UserVectorVAE(
            latent_space_dimension=self._LATENT_SPACE_DIM,
            item_mapping=item_mapping,
            encoder_hidden_layers=self._HIDDEN_LAYERS,
        )
        autoencoder.fit(sequence, epochs=1)
        recommendations = autoencoder.recommend_items(
            sequence[0][0][0, :], top_K_items=3
        )
        print(recommendations)

        # EXPECTED
        expected_recommendations = [
            {
                4066: 0.525389552116394,
                1617: 0.5230278372764587,
                6954: 0.5229130387306213,
            }
        ]

        # ASSERT
        assert recommendations == expected_recommendations
