from typing import Dict, Tuple
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import pytest

from recsys.matrix_factorization.embedding import EmbeddingMatrixFactorization
from recsys.matrix_factorization.data import (
    convert_dataframe_into_train_and_validation_generators,
)
from recsys.utils.errors import ColdStartProblem


class TestEmbeddingMatrixFactorization(object):

    _LATENT_DIMENSION = 32
    _DIM_THRESHOLD = 8

    def test___init___WithDenseLayers(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        # OUTPUT
        user_mapping, item_mapping = movielens_mappings
        model = EmbeddingMatrixFactorization(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            use_dense_layers=True,
        )
        model_layers = model.layers

        # EXPECTED
        dim = self._LATENT_DIMENSION // 2
        layers = 0
        while dim >= self._DIM_THRESHOLD:
            layers += 1
            dim //= 2

        expected_number_of_layers = 2 * (layers + 1) + 1

        # ASSERT
        assert len(model_layers) == expected_number_of_layers

    def test___init___WithoutDenseLayers(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        # OUTPUT
        user_mapping, item_mapping = movielens_mappings
        model = EmbeddingMatrixFactorization(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            use_dense_layers=False,
        )
        model_layers = model.layers

        # EXPECTED
        expected_number_of_layers = 3

        # ASSERT
        assert len(model_layers) == expected_number_of_layers

    def test_get_user_embedding_RaisesColdStartProblem(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        user_mapping, item_mapping = movielens_mappings
        model = EmbeddingMatrixFactorization(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            use_dense_layers=True,
        )
        with pytest.raises(ColdStartProblem):
            _ = model.get_user_embedding(user_id=20202021)

    def test_get_item_embedding_RaisesColdStartProblem(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        user_mapping, item_mapping = movielens_mappings
        model = EmbeddingMatrixFactorization(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            use_dense_layers=True,
        )
        with pytest.raises(ColdStartProblem):
            _ = model.get_item_embedding(item_id=20202021)

    def test_predict_rating_for_user_item_RaisesColdStartProblem(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        user_mapping, item_mapping = movielens_mappings
        model = EmbeddingMatrixFactorization(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            use_dense_layers=True,
        )
        with pytest.raises(ColdStartProblem):
            _ = model.predict_rating_for_user_item(user_id=1, item_id=20202021)

    def test_predict_rating_for_user_RaisesColdStartProblem(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        user_mapping, item_mapping = movielens_mappings
        model = EmbeddingMatrixFactorization(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            use_dense_layers=True,
        )
        with pytest.raises(ColdStartProblem):
            _ = model.predict_rating_for_user(user_id=20202021)

    def test_fit_and_predict_rating_for_user_RunCorrectly(
        self,
        movielens_sample: pd.DataFrame,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
    ):
        # OUTPUT
        user_mapping, item_mapping = movielens_mappings
        train_gen, val_gen = convert_dataframe_into_train_and_validation_generators(
            df=movielens_sample,
            user_id_field_name="userId",
            item_id_field_name="movieId",
            ratings_field_name="rating",
            user_to_index_mapping=user_mapping,
            item_to_index_mapping=item_mapping,
            val_size=0.30,
            train_batch_size=32,
            val_batch_size=64,
            seed=99,
        )
        np.random.seed(99)
        tf.random.set_seed(99)
        random.seed(99)
        model = EmbeddingMatrixFactorization(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            use_dense_layers=True,
        )
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
        )
        model.fit(
            train_gen.take(2),
            validation_data=val_gen.take(1),
            epochs=1,
            verbose=0,
        )
        user_predictions = model.predict_rating_for_user(user_id=1).head(10)

        # EXPECTED
        expected_user_predictions = pd.DataFrame(
            {
                "item_id": {
                    0: 6370,
                    1: 1127,
                    2: 2687,
                    3: 909,
                    4: 6537,
                    5: 1882,
                    6: 34,
                    7: 44633,
                    8: 928,
                    9: 260,
                },
                "score": {
                    0: 0.012773723341524601,
                    1: 0.0121458163484931,
                    2: 0.011640172451734543,
                    3: 0.010912295430898666,
                    4: 0.010879358276724815,
                    5: 0.010389523580670357,
                    6: 0.010335313156247139,
                    7: 0.010247363708913326,
                    8: 0.010126914829015732,
                    9: 0.010076092556118965,
                },
            }
        )

        # ASSERTION
        pd.testing.assert_frame_equal(expected_user_predictions, user_predictions)

    def test_fit_and_get_user_embedding_RunCorrectly(
        self,
        movielens_sample: pd.DataFrame,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
    ):
        # OUTPUT
        user_mapping, item_mapping = movielens_mappings
        train_gen, val_gen = convert_dataframe_into_train_and_validation_generators(
            df=movielens_sample,
            user_id_field_name="userId",
            item_id_field_name="movieId",
            ratings_field_name="rating",
            user_to_index_mapping=user_mapping,
            item_to_index_mapping=item_mapping,
            val_size=0.30,
            train_batch_size=32,
            val_batch_size=64,
            seed=99,
        )
        np.random.seed(99)
        tf.random.set_seed(99)
        random.seed(99)
        random
        model = EmbeddingMatrixFactorization(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            use_dense_layers=True,
        )
        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(),
        )
        model.fit(
            train_gen.take(2),
            validation_data=val_gen.take(1),
            epochs=1,
            verbose=0,
        )
        user_embedding = model.get_user_embedding(user_id=1)

        # EXPECTED
        expected_user_embedding = np.array(
            [
                -0.018551576882600784,
                -0.009452462196350098,
                0.01631157472729683,
                -0.047664787620306015,
                -0.03186660632491112,
                -0.036276862025260925,
                0.03950244188308716,
                0.029745418578386307,
                0.033144284039735794,
                0.046700164675712585,
                0.0443011149764061,
                0.013693534769117832,
                -0.04025751352310181,
                -0.01196687575429678,
                0.012486916035413742,
                0.04581305384635925,
                -0.04365605115890503,
                -0.027174662798643112,
                0.04538871720433235,
                -0.04532766714692116,
                -0.030307035893201828,
                -0.011748716235160828,
                -0.005235504824668169,
                -0.02402733825147152,
                -0.012572187930345535,
                -0.011427647434175014,
                0.024041958153247833,
                -0.00404916238039732,
                0.02144928090274334,
                -0.03894893452525139,
                -0.009888377040624619,
                -0.05125913769006729,
            ]
        )

        # ASSERTION
        np.testing.assert_array_almost_equal(expected_user_embedding, user_embedding)
