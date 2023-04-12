from typing import Dict, Tuple
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import pytest

from recsys.matrix_factorization.als import ALS
from recsys.matrix_factorization.data import (
    convert_dataframe_into_train_and_validation_generators,
)
from recsys.utils.errors import ColdStartProblem


class TestALS(object):

    _LATENT_DIMENSION = 5

    def test___init___UserItemDimensionsCorrect(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        # OUTPUT
        user_mapping, item_mapping = movielens_mappings
        model = ALS(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
        )
        user_dim = model.user_dimension
        item_dim = model.item_dimension

        # EXPECTED
        expected_user_dim = max(user_mapping.values()) + 1
        expected_item_dim = max(item_mapping.values()) + 1

        # ASSERT
        assert user_dim == expected_user_dim
        assert item_dim == expected_item_dim

    def test_get_user_embedding_RaisesColdStartProblem(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        user_mapping, item_mapping = movielens_mappings
        model = model = ALS(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
        )
        with pytest.raises(ColdStartProblem):
            _ = model.get_user_embedding(user_id=20202021)

    def test_get_item_embedding_RaisesColdStartProblem(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        user_mapping, item_mapping = movielens_mappings
        model = ALS(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
        )
        with pytest.raises(ColdStartProblem):
            _ = model.get_item_embedding(item_id=20202021)

    def test_predict_rating_for_user_item_RaisesColdStartProblem(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        user_mapping, item_mapping = movielens_mappings
        model = ALS(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
        )
        with pytest.raises(ColdStartProblem):
            _ = model.predict_rating_for_user_item(user_id=1, item_id=20202021)

    def test_predict_rating_for_user_RaisesColdStartProblem(
        self, movielens_mappings: Tuple[Dict[int, int], Dict[int, int]]
    ):
        user_mapping, item_mapping = movielens_mappings
        model = ALS(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
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
        model = ALS(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
        )
        model.fit(
            train_gen.take(2),
            val_generator=val_gen.take(1),
            epochs=1,
        )
        user_predictions = model.predict_rating_for_user(user_id=1).head(10)

        # EXPECTED
        expected_user_predictions = pd.DataFrame(
            {
                "item_id": {
                    0: 1387,
                    1: 1248,
                    2: 1243,
                    3: 3435,
                    4: 1079,
                    5: 1513,
                    6: 2078,
                    7: 53000,
                    8: 48,
                    9: 8360,
                },
                "score": {
                    0: 2.606652895491024,
                    1: 2.520634226079329,
                    2: 2.4916805048009274,
                    3: 2.4833268294610873,
                    4: 2.395211854403,
                    5: 2.3876052312816793,
                    6: 2.380908865849206,
                    7: 2.349398956896578,
                    8: 2.3431298863848466,
                    9: 2.3411584995647443,
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
        model = ALS(
            latent_dimension=self._LATENT_DIMENSION,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
        )
        model.fit(
            train_gen.take(2),
            val_generator=val_gen.take(1),
            epochs=1,
        )
        user_embedding = model.get_user_embedding(user_id=1)

        # EXPECTED
        expected_user_embedding = np.array(
            [0.25807566, 0.62006184, 0.32897039, 0.87328222, 0.89221223]
        )

        # ASSERTION
        np.testing.assert_array_almost_equal(expected_user_embedding, user_embedding)
