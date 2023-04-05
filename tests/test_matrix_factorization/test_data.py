from typing import Dict, Tuple
import pandas as pd
import numpy as np
import pytest

from recsys.matrix_factorization.data import (
    create_data_generator,
    convert_dataframe_into_train_and_validation_generators,
)
from recsys.utils.errors import NotA2DArray, FeaturesNotAllowedForMatrixFactorization


def test_convert_dataframe_into_train_and_validation_generators(
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
        train_batch_size=5,
        val_batch_size=10,
        seed=99,
    )
    for (user_train, movie_train), y_train in train_gen.take(1):
        break
    for (user_val, movie_val), y_val in val_gen.take(1):
        break
    print(user_val, movie_val, y_val)

    # EXPECTED
    expected_user_train = np.array([1.0, 0.0, 0.0, 5.0, 5.0])
    expected_movie_train = np.array([284, 82, 219, 436, 505.0])
    expected_y_train = np.array([3.5, 4.0, 5.0, 0.5, 5.0])
    expected_user_val = np.array([0.0, 5.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    expected_movie_val = np.array(
        [119.0, 318.0, 430.0, 120.0, 181.0, 98.0, 114.0, 187.0, 18.0, 31.0]
    )
    expected_y_val = np.array([1.0, 0.5, 3.0, 3.5, 5.0, 3.0, 2.0, 5.0, 5.0, 5.0])

    # ASSERT
    np.testing.assert_array_almost_equal(user_train.numpy(), expected_user_train)
    np.testing.assert_array_almost_equal(movie_train.numpy(), expected_movie_train)
    np.testing.assert_array_almost_equal(y_train.numpy(), expected_y_train)
    np.testing.assert_array_almost_equal(user_val.numpy(), expected_user_val)
    np.testing.assert_array_almost_equal(movie_val.numpy(), expected_movie_val)
    np.testing.assert_array_almost_equal(y_val.numpy(), expected_y_val)


class Testcreate_data_generator:
    def test_raisesNotA2DArray(self):
        np.random.seed(99)
        X = np.random.normal(size=(100, 3, 1))
        y = np.random.normal(size=(100,))
        with pytest.raises(NotA2DArray):
            _ = create_data_generator(
                X=X,
                y=y,
                batch_size=32,
            )

    def test_raisesFeaturesNotAllowedForMatrixFactorization(self):
        np.random.seed(99)
        X = np.random.normal(size=(100, 3))
        y = np.random.normal(size=(100,))
        with pytest.raises(FeaturesNotAllowedForMatrixFactorization):
            _ = create_data_generator(
                X=X,
                y=y,
                batch_size=32,
            )

    def test_outputCorrect(self):
        np.random.seed(99)
        X = np.random.normal(size=(100, 2))
        y = np.random.normal(size=(100,))

        # OUTPUT
        gen = create_data_generator(
            X=X,
            y=y,
            batch_size=5,
        )
        for (x_batch_1, x_batch_2), y_batch in gen.take(1):
            break

        # EXPECTED
        expected_x_1 = np.array(
            [1.3966199, -1.67850204, 0.50265173, -0.29510767, 0.2668912]
        )
        expected_x_2 = np.array(
            [0.03221944, 0.18310987, 0.20955906, -0.361676, -0.15874187]
        )
        expected_y = np.array(
            [0.51966199, 0.47986554, 1.16540462, -0.35728133, 0.8017332]
        )

        # ASSERT
        np.testing.assert_array_almost_equal(x_batch_1.numpy(), expected_x_1)
        np.testing.assert_array_almost_equal(x_batch_2.numpy(), expected_x_2)
        np.testing.assert_array_almost_equal(y_batch.numpy(), expected_y)
