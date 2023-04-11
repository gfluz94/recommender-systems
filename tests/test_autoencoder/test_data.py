from typing import Dict, Tuple
import numpy as np
import pandas as pd
import pytest

from recsys.autoencoder.data import UserItemMatrix


class TestUserItemMatrix:

    _USER_ID_FIELD = "userId"
    _ITEM_ID_FIELD = "movieId"
    _BATCH_SIZE = 3

    def test___init___RunsSuccessfully(
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

        # EXPECTED
        length = movielens_sample[self._USER_ID_FIELD].nunique() // self._BATCH_SIZE + 1

        # ASSERT
        assert len(sequence) == length

    def test__get_input_for_user_RunsSuccessfully(
        self,
        movielens_sample: pd.DataFrame,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
        output_user_array: np.ndarray,
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
        output = sequence._get_input_for_user(user=1)

        # EXPECTED
        expected = output_user_array.copy()

        # ASSERT
        np.testing.assert_almost_equal(output, expected)

    def test__get_batch_RunsSuccessfully(
        self,
        movielens_sample: pd.DataFrame,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
        output_batch_array: np.ndarray,
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
        output = sequence._get_batch(batch_index=0)

        # EXPECTED
        expected = output_batch_array.copy()

        # ASSERT
        np.testing.assert_almost_equal(output, expected)

    def test___getitem___RunsSuccessfullyWithIndex(
        self,
        movielens_sample: pd.DataFrame,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
        output_batch_array: np.ndarray,
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
        output = sequence[0]

        # EXPECTED
        expected = output_batch_array.copy()

        # ASSERT
        assert len(output) == 2
        np.testing.assert_almost_equal(output[0], expected)
        np.testing.assert_almost_equal(output[1], expected)

    def test___getitem___RunsSuccessfullyWithSlice(
        self,
        movielens_sample: pd.DataFrame,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
        output_batch_array: np.ndarray,
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
        output = sequence[:1]

        # EXPECTED
        expected = output_batch_array.copy()

        # ASSERT
        assert len(output) == 2
        np.testing.assert_almost_equal(output[0], expected)
        np.testing.assert_almost_equal(output[1], expected)

    def test___getitem___RunsSuccessfullyWithList(
        self,
        movielens_sample: pd.DataFrame,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
        output_batch_array: np.ndarray,
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
        output = sequence[[0, 0]]

        # EXPECTED
        expected = np.vstack(2 * [output_batch_array])

        # ASSERT
        assert len(output) == 2
        np.testing.assert_almost_equal(output[0], expected)
        np.testing.assert_almost_equal(output[1], expected)

    def test___getitem___RaisesAttributeError(
        self,
        movielens_sample: pd.DataFrame,
        movielens_mappings: Tuple[Dict[int, int], Dict[int, int]],
    ):
        user_mapping, item_mapping = movielens_mappings
        sequence = UserItemMatrix(
            data=movielens_sample,
            user_id_field_name=self._USER_ID_FIELD,
            item_id_field_name=self._ITEM_ID_FIELD,
            user_mapping=user_mapping,
            item_mapping=item_mapping,
            batch_size=self._BATCH_SIZE,
        )
        with pytest.raises(AttributeError):
            _ = sequence[0, 0]
