import pytest
import pandas as pd

from recsys.collaborative_filtering import ItemItemCollaborativeFiltering
from recsys.utils.errors import (
    SimilarityMethodNotAvailable,
    SimilarityMethodRequiresRating,
    ModelNotFittedYet,
    UserNotPresent,
)


class TestItemItemCollaborativeFiltering(object):

    _USER_COL = "userId"
    _MOVIE_COL = "movieId"
    _RATING_COL = "rating"
    _MINIMUM_COMMON_USERS = 1
    _TOP_SIMILAR_ITEMS = 3

    def test___init__RaisesSimilarityMethodNotAvailable(self):
        with pytest.raises(SimilarityMethodNotAvailable):
            _ = ItemItemCollaborativeFiltering(
                user_id_field_name=self._USER_COL,
                item_id_field_name=self._MOVIE_COL,
                rating_field_name=self._RATING_COL,
                similarity_method="SEP99",
                minimum_common_users=self._MINIMUM_COMMON_USERS,
                K_top_similar_items=self._TOP_SIMILAR_ITEMS,
            )

    def test___init__RaisesSimilarityMethodRequiresRating(self):
        with pytest.raises(SimilarityMethodRequiresRating):
            _ = ItemItemCollaborativeFiltering(
                user_id_field_name=self._USER_COL,
                item_id_field_name=self._MOVIE_COL,
                similarity_method="correlation",
                minimum_common_users=self._MINIMUM_COMMON_USERS,
                K_top_similar_items=self._TOP_SIMILAR_ITEMS,
            )

        with pytest.raises(SimilarityMethodRequiresRating):
            _ = ItemItemCollaborativeFiltering(
                user_id_field_name=self._USER_COL,
                item_id_field_name=self._MOVIE_COL,
                similarity_method="mse",
                minimum_common_users=self._MINIMUM_COMMON_USERS,
                K_top_similar_items=self._TOP_SIMILAR_ITEMS,
            )

    def test_fitRunsCorrectlyWithCorrelation(self):
        # OUTPUT
        model = ItemItemCollaborativeFiltering(
            user_id_field_name=self._USER_COL,
            item_id_field_name=self._MOVIE_COL,
            rating_field_name=self._RATING_COL,
            similarity_method="correlation",
            minimum_common_users=self._MINIMUM_COMMON_USERS,
            K_top_similar_items=self._TOP_SIMILAR_ITEMS,
        )
        input_sample = pd.DataFrame(
            {
                "userId": [1, 1, 2, 2, 3, 3, 4, 4],
                "movieId": [1, 2, 1, 3, 2, 4, 3, 4],
                "rating": [5, 5, 5, 4, 1, 5, 3, 5],
            }
        )
        model.fit(input_sample)

        # EXPECTED
        expected_item_similarity = {1: [2, 3], 2: [1, 4], 3: [1, 4], 4: [2, 3]}

        # ASSERT
        model.item_similarity == expected_item_similarity

    def test_fitRunsCorrectlyWithMSE(self, movielens_sample: pd.DataFrame):
        # OUTPUT
        model = ItemItemCollaborativeFiltering(
            user_id_field_name=self._USER_COL,
            item_id_field_name=self._MOVIE_COL,
            rating_field_name=self._RATING_COL,
            similarity_method="mse",
            minimum_common_users=self._MINIMUM_COMMON_USERS,
            K_top_similar_items=self._TOP_SIMILAR_ITEMS,
        )
        input_sample = pd.DataFrame(
            {
                "userId": [1, 1, 2, 2, 3, 3, 4, 4],
                "movieId": [1, 2, 1, 3, 2, 4, 3, 4],
                "rating": [5, 5, 5, 4, 1, 5, 3, 5],
            }
        )
        model.fit(input_sample)

        # EXPECTED
        expected_item_similarity = {1: [3, 2], 2: [1, 4], 3: [1, 4], 4: [3, 2]}

        # ASSERT
        model.item_similarity == expected_item_similarity

    def test_fitRunsCorrectlyWithIoUAndWithoutRatings(
        self, movielens_sample: pd.DataFrame
    ):
        # OUTPUT
        model = ItemItemCollaborativeFiltering(
            user_id_field_name=self._USER_COL,
            item_id_field_name=self._MOVIE_COL,
            similarity_method="IoU",
            minimum_common_users=self._MINIMUM_COMMON_USERS,
            K_top_similar_items=self._TOP_SIMILAR_ITEMS,
        )
        input_sample = pd.DataFrame(
            {
                "userId": [1, 1, 2, 2, 3, 3, 4, 4],
                "movieId": [1, 2, 1, 3, 2, 4, 3, 4],
                "rating": [5, 5, 5, 4, 1, 5, 3, 5],
            }
        )
        model.fit(input_sample)

        # EXPECTED
        expected_item_similarity = {1: [2, 3], 2: [1, 4], 3: [1, 4], 4: [2, 3]}

        # ASSERT
        model.item_similarity == expected_item_similarity

    def test_predictRaisesModelNotFittedYet(self):
        model = ItemItemCollaborativeFiltering(
            user_id_field_name=self._USER_COL,
            item_id_field_name=self._MOVIE_COL,
            similarity_method="IoU",
            minimum_common_users=self._MINIMUM_COMMON_USERS,
            K_top_similar_items=self._TOP_SIMILAR_ITEMS,
        )
        with pytest.raises(ModelNotFittedYet):
            model.predict(user=1)

    def test_predictRaisesUserNotPresent(self, movielens_sample: pd.DataFrame):
        model = ItemItemCollaborativeFiltering(
            user_id_field_name=self._USER_COL,
            item_id_field_name=self._MOVIE_COL,
            rating_field_name=self._RATING_COL,
            similarity_method="correlation",
            minimum_common_users=self._MINIMUM_COMMON_USERS,
            K_top_similar_items=self._TOP_SIMILAR_ITEMS,
        )
        model.fit(movielens_sample)
        with pytest.raises(UserNotPresent):
            model.predict(user=20202021)

    def test_predictRunsCorrectlyWithDataFrame(self):
        # OUTPUT
        model = ItemItemCollaborativeFiltering(
            user_id_field_name=self._USER_COL,
            item_id_field_name=self._MOVIE_COL,
            rating_field_name=self._RATING_COL,
            similarity_method="correlation",
            minimum_common_users=self._MINIMUM_COMMON_USERS,
            K_top_similar_items=self._TOP_SIMILAR_ITEMS,
        )
        input_sample = pd.DataFrame(
            {
                "userId": [1, 1, 2, 2, 3, 3, 4, 4],
                "movieId": [1, 2, 1, 3, 2, 4, 3, 4],
                "rating": [5, 5, 5, 4, 1, 5, 3, 5],
            }
        )
        model.fit(input_sample)
        output = model.predict(input_sample)

        # EXPECTED
        expected = pd.DataFrame(
            {
                "userId": {0: 1, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3, 6: 4, 7: 4},
                "movieId": {0: 1, 1: 2, 2: 1, 3: 3, 4: 2, 5: 4, 6: 3, 7: 4},
                "rating": {0: 5, 1: 5, 2: 5, 3: 4, 4: 1, 5: 5, 6: 3, 7: 5},
                "predicted_rating": {
                    0: 7.0,
                    1: 5.0,
                    2: 3.5,
                    3: 3.0,
                    4: 3.5,
                    5: 1.5,
                    6: 5.0,
                    7: 4.5,
                },
            }
        )

        # ASSERT
        pd.testing.assert_frame_equal(output, expected)

    def test_predictRunsCorrectlyWithUser(
        self,
    ):
        # OUTPUT
        model = ItemItemCollaborativeFiltering(
            user_id_field_name=self._USER_COL,
            item_id_field_name=self._MOVIE_COL,
            rating_field_name=self._RATING_COL,
            similarity_method="correlation",
            minimum_common_users=self._MINIMUM_COMMON_USERS,
            K_top_similar_items=self._TOP_SIMILAR_ITEMS,
        )
        input_sample = pd.DataFrame(
            {
                "userId": [1, 1, 2, 2, 3, 3, 4, 4],
                "movieId": [1, 2, 1, 3, 2, 4, 3, 4],
                "rating": [5, 5, 5, 4, 1, 5, 3, 5],
            }
        )
        model.fit(input_sample)
        output = model.predict(user=1, K_top_items=2)

        # EXPECTED
        expected = pd.DataFrame({"movieId": {0: 3}, "predicted_rating": {0: 5.0}})

        # ASSERT
        pd.testing.assert_frame_equal(output, expected)
