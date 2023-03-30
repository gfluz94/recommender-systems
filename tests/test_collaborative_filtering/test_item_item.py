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
    _MINIMUM_COMMON_ITEMS = 5
    _TOP_SIMILAR_USERS = 5

    def test___init__RaisesSimilarityMethodNotAvailable(self):
        with pytest.raises(SimilarityMethodNotAvailable):
            _ = ItemItemCollaborativeFiltering(
                user_id_field_name=self._USER_COL,
                item_id_field_name=self._MOVIE_COL,
                rating_field_name=self._RATING_COL,
                similarity_method="SEP99",
                minimum_common_users=self._MINIMUM_COMMON_ITEMS,
                K_top_similar_items=self._TOP_SIMILAR_USERS,
            )

    def test___init__RaisesSimilarityMethodRequiresRating(self):
        with pytest.raises(SimilarityMethodRequiresRating):
            _ = ItemItemCollaborativeFiltering(
                user_id_field_name=self._USER_COL,
                item_id_field_name=self._MOVIE_COL,
                similarity_method="correlation",
                minimum_common_users=self._MINIMUM_COMMON_ITEMS,
                K_top_similar_items=self._TOP_SIMILAR_USERS,
            )

        with pytest.raises(SimilarityMethodRequiresRating):
            _ = ItemItemCollaborativeFiltering(
                user_id_field_name=self._USER_COL,
                item_id_field_name=self._MOVIE_COL,
                similarity_method="mse",
                minimum_common_users=self._MINIMUM_COMMON_ITEMS,
                K_top_similar_items=self._TOP_SIMILAR_USERS,
            )

    def test_fitRunsCorrectlyWithCorrelation(self, movielens_sample: pd.DataFrame):
        # OUTPUT
        model = ItemItemCollaborativeFiltering(
            user_id_field_name=self._USER_COL,
            item_id_field_name=self._MOVIE_COL,
            rating_field_name=self._RATING_COL,
            similarity_method="correlation",
            minimum_common_users=self._MINIMUM_COMMON_ITEMS,
            K_top_similar_items=self._TOP_SIMILAR_USERS,
        )
        model.fit(movielens_sample)

        # EXPECTED
        expected_user_similarity = {
            1: [3196, 44950, 42181, 1284, 670],
            670: [1, 42181, 44950],
            1284: [1, 42181, 3196],
            3196: [1, 42181, 44950, 1284],
            42181: [1, 1284, 670, 44950, 3196],
            44950: [1, 42181, 3196, 670],
        }

        # ASSERT
        model._user_similarity == expected_user_similarity

    def test_fitRunsCorrectlyWithMSE(self, movielens_sample: pd.DataFrame):
        # OUTPUT
        model = ItemItemCollaborativeFiltering(
            user_id_field_name=self._USER_COL,
            item_id_field_name=self._MOVIE_COL,
            rating_field_name=self._RATING_COL,
            similarity_method="mse",
            minimum_common_users=self._MINIMUM_COMMON_ITEMS,
            K_top_similar_items=self._TOP_SIMILAR_USERS,
        )
        model.fit(movielens_sample)

        # EXPECTED
        expected_user_similarity = {
            1: [42181, 670, 1284, 3196, 44950],
            670: [1, 42181, 44950],
            1284: [42181, 1, 3196],
            3196: [1, 42181, 1284, 44950],
            42181: [1, 1284, 670, 44950, 3196],
            44950: [1, 42181, 670, 3196],
        }

        # ASSERT
        model._user_similarity == expected_user_similarity

    def test_fitRunsCorrectlyWithIoUAndWithoutRatings(
        self, movielens_sample: pd.DataFrame
    ):
        # OUTPUT
        model = ItemItemCollaborativeFiltering(
            user_id_field_name=self._USER_COL,
            item_id_field_name=self._MOVIE_COL,
            similarity_method="IoU",
            minimum_common_users=self._MINIMUM_COMMON_ITEMS,
            K_top_similar_items=self._TOP_SIMILAR_USERS,
        )
        model.fit(movielens_sample)

        # EXPECTED
        expected_user_similarity = {
            1: [670, 1284, 3196, 42181, 44950],
            670: [44950, 1, 42181],
            1284: [1, 3196, 42181],
            3196: [1, 1284, 42181, 44950],
            42181: [1, 670, 1284, 3196, 44950],
            44950: [670, 1, 3196, 42181],
        }

        # ASSERT
        model._user_similarity == expected_user_similarity

    def test_predictRaisesModelNotFittedYet(self):
        model = ItemItemCollaborativeFiltering(
            user_id_field_name=self._USER_COL,
            item_id_field_name=self._MOVIE_COL,
            similarity_method="IoU",
            minimum_common_users=self._MINIMUM_COMMON_ITEMS,
            K_top_similar_items=self._TOP_SIMILAR_USERS,
        )
        with pytest.raises(ModelNotFittedYet):
            model.predict(user=1)

    def test_predictRaisesUserNotPresent(self, movielens_sample: pd.DataFrame):
        model = ItemItemCollaborativeFiltering(
            user_id_field_name=self._USER_COL,
            item_id_field_name=self._MOVIE_COL,
            rating_field_name=self._RATING_COL,
            similarity_method="correlation",
            minimum_common_users=self._MINIMUM_COMMON_ITEMS,
            K_top_similar_items=self._TOP_SIMILAR_USERS,
        )
        model.fit(movielens_sample)
        with pytest.raises(UserNotPresent):
            model.predict(user=20202021)

    def test_predictRunsCorrectlyWithDataFrame(self, movielens_sample: pd.DataFrame):
        # OUTPUT
        model = ItemItemCollaborativeFiltering(
            user_id_field_name=self._USER_COL,
            item_id_field_name=self._MOVIE_COL,
            rating_field_name=self._RATING_COL,
            similarity_method="correlation",
            minimum_common_users=self._MINIMUM_COMMON_ITEMS,
            K_top_similar_items=self._TOP_SIMILAR_USERS,
        )
        model.fit(movielens_sample)
        output = model.predict(
            df=pd.DataFrame(
                [{"userId": 1, "movieId": 4022}, {"userId": 1, "movieId": 37729}]
            )
        )

        # EXPECTED
        expected = pd.DataFrame(
            {
                "userId": {0: 1, 1: 1},
                "movieId": {0: 4022, 1: 37729},
                "predicted_rating": {0: 5.166849816849817, 1: 0.5979736575481258},
            }
        )

        # ASSERT
        pd.testing.assert_frame_equal(output, expected)

    def test_predictRunsCorrectlyWithUser(
        self,
        movielens_sample: pd.DataFrame,
        output_predictions_for_user_user_filtering: pd.DataFrame,
    ):
        # OUTPUT
        model = ItemItemCollaborativeFiltering(
            user_id_field_name=self._USER_COL,
            item_id_field_name=self._MOVIE_COL,
            rating_field_name=self._RATING_COL,
            similarity_method="correlation",
            minimum_common_users=self._MINIMUM_COMMON_ITEMS,
            K_top_similar_items=self._TOP_SIMILAR_USERS,
        )
        model.fit(movielens_sample)
        output = model.predict(user=1, K_top_items=5)

        # EXPECTED
        expected = output_predictions_for_user_user_filtering.copy()

        # ASSERT
        pd.testing.assert_frame_equal(output, expected)
