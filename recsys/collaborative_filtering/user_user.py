__all__ = ["UserUserCollaborativeFiltering"]

from typing import Dict, List, Optional, Tuple
from functools import reduce
from tqdm import tqdm
import numpy as np
import pandas as pd

from recsys.collaborative_filtering.base import SimilarityComputer, SimilarityMethod
from recsys.utils.errors import (
    SimilarityMethodNotAvailable,
    SimilarityMethodRequiresRating,
    ModelNotFittedYet,
    UserNotPresent,
)
from recsys.utils.logging import logger


class UserUserCollaborativeFiltering(object):
    """Class that implements user-user similarity for collaborative-filtering based recommender system.

    Parameters:
        user_id_field_name (str): Column name for user IDs.
        item_id_field_name (str): Column name for item IDs.
        rating_field_name (optional, str): Column name for ratings, if available. Defaults to None.
        similarity_method (optional, str): Method for similarity computation. Defaults to "IoU" - allowed only for binary case.
        minimum_common_items (optional, int): Minimum number of common items for similarity computation. Defaults to 5.
        K_top_similar_users (optional, int): Maximum number of similar users for further predictions. Defaults to 25.

    Raises:
        AttributeError: Raised for predictions, when none of entries are passed to the method.
        UserNotPresent: Raised for cold-start problems, when user wasn't found in the training set.
        ModelNotFittedYet: Raised for predictions, when model hasn't been previously trained.
        SimilarityMethodNotAvailable: Raised during instantiation, when chosen method is not implemented.
        SimilarityMethodRequiresRating: Raised during instantiation, when chosen method requires a rating column.
    """

    def __init__(
        self,
        user_id_field_name: str,
        item_id_field_name: str,
        rating_field_name: Optional[str] = None,
        similarity_method: str = "IoU",
        minimum_common_items: int = 5,
        K_top_similar_users: int = 25,
    ) -> None:
        """Constructor method for UserUserCollaborativeFiltering.

        Args:
            user_id_field_name (str): Column name for user IDs.
            item_id_field_name (str): Column name for item IDs.
            rating_field_name (optional, str): Column name for ratings, if available. Defaults to None.
            similarity_method (optional, str): Method for similarity computation. Defaults to "IoU" - allowed only for binary case.
            minimum_common_items (optional, int): Minimum number of common items for similarity computation. Defaults to 5.
            K_top_similar_users (optional, int): Maximum number of similar users for further predictions. Defaults to 25.
        """
        self._user_id_field_name = user_id_field_name
        self._item_id_field_name = item_id_field_name
        self._rating_field_name = rating_field_name
        self._similarity_method = similarity_method
        self._minimum_common_items = minimum_common_items
        self._K_top_similar_users = K_top_similar_users

        if similarity_method not in SimilarityMethod.__members__.keys():
            raise SimilarityMethodNotAvailable(
                f"Method {similarity_method} not implemented! Options are: {'|'.join(SimilarityMethod.__members__.keys())}"
            )

        if (
            similarity_method
            in (SimilarityMethod.correlation.name, SimilarityMethod.mse.name)
            and not rating_field_name
        ):
            raise SimilarityMethodRequiresRating(
                f"Method {similarity_method} is for continuous ratings! Please inform column accordingly!"
            )

        self._higher_better, self._sim_func = SimilarityMethod.__members__[
            self._similarity_method
        ].value
        self._sim_computer = SimilarityComputer(
            ref_col=self._user_id_field_name,
            agg_col=self._item_id_field_name,
            rating_col=self._rating_field_name,
        )
        # TO BE COMPUTED DURING TRAINING
        self._user_similarity: Dict[int, List[int]] = None
        self._items_by_user: Dict[int, List[int]] = None
        self._user_average_ratings: Dict[int, float] = None
        self._ratings_by_user_item: Dict[Tuple[int, int], float] = None

        logger.info("User-User Collaborative Filtering model instantiated...")

    def fit(self, df: pd.DataFrame):
        """Method for fitting the model, based on user-user similarity.

        Args:
            df (pd.DataFrame): Dataframe containing users, items and corresponding ratings (if available).
        """
        logger.info("Finding most similar users...")
        self._user_similarity = self._sim_computer.find_most_similar(
            df=df,
            minimum_agg_count=self._minimum_common_items,
            maximum_similar=self._K_top_similar_users,
            sim_func=self._sim_func,
            higher_better=self._higher_better,
        )
        self._items_by_user = self._sim_computer.get_list_of_agg_by_ref(df)
        if self._rating_field_name:
            self._user_similarity, self._user_average_ratings = self._user_similarity
            self._ratings_by_user_item = self._sim_computer.get_outcome_by_ref_and_agg(
                df
            )

        return self

    def predict(
        self,
        df: Optional[pd.DataFrame] = None,
        user: Optional[int] = None,
        K_top_items: Optional[int] = None,
    ) -> pd.DataFrame:
        """Method for predicting/inferencing. If `df` is passed, then model generates either rating or propensity for each entry.
        On the other hand, if `user` is passed, then it generates a list with the most likely items - it can be limited by `K_top_items`.

        Args:
            df (optional, pd.DataFrame): Dataframe containing users and items. Defaults to None.
            user (optional, int): User ID for specific recommendations. Defaults to None.
            K_top_items (optional, int): Number of top recommendations to return for specified user. Defaults to None.

        Raises:
            AttributeError: Raised for predictions, when none of entries are passed to the method.
            UserNotPresent: Raised for cold-start problems, when user wasn't found in the training set.
            ModelNotFittedYet: Raised for predictions, when model hasn't been previously trained.
        """
        if df is not None:
            return self._predict_df(df)
        elif user is not None:
            return self._predict_for_user(user=user, K_top_items=K_top_items)
        else:
            raise AttributeError("Please inform either dataframe or user id!")

    def _predict_user_item(self, user: int, item: int) -> float:
        """Method for predicting a single score/rating for a given pair (user, item).

        Args:
            user (int): User ID.
            item (int): Item ID.

        Raises:
            AttributeError: Raised for predictions, when none of entries are passed to the method.
            UserNotPresent: Raised for cold-start problems, when user wasn't found in the training set.
        """
        ratings = []
        if user not in self._user_similarity.keys():
            raise UserNotPresent(
                f"User {user} not found in training data... Cold-Start problem!"
            )
        for similar_user in self._user_similarity[user]:
            if self._rating_field_name:
                if (similar_user, item) in self._ratings_by_user_item.keys():
                    ratings.append(
                        self._ratings_by_user_item[(similar_user, item)]
                        - self._user_average_ratings[similar_user]
                    )
            else:
                ratings.append(1 if item in self._items_by_user[similar_user] else 0)
        if len(ratings) == 0:
            predicted_score = 0.0 if self._rating_field_name else 0.5
        else:
            predicted_score = np.mean(ratings)
        if self._rating_field_name:
            return predicted_score + self._user_average_ratings[user]
        return predicted_score

    def _predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method for predicting scores/ratings for each entry in dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing users and items.

        Raises:
            UserNotPresent: Raised for cold-start problems, when user wasn't found in the training set.
            ModelNotFittedYet: Raised for predictions, when model hasn't been previously trained.
        """
        if self._user_similarity is None:
            raise ModelNotFittedYet("Plese call `.fit()` before predictions!")
        df_ = df.copy()
        score_col = (
            f"predicted_{self._rating_field_name}"
            if self._rating_field_name
            else "predicted_propensity"
        )
        ratings = []
        for _, row in tqdm(df_.iterrows()):
            ratings.append(
                self._predict_user_item(
                    user=row[self._user_id_field_name],
                    item=row[self._item_id_field_name],
                )
            )
        df_[score_col] = ratings
        return df_

    def _predict_for_user(
        self, user: int, K_top_items: Optional[int] = None
    ) -> pd.DataFrame:
        """Method for returning best recommendations for a single user.

        Args:
            user (int): User ID for specific recommendations.
            K_top_items (int): Number of top recommendations to return for specified user.

        Raises:
            UserNotPresent: Raised for cold-start problems, when user wasn't found in the training set.
            ModelNotFittedYet: Raised for predictions, when model hasn't been previously trained.
        """
        if self._user_similarity is None:
            raise ModelNotFittedYet("Plese call `.fit()` before predictions!")
        if user not in self._user_similarity.keys():
            raise UserNotPresent(
                f"User {user} not found in training data... Cold-Start problem!"
            )

        similar_users = self._user_similarity[user]
        items_not_to_consider = self._items_by_user[user]
        item_pool = set(
            reduce(lambda a, b: a + b, [self._items_by_user[u] for u in similar_users])
        )
        predicted_ratings = {}
        for item in item_pool:
            if not item in items_not_to_consider:
                predicted_ratings[item] = self._predict_user_item(user=user, item=item)
        score_col = (
            f"predicted_{self._rating_field_name}"
            if self._rating_field_name
            else "predicted_propensity"
        )
        recommendations = pd.DataFrame(
            {
                self._item_id_field_name: list(predicted_ratings.keys()),
                score_col: list(predicted_ratings.values()),
            }
        )
        recommendations = recommendations.sort_values(by=score_col, ascending=False)
        if K_top_items:
            if K_top_items < len(recommendations):
                logger.warning(
                    "Less potential %ss found than %d...",
                    self._item_id_field_name,
                    K_top_items,
                )
            recommendations = recommendations.head(K_top_items)
        return recommendations
