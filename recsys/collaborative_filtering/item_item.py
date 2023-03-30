__all__ = ["ItemItemCollaborativeFiltering"]

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


class ItemItemCollaborativeFiltering(object):
    """Class that implements item-item similarity for collaborative-filtering based recommender system.

    Parameters:
        item_id_field_name (str): Column name for item IDs.
        user_id_field_name (str): Column name for user IDs.
        rating_field_name (optional, str): Column name for ratings, if available. Defaults to None.
        similarity_method (optional, str): Method for similarity computation. Defaults to "IoU" - allowed only for binary case.
        minimum_common_users (optional, int): Minimum number of common users for similarity computation. Defaults to 5.
        K_top_similar_items (optional, int): Maximum number of similar items for further predictions. Defaults to 25.

    Raises:
        AttributeError: Raised for predictions, when none of entries are passed to the method.
        UserNotPresent: Raised for cold-start problems, when user wasn't found in the training set.
        ModelNotFittedYet: Raised for predictions, when model hasn't been previously trained.
        SimilarityMethodNotAvailable: Raised during instantiation, when chosen method is not implemented.
        SimilarityMethodRequiresRating: Raised during instantiation, when chosen method requires a rating column.
    """

    def __init__(
        self,
        item_id_field_name: str,
        user_id_field_name: str,
        rating_field_name: Optional[str] = None,
        similarity_method: str = "IoU",
        minimum_common_users: int = 5,
        K_top_similar_items: int = 25,
    ) -> None:
        """Constructor method for ItemItemCollaborativeFiltering.

        Args:
            item_id_field_name (str): Column name for item IDs.
            user_id_field_name (str): Column name for user IDs.
            rating_field_name (optional, str): Column name for ratings, if available. Defaults to None.
            similarity_method (optional, str): Method for similarity computation. Defaults to "IoU" - allowed only for binary case.
            minimum_common_users (optional, int): Minimum number of common users for similarity computation. Defaults to 5.
            K_top_similar_items (optional, int): Maximum number of similar items for further predictions. Defaults to 25.
        """
        self._item_id_field_name = item_id_field_name
        self._user_id_field_name = user_id_field_name
        self._rating_field_name = rating_field_name
        self._similarity_method = similarity_method
        self._minimum_common_users = minimum_common_users
        self._K_top_similar_items = K_top_similar_items

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
            ref_col=self._item_id_field_name,
            agg_col=self._user_id_field_name,
            rating_col=self._rating_field_name,
        )
        # TO BE COMPUTED DURING TRAINING
        self._item_similarity: Dict[int, List[int]] = None
        self._users_by_item: Dict[int, List[int]] = None
        self._item_average_ratings: Dict[int, float] = None
        self._ratings_by_item_user: Dict[Tuple[int, int], float] = None
        self._best_rated_item_by_user: Dict[int, int] = None

        logger.info("Item-Item Collaborative Filtering model instantiated...")

    def fit(self, df: pd.DataFrame):
        """Method for fitting the model, based on item-item similarity.

        Args:
            df (pd.DataFrame): Dataframe containing users, items and corresponding ratings (if available).
        """
        logger.info("Finding most similar items...")
        self._item_similarity = self._sim_computer.find_most_similar(
            df=df,
            minimum_agg_count=self._minimum_common_users,
            maximum_similar=self._K_top_similar_items,
            sim_func=self._sim_func,
            higher_better=self._higher_better,
        )
        self._users_by_item = self._sim_computer.get_list_of_agg_by_ref(df)
        self._best_rated_item_by_user = {}
        if self._rating_field_name:
            self._item_similarity, self._item_average_ratings = self._item_similarity
            self._ratings_by_item_user = self._sim_computer.get_outcome_by_ref_and_agg(
                df
            )
            for user in set(
                reduce(lambda a, b: a + b, list(self._users_by_item.values()))
            ):
                self._best_rated_item_by_user[user] = sorted(
                    [
                        (i, r)
                        for (i, u), r in self._ratings_by_item_user.items()
                        if u == user
                    ],
                    key=lambda x: -x[1],
                )[0][0]
        else:
            for user in set(
                reduce(lambda a, b: a + b, list(self._users_by_item.values()))
            ):
                self._best_rated_item_by_user[user] = [
                    i for (i, u), r in self._ratings_by_item_user.items() if u == user
                ][-1]

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
        if user not in self._best_rated_item_by_user.keys():
            raise UserNotPresent(
                f"User {user} not found in training data... Cold-Start problem!"
            )
        for similar_item in self._item_similarity[item]:
            if self._ratings_by_item_user:
                if (similar_item, user) in self._ratings_by_item_user.keys():
                    ratings.append(
                        self._ratings_by_item_user[(similar_item, user)]
                        - self._item_average_ratings[similar_item]
                    )
            else:
                ratings.append(1 if user in self._users_by_item[similar_item] else 0)
        if len(ratings) == 0:
            predicted_score = 0.0 if self._rating_field_name else 0.5
        else:
            predicted_score = np.mean(ratings)
        if self._rating_field_name:
            return predicted_score + self._item_average_ratings[user]
        return predicted_score

    def _predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Method for predicting scores/ratings for each entry in dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing users and items.

        Raises:
            UserNotPresent: Raised for cold-start problems, when user wasn't found in the training set.
            ModelNotFittedYet: Raised for predictions, when model hasn't been previously trained.
        """
        if self._item_similarity is None:
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
        if self._item_similarity is None:
            raise ModelNotFittedYet("Plese call `.fit()` before predictions!")
        if user not in self._best_rated_item_by_user.keys():
            raise UserNotPresent(
                f"User {user} not found in training data... Cold-Start problem!"
            )

        best_item = self._best_rated_item_by_user[user]
        items_not_to_consider = list(
            map(
                lambda x: x[0][0],
                filter(lambda x: user == x[0][1], self._ratings_by_item_user.items()),
            )
        )
        predicted_ratings = {}
        for item in self._item_similarity[best_item]:
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
