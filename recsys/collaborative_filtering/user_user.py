from typing import Dict, List, Optional, Tuple
import pandas as pd

from recsys.collaborative_filtering.base import SimilarityComputer, SimilarityMethod
from recsys.utils.errors import (
    SimilarityMethodNotAvailable,
    SimilarityMethodRequiresRating,
    ModelNotFittedYet,
)


class UserUserCollaborativeFiltering(object):
    def __init__(
        self,
        user_id_field_name: str,
        item_id_field_name: str,
        rating_field_name: Optional[str] = None,
        similarity_method: str = "IoU",
        minimum_common_items: int = 5,
        K_top_similar_users: int = 25,
    ) -> None:
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
        ]
        self._sim_computer = SimilarityComputer(
            ref_col=self._user_id_field_name,
            agg_col=self._item_id_field_name,
            rating_col=self._rating_field_name,
        )
        # TO BE COMPUTED DURING TRAINING
        self._user_similarity: Dict[int, List[int]] = None
        self._user_average_ratings: Dict[int, float] = None
        self._ratings_by_user_item: Dict[Tuple[int, int], float] = None

    def fit(self, df: pd.DataFrame):
        self._user_similarity = self._sim_computer.find_most_similar(
            df=df,
            minimum_agg_count=self._minimum_common_items,
            maximum_similar=self._K_top_similar_users,
            sim_func=self._sim_func,
            higher_better=self._higher_better,
        )
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
        if df is not None:
            return self._predict_df(df)
        elif user is not None:
            return self._predict_for_user(user=user, K_top_items=K_top_items)
        else:
            raise AttributeError("Please inform either dataframe or user id!")

    def _predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._user_similarity is None:
            raise ModelNotFittedYet("Plese call `.fit()` before predictions!")
        # TO IMPLEMENT
        return df

    def _predict_for_user(
        self, user: int, K_top_items: Optional[int] = None
    ) -> pd.DataFrame:
        if self._user_similarity is None:
            raise ModelNotFittedYet("Plese call `.fit()` before predictions!")
        # TO IMPLEMENT
        return pd.DataFrame([{user: K_top_items}])
