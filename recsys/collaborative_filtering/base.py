__all__ = ["SimilarityMethod", "SimilarityComputer"]

from joblib import Parallel, delayed
from multiprocessing import cpu_count
from functools import reduce
from typing import Callable, Dict, List, Optional, Tuple, Union
from tqdm import tqdm
from enum import Enum
import numpy as np
import pandas as pd

from recsys.utils.logging import logger


def _pearson_correlation(x: List[float], y: List[float]):
    return np.corrcoef(x=x, y=y)[0, 1]


def _mse(x: List[float], y: List[float]):
    return np.mean((np.array(x) - np.array(y)) ** 2)


def _IoU(x: List[float], y: List[float]):
    intersection = x[y == 1] == 1
    return np.mean(intersection)


class SimilarityMethod(Enum):
    correlation = (True, _pearson_correlation)
    mse = (False, _mse)
    IoU = (True, _IoU)


class SimilarityComputer(object):
    """Class that implements methodology for computing similarity among items and among users.
    Basically, it is a class that allows computations for both, depending on the type of collaborative filtering further used.

    Parameters:
        ref_col (str): Column on top of which similarity is computed.
        agg_col (str): Column that contains the evaluation from the `ref_col`.
        rating_col (optional, str): Column containing the ratings, if available. Defaults to None.
    """

    def __init__(
        self,
        ref_col: str,
        agg_col: str,
        rating_col: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        """Constructor method for SimilarityComputer.

        Args:
            ref_col (str): Column on top of which similarity is computed.
            agg_col (str): Column that contains the evaluation from the `ref_col`.
            rating_col (optional, str): Column containing the ratings, if available. Defaults to None.
            n_jobs (int, optional): Enabling parallelization for huge datasets.
        """
        self._ref_col = ref_col
        self._agg_col = agg_col
        self._rating_col = rating_col
        if n_jobs is None or n_jobs == -1:
            self._n_jobs = max(1, cpu_count() - 1)
        else:
            if n_jobs == 0:
                raise AttributeError(
                    "`n_jobs` must be either a positive integer or -1 indicating all cores."
                )
            self._n_jobs = min(n_jobs, max(1, cpu_count() - 1))
        self._binary = False
        if not self._rating_col:
            self._binary = True
        self._outcome_by_ref_and_agg = None
        self._list_of_agg_by_ref = None

    def get_list_of_agg_by_ref(self, df: pd.DataFrame) -> Dict[int, List[int]]:
        """Method that returns the list of items/users for the reference column.

        Args:
            df (pd.DataFrame): Dataframe containing user/item/rating information.

        Returns:
            Dict[int, List[int]]: Dictionary {reference: list of users/items}
        """
        if self._list_of_agg_by_ref is None:
            logger.info(
                "Generating list of `%s` for `%s`...", self._agg_col, self._ref_col
            )
            self._list_of_agg_by_ref = (
                df.groupby(self._ref_col)
                .agg({self._agg_col: list})[self._agg_col]
                .to_dict()
            )
        return self._list_of_agg_by_ref

    def get_outcome_by_ref_and_agg(self, df: pd.DataFrame) -> Dict[Tuple[int], float]:
        """Method that returns the rating, if present, by each pair (user, item).

        Args:
            df (pd.DataFrame): Dataframe containing user/item/rating information.

        Returns:
            Dict[int, List[int]]: Dictionary {(reference, user/item): rating}
        """
        if self._outcome_by_ref_and_agg is None:
            logger.info(
                "Generating list of outcomes for each (%s, %s) pair...",
                self._agg_col,
                self._ref_col,
            )
            df_ = df.copy()
            output = self._rating_col
            if self._binary:
                output = "binary_outcome"
                df_[output] = 1.0

            df_ = df_[[self._ref_col, self._agg_col, output]].values
            batch_size = len(df_) // self._n_jobs
            batches = [
                df_[i * batch_size : (i + 1) * batch_size, :]
                if i < self._n_jobs - 1
                else df_[i * batch_size :, :]
                for i in range(self._n_jobs)
            ]

            def get_outcome_dictionary(
                outcome_array: np.ndarray,
            ) -> Dict[Tuple[int, int], float]:
                user_movie_ratings = {}
                for ref, agg, out in outcome_array:
                    user_movie_ratings[(ref, agg)] = out
                return user_movie_ratings

            results = Parallel(n_jobs=self._n_jobs, prefer="threads")(
                delayed(get_outcome_dictionary)(batch) for batch in batches
            )

            self._outcome_by_ref_and_agg = reduce(lambda a, b: a | b, results)
        return self._outcome_by_ref_and_agg

    def find_most_similar(
        self,
        df: pd.DataFrame,
        minimum_agg_count: int,
        maximum_similar: int,
        sim_func: Callable[[List[float], List[float]], float],
        higher_better: bool,
    ) -> Union[Dict[int, List[int]], Tuple[Dict[int, List[int]], Dict[int, float]]]:
        """Method that generates the top similar users/items for each entry of reference column.

        Args:
            df (pd.DataFrame):  Dataframe containing user/item/rating information.
            minimum_agg_count (int): Minimum number of common users/items evaluated for it to be considered.
            maximum_similar (int): Maximum number of most similar users/items to each entry of the reference column.
            sim_func (Callable[[List[float], List[float]], float]): Function that computes similarity or dissimilarity between entries.
            higher_better (bool): Whether or not a higher `sim_func` output means more similar.

        Returns:
            Union[Dict[int, List[int]], Tuple[Dict[int, List[int]], Dict[int, float]]]: Dictionary {ref entry: list of similar entries}.
            If rating is present, a dictionary {ref: average rating} is also returned along.
        """
        ref_ref_similarities = {}
        top_similar = {}
        avg_rating_by_ref = {}
        multiplier = 1
        if higher_better:
            multiplier = -1

        refs = list(self.get_list_of_agg_by_ref(df).keys())
        logger.info(
            "Searching top %d similar %ss for each %s",
            maximum_similar,
            self._ref_col,
            self._ref_col,
        )
        for idx, i in tqdm(enumerate(refs)):
            aggs_i = self.get_list_of_agg_by_ref(df)[i]
            set_aggs_i = set(aggs_i)
            ratings_i = {
                agg: self.get_outcome_by_ref_and_agg(df)[(i, agg)] for agg in set_aggs_i
            }
            mean_rating_i = np.mean(list(ratings_i.values()))
            centered_ratings_i = {
                agg: rating - mean_rating_i for agg, rating in ratings_i.items()
            }
            avg_rating_by_ref[i] = avg_rating_by_ref.get(i, mean_rating_i)

            similar_refs = {}
            for j in refs[:idx] + refs[idx + 1 :]:
                pair_i_j = (i, j) if i < j else (j, i)
                aggs_j = self.get_list_of_agg_by_ref(df)[j]
                set_aggs_j = set(aggs_j)
                common_aggs = set_aggs_i & set_aggs_j
                if len(common_aggs) >= minimum_agg_count:
                    if not pair_i_j in ref_ref_similarities:
                        ratings_j = {
                            agg: self.get_outcome_by_ref_and_agg(df)[(j, agg)]
                            for agg in set_aggs_j
                        }
                        mean_rating_j = np.mean(list(ratings_j.values()))
                        avg_rating_by_ref[j] = avg_rating_by_ref.get(j, mean_rating_j)
                        centered_ratings_j = {
                            agg: rating - mean_rating_j
                            for agg, rating in ratings_j.items()
                        }
                        if self._binary:
                            all_aggs = set_aggs_i | set_aggs_j
                            x = [1.0 if agg in common_aggs else 0.0 for agg in all_aggs]
                            y = [1.0 if agg in common_aggs else 0.0 for agg in all_aggs]
                        else:
                            x = (
                                [
                                    rating
                                    for agg, rating in centered_ratings_i.items()
                                    if agg in common_aggs
                                ],
                            )
                            y = (
                                [
                                    rating
                                    for agg, rating in centered_ratings_j.items()
                                    if agg in common_aggs
                                ],
                            )
                        ref_ref_similarities[pair_i_j] = sim_func(x=x, y=y)
                    similar_refs[j] = ref_ref_similarities[pair_i_j]
            top_similar[i] = [
                u
                for u, _ in sorted(
                    similar_refs.items(), key=lambda x: multiplier * x[1]
                )[:maximum_similar]
            ]

        if self._binary:
            return top_similar
        return top_similar, avg_rating_by_ref
