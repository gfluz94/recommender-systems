__all__ = ["remove_extreme_movies_and_users"]

from typing import Any, Optional, List, Tuple
import pandas as pd


def summarize_ratings(
    ratings_df: pd.DataFrame,
    agg_column: str,
    count_column: str,
    output_name: str,
) -> pd.DataFrame:
    """Function that returns number of ratings aggregated by a column of interest (user, movie).

    Args:
        ratings_df (pd.DataFrame): Base dataframe containing users, movies and corresponding ratings.
        agg_column (str): Column by which data is subsequently groupped.
        count_column (str): Column on top of which to perform "unique count".
        output_name (str): Final name of count column.

    Returns:
        pd.DataFrame: Dataframe with aggregations computed.
    """
    return (
        ratings_df.groupby(agg_column)
        .agg(
            {
                count_column: pd.Series.nunique,
            }
        )
        .rename(
            columns={
                count_column: output_name,
            }
        )
        .reset_index()
    )


def get_outliers(
    df: pd.DataFrame,
    values_column: str,
    target_column: str,
    lower_boundary: Optional[int] = None,
    upper_boundary: Optional[int] = None,
) -> List[Any]:
    """Function that generate a list of outliers from a column of interest, based on thresholds for a column with values.

    Args:
        df (pd.DataFrame): Input dataframe.
        values_column (str): Name of column containing values to apply thresholds.
        target_column (str): Name of column to retrieve outlier values.
        lower_boundary (int, optional): Lower threshold for 'values_column'. Defaults to None.
        upper_boundary (int, optional): Upper threshold for 'values_column'. Defaults to None.

    Returns:
        List[Any]: List of values that are considered to be outliers.
    """
    to_remove = set()
    if lower_boundary:
        to_remove |= set(
            df.loc[df[values_column] < lower_boundary, target_column].tolist()
        )
    if upper_boundary:
        to_remove |= set(
            df.loc[df[values_column] > upper_boundary, target_column].tolist()
        )
    return list(to_remove)


def remove_extreme_movies_and_users(
    ratings_df: pd.DataFrame,
    movie_id_column: str,
    user_id_column: str,
    movie_ratings_boundaries: Tuple[Optional[int], Optional[int]] = (None, None),
    user_ratings_boundaries: Tuple[Optional[int], Optional[int]] = (None, None),
) -> pd.DataFrame:
    """Function that removes movies and users considered to be outliers from the dataset containing ratings.

    Args:
        ratings_df (pd.DataFrame): Base dataframe containing users, movies and corresponding ratings.
        movie_id_column (str): Name of column containing movie Ids.
        user_id_column (str): Name of column containing user Ids.
        movie_ratings_boundaries (Tuple[int, int], optional): Thresholds for movies'ratings. Defaults to (None, None).
        user_ratings_boundaries (Tuple[int, int], optional): Thresholds for users'ratings. Defaults to (None, None).

    Returns:
        pd.DataFrame: Dataframe without users and movies that are not in the range [lower, upper] for the specified boundaries.
    """

    n_ratings = "n_ratings"
    target_columns = [user_id_column, movie_id_column]
    for target_column, boundaries in zip(
        target_columns, [user_ratings_boundaries, movie_ratings_boundaries]
    ):
        stats = summarize_ratings(
            ratings_df=ratings_df,
            agg_column=target_column,
            count_column=[
                column for column in target_columns if column != target_column
            ][0],
            output_name=n_ratings,
        )
        lower, upper = boundaries
        to_remove = get_outliers(
            df=stats,
            values_column=n_ratings,
            target_column=target_column,
            lower_boundary=lower,
            upper_boundary=upper,
        )
        if to_remove:
            ratings_df = ratings_df[~ratings_df[target_column].isin(to_remove)].copy()

    return ratings_df
