import pandas as pd

from recsys.preprocessing.filtering import (
    summarize_ratings,
    get_outliers,
    remove_extreme_movies_and_users,
)


def test_summarize_ratings_expectedOutputUsers(dummy_ratings_df: pd.DataFrame):
    # OUTPUT
    output = summarize_ratings(
        ratings_df=dummy_ratings_df,
        agg_column="userId",
        count_column="movieId",
        output_name="n_ratings",
    )

    # EXPECTED
    expected = pd.DataFrame(
        {"userId": {0: 1, 1: 2, 2: 3, 3: 4}, "n_ratings": {0: 4, 1: 3, 2: 3, 3: 1}}
    )

    # ASSERT
    pd.testing.assert_frame_equal(output, expected)


def test_summarize_ratings_expectedOutputMovies(dummy_ratings_df: pd.DataFrame):
    # OUTPUT
    output = summarize_ratings(
        ratings_df=dummy_ratings_df,
        agg_column="movieId",
        count_column="userId",
        output_name="n_ratings",
    )

    # EXPECTED
    expected = pd.DataFrame(
        {
            "movieId": {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
            "n_ratings": {0: 3, 1: 3, 2: 1, 3: 2, 4: 2},
        }
    )

    # ASSERT
    pd.testing.assert_frame_equal(output, expected)


def test_get_outliers_expectedOutput():
    # INPUT
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1, -10, 3, 400, 5],
        }
    )

    # OUTPUT
    output = get_outliers(
        df=df,
        values_column="b",
        target_column="a",
        lower_boundary=0,
        upper_boundary=100,
    )

    # EXPECTED
    expected = [1, 3, 5]

    # ASSERT
    output == expected


def test_remove_extreme_movies_and_users_expectedOutput(dummy_ratings_df: pd.DataFrame):
    # OUTPUT
    output = remove_extreme_movies_and_users(
        ratings_df=dummy_ratings_df,
        movie_id_column="movieId",
        user_id_column="userId",
        movie_ratings_boundaries=(2, None),
        user_ratings_boundaries=(1, 3),
    )

    # EXPECTED
    expected = pd.DataFrame(
        {
            "movieId": {1: 1, 2: 1, 4: 2, 5: 2},
            "userId": {1: 2, 2: 3, 4: 2, 5: 3},
            "rating": {
                1: 1.7876270072767075,
                2: 1.520831541038914,
                4: 3.7376103640350338,
                5: 2.4039790705600743,
            },
        }
    )

    # ASSERT
    pd.testing.assert_frame_equal(output, expected)
