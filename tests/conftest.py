import os
import pandas as pd
import numpy as np
import pytest


@pytest.fixture(scope="module")
def dummy_ratings_df() -> pd.DataFrame:
    np.random.seed(123)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start="2022-01-01", end="2022-01-11", freq="1D"),
            "movieId": [1, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5],
            "userId": [1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 3],
            "rating": np.random.uniform(low=0.5, high=5.0, size=11),
        }
    )


@pytest.fixture(scope="module")
def movielens_sample() -> pd.DataFrame:
    wdir = os.path.abspath(os.curdir)
    return pd.read_csv(os.path.join(wdir, "tests", "movielens_sample.csv"))


@pytest.fixture(scope="module")
def output_predictions_for_user_user_filtering() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movieId": {
                250: 7153,
                70: 267,
                322: 7380,
                69: 4361,
                462: 4022,
            },
            "predicted_rating": {
                250: 5.666849816849817,
                70: 5.166849816849817,
                322: 5.166849816849817,
                69: 5.166849816849817,
                462: 5.166849816849817,
            },
        }
    )
