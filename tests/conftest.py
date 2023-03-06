import pandas as pd
import numpy as np
import pytest


@pytest.fixture(scope="module")
def dummy_ratings_df() -> pd.DataFrame:
    np.random.seed(123)
    return pd.DataFrame(
        {
            "movieId": [1, 1, 1, 2, 2, 2, 3, 4, 4, 5, 5],
            "userId": [1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 3],
            "rating": np.random.uniform(low=0.5, high=5.0, size=11),
        }
    )
