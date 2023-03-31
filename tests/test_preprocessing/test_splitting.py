import pandas as pd
import pytest

from recsys.preprocessing.splitting import split_dataset


def test_split_dataset_raisesAttributeError(dummy_ratings_df: pd.DataFrame):
    with pytest.raises(AttributeError):
        _ = split_dataset(
            df=dummy_ratings_df,
            sets_size=[0.60, 0.20, 0.20],
            time_order=True,
        )


def test_split_dataset_raisesKeyError(dummy_ratings_df: pd.DataFrame):
    with pytest.raises(KeyError):
        _ = split_dataset(
            df=dummy_ratings_df,
            sets_size=[0.60, 0.20, 0.20],
            time_order=True,
            time_column="NOT_AVAILABLE",
        )


def test_split_dataset_raisesAssertionError(dummy_ratings_df: pd.DataFrame):
    with pytest.raises(AssertionError):
        _ = split_dataset(
            df=dummy_ratings_df,
            sets_size=[0.60, 0.50, 0.20],
            time_order=False,
        )


def test_split_dataset_expectedOutputRandom(dummy_ratings_df: pd.DataFrame):
    # OUTPUT
    output = split_dataset(
        df=dummy_ratings_df.drop(columns="timestamp"),
        sets_size=[0.60, 0.20, 0.20],
        time_order=False,
    )

    # EXPECTED
    expected = (
        pd.DataFrame(
            {
                "movieId": {5: 2, 9: 5, 6: 3, 7: 4, 4: 2, 2: 1},
                "userId": {5: 3, 9: 1, 6: 4, 7: 1, 4: 2, 2: 3},
                "rating": {
                    5: 2.4039790705600743,
                    9: 2.2645288318736774,
                    6: 4.9134388927307695,
                    7: 3.5817338236318847,
                    4: 3.7376103640350338,
                    2: 1.520831541038914,
                },
            }
        ),
        pd.DataFrame(
            {
                "movieId": {0: 1, 10: 5},
                "userId": {0: 1, 10: 3},
                "rating": {0: 3.6341113351903775, 10: 2.0443010726789126},
            }
        ),
        pd.DataFrame(
            {
                "movieId": {8: 4, 3: 2, 1: 1},
                "userId": {8: 2, 3: 1, 1: 2},
                "rating": {
                    8: 2.664193556679624,
                    3: 2.9809164608730105,
                    1: 1.7876270072767075,
                },
            }
        ),
    )

    # ASSERT
    for out, exp in zip(output, expected):
        pd.testing.assert_frame_equal(out, exp)


def test_split_dataset_expectedOutputTimeOrder(dummy_ratings_df: pd.DataFrame):
    # OUTPUT
    output = split_dataset(
        df=dummy_ratings_df,
        sets_size=[0.60, 0.40],
        time_order=True,
        time_column="timestamp",
    )

    # EXPECTED
    expected = (
        pd.DataFrame(
            {
                "timestamp": {
                    0: pd.Timestamp("2022-01-01 00:00:00"),
                    1: pd.Timestamp("2022-01-02 00:00:00"),
                    2: pd.Timestamp("2022-01-03 00:00:00"),
                    3: pd.Timestamp("2022-01-04 00:00:00"),
                    4: pd.Timestamp("2022-01-05 00:00:00"),
                    5: pd.Timestamp("2022-01-06 00:00:00"),
                    6: pd.Timestamp("2022-01-07 00:00:00"),
                },
                "movieId": {0: 1, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 3},
                "userId": {0: 1, 1: 2, 2: 3, 3: 1, 4: 2, 5: 3, 6: 4},
                "rating": {
                    0: 3.6341113351903775,
                    1: 1.7876270072767075,
                    2: 1.520831541038914,
                    3: 2.9809164608730105,
                    4: 3.7376103640350338,
                    5: 2.4039790705600743,
                    6: 4.9134388927307695,
                },
            },
        ),
        pd.DataFrame(
            {
                "timestamp": {
                    7: pd.Timestamp("2022-01-08 00:00:00"),
                    8: pd.Timestamp("2022-01-09 00:00:00"),
                    9: pd.Timestamp("2022-01-10 00:00:00"),
                    10: pd.Timestamp("2022-01-11 00:00:00"),
                },
                "movieId": {7: 4, 8: 4, 9: 5, 10: 5},
                "userId": {7: 1, 8: 2, 9: 1, 10: 3},
                "rating": {
                    7: 3.5817338236318847,
                    8: 2.664193556679624,
                    9: 2.2645288318736774,
                    10: 2.0443010726789126,
                },
            }
        ),
    )

    print(output[0].to_dict())
    print(output[1].to_dict())

    # ASSERT
    for out, exp in zip(output, expected):
        pd.testing.assert_frame_equal(out, exp)
