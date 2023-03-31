__all__ = ["split_dataset"]

from typing import List, Optional, Tuple
import pandas as pd


def split_dataset(
    df: pd.DataFrame,
    sets_size: List[float],
    time_order: bool = False,
    time_column: Optional[str] = None,
    seed: int = 99,
) -> Tuple[pd.DataFrame]:
    """Function that splits the dataset containing users, items, and ratings into subsets.
    The number and size of subsets are determined by the user.

    Args:
        df (pd.DataFrame): Dataframe containing users, items, and ratings.
        sets_size (List[float], optional): List containing sizes of sets - it needs to add up to 1.0.
        time_order (bool, optional): Whether or not split should respect time order. Defaults to False.
        time_column (Optional[str], optional): Name of column containing time information. Defaults to None.
        seed (int): Random seed to ensure reproducibility.

    Raises:
        AttributeError: Raised when `time_column` is None and `time_order` is set to True.
        KeyError: Raised when `time_column` is not present in the dataframe.

    Returns:
        Tuple[pd.DataFrame]: Tuple containing sets of the original data.
    """

    df_ = df.copy()
    assert sum(sets_size) == 1.0, "Sum of all set sizes needs to be 1.0!"

    if time_order:
        if time_column is None:
            raise AttributeError(
                "`time_column` needs to be informed if `time_order = True`"
            )
        if not time_column in df_.columns:
            raise KeyError(f"{time_column} not found in the dataset!")
        df_ = df_.sort_values(by=time_column)

        total_length = len(df_)
        df_sets = []
        cum_size = 0.0
        for size in sets_size[:-1]:
            cum_size += size
            time_cut = df_.iloc[int(cum_size * total_length)][time_column].normalize()
            df_sets.append(df_[df_[time_column] <= time_cut])
        df_sets.append(df_[df_[time_column] > time_cut])
    else:
        df_ = df_.sample(frac=1.0, replace=False, random_state=seed)
        total_length = len(df_)
        df_sets = []
        cum_size = 0.0
        for size in sets_size[:-1]:
            df_sets.append(
                df_.iloc[
                    int(cum_size * total_length) : int((cum_size + size) * total_length)
                ]
            )
            cum_size += size
        df_sets.append(df_.iloc[int(cum_size * total_length) :])

    return tuple(df_sets)
