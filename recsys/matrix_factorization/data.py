__all__ = ["create_data_generator", "convert_dataframe_into_train_and_validation_generators"]

from typing import Dict, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf

from recsys.utils.errors import NotA2DArray, FeaturesNotAllowedForMatrixFactorization


def create_data_generator(
    X: np.ndarray, y: np.ndarray, batch_size: int, seed: int = 99
) -> tf.data.Dataset:
    """Function that creates tensorflow tensor generator on top of numpy arrays.

    Args:
        X (np.ndarray): 2-D array containing features - user_id and item_id.
        y (np.ndarray): Array containing rating/outcomes given by user_id for each item_id.
        batch_size (int): Batch size used for further model training.
        seed (int, optional): Random seed for repreducibility purposes. Defaults to 99.

    Raises:
        NotA2DArray: Raised when X is not a 2-D array.
        FeaturesNotAllowedForMatrixFactorization: Raised when X has more than 2 columns.

    Returns:
        tf.data.Dataset: Tensorflow dataset generator to be fed to neural network.
    """
    if X.ndim != 2:
        raise NotA2DArray("X must be a 2D-array!")
    if X.shape[-1] != 2:
        raise FeaturesNotAllowedForMatrixFactorization(
            "X must have 2 columns: user_id + item_id!"
        )
    y_ = y.copy()
    if y_.ndim == 1:
        y_ = y_.reshape(-1, 1)

    data = np.concatenate([X, y_], axis=-1)
    data_gen = tf.data.Dataset.from_tensor_slices(data)
    data_gen = data_gen.map(lambda x: ((x[0], x[1]), x[2]))
    data_gen = data_gen.shuffle(buffer_size=1_000, seed=seed).batch(
        batch_size=batch_size
    )
    return data_gen


def convert_dataframe_into_train_and_validation_generators(
    df: pd.DataFrame,
    user_id_field_name: str,
    item_id_field_name: str,
    ratings_field_name: str,
    user_to_index_mapping: Dict[int, int],
    item_to_index_mapping: Dict[int, int],
    val_size: float = 0.30,
    train_batch_size: int = 64,
    val_batch_size: int = 128,
    seed: int = 99,
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Function that accepts a dataframe containing ratings and convert it into train and validation generators.

    Args:
        df (pd.DataFrame): Dataframe containing users, items and ratings.
        user_id_field_name (str): Name of the column containing user ids.
        item_id_field_name (str): Name of the column containing item ids.
        ratings_field_name (str): Name of the column containing ratings.
        user_to_index_mapping (Dict[int, int]): Mapping from user_id to an index in range [0, # of users).
        item_to_index_mapping (Dict[int, int]): Mapping from item_id to an index in range [0, # of items).
        val_size (float, optional): Fraction to be used as validation set. Defaults to 0.30.
        train_batch_size (int, optional): Batch size for the training generator. Defaults to 64.
        val_batch_size (int, optional): Batch size for the validation generator. Defaults to 128.
        seed (int, optional): Random seed for repreducibility purposes. Defaults to 99.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Tuple containing training and validation generators, respectively.
    """
    df_ = df.copy()
    df_["user_index"] = df_[user_id_field_name].map(user_to_index_mapping)
    df_["item_index"] = df_[item_id_field_name].map(item_to_index_mapping)

    df_ = df_.sample(frac=1.0, replace=False, seed=seed)
    df_train = df_.iloc[: -int(len(df_) * val_size)]
    df_val = df_.iloc[-int(len(df_) * val_size) :]

    X_train = df_train[["user_index", "item_index"]].values
    y_train = df_train[ratings_field_name].values

    X_val = df_val[["user_index", "item_index"]].values
    y_val = df_val[ratings_field_name].values

    train_gen = create_data_generator(
        X=X_train, y=y_train, batch_size=train_batch_size, seed=seed
    )
    val_gen = create_data_generator(
        X=X_val, y=y_val, batch_size=val_batch_size, seed=seed
    )
    return (train_gen, val_gen)
