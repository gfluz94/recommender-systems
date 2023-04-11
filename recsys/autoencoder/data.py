__all__ = ["UserItemMatrix"]

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import tensorflow as tf


class UserItemMatrix(tf.keras.utils.Sequence):
    """Class that implements a tensorflow.keras sequence to be fed to the neural netowrk.
    The mais advantage of this class is that computations are carried out on the go, avoiding memory issues.

    Parameters:
        data (pd.DataFrame): Dataframe containing users, items, and possibly ratings.
        user_id_field_name (str): Name of the column containing user ids.
        item_id_field_name (str): Name of the column containing item ids.
        user_mapping (Dict[int, int]): Dictionary containing the mapping from user ids to indices (starting from 0).
        item_mapping (Dict[int, int]): Dictionary containing the mapping from item ids to indices (starting from 0).
        batch_size (int, optional): Size of the batches to be generated. Defaults to 64.
        seed (int, optional): Random seed for reproducibility. Defaults to 99.

    Raises:
        AttributeError: Raised when items retrieved are not integers, slices or list of integers.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        user_id_field_name: str,
        item_id_field_name: str,
        user_mapping: Dict[int, int],
        item_mapping: Dict[int, int],
        batch_size: int = 64,
        seed: int = 99,
    ) -> None:
        """Constructor method for UserItemMatrix.

        Args:
            data (pd.DataFrame): Dataframe containing users, items, and possibly ratings.
            user_id_field_name (str): Name of the column containing user ids.
            item_id_field_name (str): Name of the column containing item ids.
            user_mapping (Dict[int, int]): Dictionary containing the mapping from user ids to indices (starting from 0).
            item_mapping (Dict[int, int]): Dictionary containing the mapping from item ids to indices (starting from 0).
            batch_size (int, optional): Size of the batches to be generated. Defaults to 64.
            seed (int, optional): Random seed for reproducibility. Defaults to 99.
        """
        super(UserItemMatrix, self).__init__()
        self._user_id_field_name = user_id_field_name
        self._item_id_field_name = item_id_field_name
        self._user_mapping = user_mapping
        self._item_mapping = item_mapping
        self._batch_size = batch_size
        self._seed = seed

        self._total_users = max(self._user_mapping.values()) + 1
        self._total_items = max(self._item_mapping.values()) + 1

        self._users_items = data.groupby(self._user_id_field_name).agg(
            {self._item_id_field_name: list}
        )
        self._users_items = self._users_items.sample(
            frac=1.0, replace=False, random_state=seed
        )[self._item_id_field_name].to_dict()
        self._users = list(self._users_items.keys())
        self._user_arrays = {}

    def _get_input_for_user(self, user: int) -> np.ndarray:
        """Method that fetches vector for a given user. If vector has already been computed for the user, it simply retrieves from memory.

        Args:
            user (int): User id of interest.

        Returns:
            np.ndarray: Array of size (1, total of items) for the user id - containing either ratings or binary variable for each item.
        """
        retrieval = self._user_arrays.get(user, None)
        if retrieval is None:
            items = self._users_items[user]
            items_idx = list(map(self._item_mapping.get, items))
            retrieval = np.zeros((1, self._total_items))
            retrieval[:, items_idx] = 1.0
            self._user_arrays[user] = retrieval
        return retrieval

    def _get_batch(self, batch_index: int) -> np.ndarray:
        """Method that returns a batch of vectors for users.

        Args:
            batch_index (int): Batch index.

        Returns:
            np.ndarray: Array of size (batch size, total of items) for a batch of users.
        """
        outputs = []
        for idx in range(
            batch_index * self._batch_size, (batch_index + 1) * self._batch_size
        ):
            idx_adj = idx % self._total_users
            outputs.append(self._get_input_for_user(self._users[idx_adj]))
        return np.concatenate(outputs, axis=0)

    def __len__(self) -> int:
        """Length of the sequence.

        Returns:
            int: Number of batches in the data.
        """
        return self._total_users // self._batch_size + 1

    def __getitem__(
        self, index: Union[int, slice, List[int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Method that retrieves a batch of item vectors for a batch of users.

        Args:
            index (Union[int, slice, List[int]]): Index of the batch.

        Raises:
            AttributeError: Raised when items retrieved are not integers, slices or list of integers.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of arrays for further NN training - (X, y).
        """
        if isinstance(index, int):
            data = self._get_batch(batch_index=index)
            return data, data
        elif isinstance(index, slice):
            outputs = []
            start = index.start if index.start else 0
            stop = index.stop if index.stop else self._total_users
            for idx in range(start, stop):
                outputs.append(self._get_batch(batch_index=idx))
            outputs = np.concatenate(outputs, axis=0)
            return outputs, outputs
        elif isinstance(index, list):
            outputs = []
            for idx in index:
                outputs.append(self._get_batch(batch_index=idx))
            outputs = np.concatenate(outputs, axis=0)
            return outputs, outputs
        raise AttributeError(
            "`index` must be either integer, slice or list of integers."
        )
