__all__ = ["ALS"]

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import tensorflow as tf

from recsys.utils.logging import logger
from recsys.utils.errors import ColdStartProblem


class ALS(object):
    """Class that implements Alternating Least Squares (ALS) for recommendations.

    Parameters:
        latent_dimension (int): Latent space dimension for embeddings.
        user_mapping (Dict[int, int]): Dictionary containing the mapping from user ids to indices (starting from 0).
        item_mapping (Dict[int, int]): Dictionary containing the mapping from item ids to indices (starting from 0).
        learning_rate (float, optional): Learning rate for training with gradient descent. Defaults to 1e-1.
        seed (int, optional): Random seed for reproducibility purposes. Defaults to 99.
    """

    def __init__(
        self,
        latent_dimension: int,
        user_mapping: Dict[int, int],
        item_mapping: Dict[int, int],
        learning_rate: float = 1e-1,
        seed: int = 99,
    ) -> None:
        """Constructor method for ALS class.

        Args:
            latent_dimension (int): Latent space dimension for embeddings.
            user_mapping (Dict[int, int]): Dictionary containing the mapping from user ids to indices (starting from 0).
            item_mapping (Dict[int, int]): Dictionary containing the mapping from item ids to indices (starting from 0).
            learning_rate (float, optional): Learning rate for training with gradient descent. Defaults to 1e-1.
            seed (int, optional): Random seed for reproducibility purposes. Defaults to 99.
        """
        self._latent_dimension = latent_dimension
        self._user_mapping = user_mapping
        self._item_mapping = item_mapping
        self._user_dimension = max(self._user_mapping.values()) + 1
        self._item_dimension = max(self._item_mapping.values()) + 1
        self._learning_rate = learning_rate
        self._seed = seed

        self.user_embeddings = {
            idx: tf.Variable(
                initial_value=np.random.uniform(size=(self._latent_dimension, 1)),
                name=f"user_{idx}",
            )
            for idx in range(self._user_dimension)
        }
        self.item_embeddings = {
            idx: tf.Variable(
                initial_value=np.random.uniform(size=(self._latent_dimension, 1)),
                name=f"item_{idx}",
            )
            for idx in range(self._item_dimension)
        }

    @property
    def latent_dimension(self) -> int:
        """(int) Latent space dimension for embeddings"""
        return self._latent_dimension

    @property
    def user_mapping(self) -> Dict[int, int]:
        """(Dict[int, int]) Dictionary containing the mapping from user ids to indices (starting from 0)"""
        return self._user_mapping

    @property
    def item_mapping(self) -> Dict[int, int]:
        """(Dict[int, int]) Dictionary containing the mapping from item ids to indices (starting from 0)"""
        return self._item_mapping

    @property
    def user_dimension(self) -> int:
        """(int) Number of unique users"""
        return self._user_dimension

    @property
    def item_dimension(self) -> int:
        """(int) Number of unique items"""
        return self._item_dimension

    @property
    def learning_rate(self) -> float:
        """(float) Learning rate for training with gradient descent"""
        return self._learning_rate

    @property
    def seed(self) -> int:
        """(int) Random seed for reproducibility purposes"""
        return self._seed

    def _compute_rating(self, user_index: int, item_index: int) -> tf.Tensor:
        """Method to compute final rating as a dot product between user and item latent factors.

        Args:
            user_index (int): Index of user, within range [0, self.self._user_dimension)
            item_index (int): Index of item, within range [0, self.self._item_dimension)

        Returns:
            tf.Tensor: 1-D Tensor containing rating.
        """
        return tf.reshape(
            tf.transpose(self.user_embeddings[user_index])
            @ self.item_embeddings[item_index],
            shape=(-1,),
        )

    def _compute_mse(
        self, users_batch: tf.Tensor, items_batch: tf.Tensor, ratings_batch: tf.Tensor
    ) -> tf.Tensor:
        """Method to compute Mean Squared Error (MSE) as the loss function for training.

        Args:
            users_batch (tf.Tensor): Batch of user indices.
            items_batch (tf.Tensor): Batch of item indices.
            ratings_batch (tf.Tensor): Batch of ratings.

        Returns:
            tf.Tensor: Tensor containing MSE for the current batch.
        """
        mse = []
        for u, i, y in zip(users_batch, items_batch, ratings_batch):
            predicted_rating = self._compute_rating(
                user_index=u.numpy(),
                item_index=i.numpy(),
            )
            mse.append(tf.square(predicted_rating - y))

        return tf.reduce_mean(tf.stack(mse))

    def train_batch(
        self, users_batch: tf.Tensor, items_batch: tf.Tensor, ratings_batch: tf.Tensor
    ) -> float:
        """Method to run the training loop for a single batch.

        Args:
            users_batch (tf.Tensor): Batch of user indices.
            items_batch (tf.Tensor): Batch of item indices.
            ratings_batch (tf.Tensor): Batch of ratings.

        Returns:
            float: MSE for the current batch.
        """
        unique_users = set(users_batch.numpy().tolist())
        unique_items = set(items_batch.numpy().tolist())
        with tf.GradientTape() as tape:
            mse = self._compute_mse(
                users_batch=users_batch,
                items_batch=items_batch,
                ratings_batch=ratings_batch,
            )

        sources = [self.user_embeddings[u] for u in unique_users] + [
            self.item_embeddings[i] for i in unique_items
        ]
        gradients = tape.gradient(target=mse, sources=sources)
        for gradient, user in zip(gradients, unique_users):
            self.user_embeddings[user].assign_add(-gradient * self._learning_rate)

        for gradient, item in zip(gradients[len(unique_users) :], unique_items):
            self.item_embeddings[item].assign_add(-gradient * self._learning_rate)

        return float(mse.numpy())

    def fit(
        self,
        train_generator: tf.data.Dataset,
        epochs: int,
        val_generator: Optional[tf.data.Dataset] = None,
    ) -> None:
        """Method to fit the recommender system on the whole dataset.

        Args:
            train_generator (tf.data.Dataset): Training generator containing user indices, item indices and ratings.
            epochs (int): Total number of epochs for training.
            val_generator (Optional[tf.data.Dataset], optional): Validation generator containing user indices, item indices and ratings.. Defaults to None.
        """
        for epoch in range(1, epochs + 1):
            epoch_text = max(4 - len(str(epoch)), 0) * "0" + str(epoch)
            mse_epoch = []
            for (users_batch, items_batch), ratings_batch in train_generator:
                mse_batch = self.train_batch(
                    users_batch=users_batch,
                    items_batch=items_batch,
                    ratings_batch=ratings_batch,
                )
                mse_epoch.append(mse_batch)
            mse_epoch = np.round(np.mean(mse_epoch), 2)
            if val_generator is not None:
                mse_epoch_val = []
                for (users_batch, items_batch), ratings_batch in val_generator:
                    mse_batch_val = self._compute_mse(
                        users_batch=users_batch,
                        items_batch=items_batch,
                        ratings_batch=ratings_batch,
                    )
                    mse_epoch_val.append(mse_batch_val)
                mse_epoch_val = np.round(np.mean(mse_epoch_val), 2)
                logger.info(
                    "[%s EPOCH] Train MSE = %f, Val MSE = %f",
                    epoch_text,
                    mse_epoch,
                    mse_epoch_val,
                )
            else:
                logger.info("[%s EPOCH] Train MSE = %f", epoch_text, mse_epoch)

    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Method for retrieving the latent factors for a given user.

        Args:
            user_id (int): User id for which embeddings will be retrieved.

        Raises:
            ColdStartProblem: Raised when `user_id` was not present in the training dataset.

        Returns:
            np.ndarray: Array containing latent factors for the user.
        """
        user = self._user_mapping.get(user_id, None)
        if not user:
            raise ColdStartProblem(f"User {user_id} not in the training data!")
        return self.user_embeddings[user].numpy().reshape(-1)

    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """Method for retrieving the latent factors for a given item.

        Args:
            item_id (int): Item id for which embeddings will be retrieved.

        Raises:
            ColdStartProblem: Raised when `item_id` was not present in the training dataset.

        Returns:
            np.ndarray: Array containing latent factors for the item.
        """
        item = self._item_mapping.get(item_id, None)
        if not item:
            raise ColdStartProblem(f"Item {item_id} not in the training data!")
        return self.user_embeddings[item].numpy().reshape(-1)

    def predict_rating_for_user_item(self, user_id: int, item_id: int) -> float:
        """Method for retrieving a rating/score for a (user_id, item_id) pair.

        Args:
            user_id (int): User id to be considered.
            item_id (int): Item id to be considered.

        Raises:
            ColdStartProblem: Raised when either `user_id` or `item_id` was not present in the training dataset.

        Returns:
            float: Predicted rating/score.
        """
        user = self._user_mapping.get(user_id, None)
        if not user:
            raise ColdStartProblem(f"User {user_id} not in the training data!")
        item = self._item_mapping.get(item_id, None)
        if not item:
            raise ColdStartProblem(f"Item {item_id} not in the training data!")
        rating = self._compute_rating(user_index=user, item_index=item)
        return float(rating.numpy().squeeze())

    def predict_rating_for_user(
        self, user_id: int, item_ids_to_exclude: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Method for screening over all items for a given user and then returning ratings/scores for each one.

        Args:
            user_id (int): User id to be considered.
            item_ids_to_exclude (Optional[List[int]], optional): Whether some items are to be excluded from the output. Defaults to None.

        Raises:
            ColdStartProblem: Raised when `user_id` was not present in the training dataset.

        Returns:
            pd.DataFrame: Dataframe containing final items and associated scores sorted in a descending order.
        """
        user = self._user_mapping.get(user_id, None)
        if not user:
            raise ColdStartProblem(f"User {user_id} not in the training data!")
        items = np.arange(max(self._item_mapping.values()) + 1)
        ratings = []
        for item in items:
            rating = self._compute_rating(user_index=user, item_index=item)
            ratings.append(float(rating.numpy().squeeze()))

        idx2item = {idx: item_id for item_id, idx in self._item_mapping.items()}
        output = pd.DataFrame(
            {
                "item_id": [idx2item[idx] for idx in items.tolist()],
                "score": ratings,
            }
        )
        output = output.sort_values("score", ascending=False)
        if item_ids_to_exclude:
            logger.info("Filtering out unrelevant items from output...")
            output = output[~output.item_id.isin(item_ids_to_exclude)]
        return output.reset_index(drop=True)
