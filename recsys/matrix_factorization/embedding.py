__all__ = ["EmbeddingMatrixFactorization"]

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf

from recsys.utils.errors import ColdStartProblem
from recsys.utils.logging import logger


class EmbeddingMatrixFactorization(tf.keras.models.Model):
    """Class that uses matrix factorization to build a recommender system by leveraging deep learning capabilities.
    Embedding layers are applied for obtaining latent factor for users and items.

    Parameters:
        latent_dimension (int): Dimension of latent space.
        user_mapping (Dict[int, int]): Dictionary containing the mapping from user ids to indices (starting from 0).
        item_mapping (Dict[int, int]): Dictionary containing the mapping from item ids to indices (starting from 0).
        use_dense_layers (bool, optional): Whether or not to use dense layers after embedding layers. Defaults to False.
    """

    _USER_EMBEDDING_NAME = "user_embedding"
    _ITEM_EMBEDDING_NAME = "item_embedding"

    def __init__(
        self,
        latent_dimension: int,
        user_mapping: Dict[int, int],
        item_mapping: Dict[int, int],
        use_dense_layers: bool = False,
        **kwargs,
    ) -> None:
        """Constructor method for EmbeddingMatrixFactorization.

        Args:
            latent_dimension (int): Dimension of latent space.
            user_mapping (Dict[int, int]): Dictionary containing the mapping from user ids to indices (starting from 0).
            item_mapping (Dict[int, int]): Dictionary containing the mapping from item ids to indices (starting from 0).
            use_dense_layers (bool, optional): Whether or not to use dense layers after embedding layers. Defaults to False.
        """
        super(EmbeddingMatrixFactorization, self).__init__(**kwargs)

        self._latent_dimension = latent_dimension
        logger.info("Latent space dimension set to %d", self._latent_dimension)
        self._use_dense_layers = use_dense_layers
        self._user_mapping = user_mapping
        self._item_mapping = item_mapping

        self._user_embedding = tf.keras.layers.Embedding(
            input_dim=max(self._user_mapping.values()) + 1,
            output_dim=self._latent_dimension,
            name=self._USER_EMBEDDING_NAME,
        )
        self._item_embedding = tf.keras.layers.Embedding(
            input_dim=max(self._item_mapping.values()) + 1,
            output_dim=self._latent_dimension,
            name=self._ITEM_EMBEDDING_NAME,
        )

        self._dense_layers_user = []
        self._dense_layers_item = []

        if self._use_dense_layers:
            cur_dim = self._latent_dimension // 2
            while cur_dim >= 8:
                self._dense_layers_user.append(
                    tf.keras.layers.Dense(
                        units=cur_dim, activation="relu", name=f"user_dense_{cur_dim}"
                    )
                )
                self._dense_layers_item.append(
                    tf.keras.layers.Dense(
                        units=cur_dim, activation="relu", name=f"item_dense_{cur_dim}"
                    )
                )
                logger.info("Dense layer with %d units added to model...", cur_dim)
                cur_dim //= 2

        self._dot_product = tf.keras.layers.Dot(axes=(-1, -1))

    @property
    def latent_dimension(self) -> int:
        """(int) Dimension of latent space"""
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
    def use_dense_layers(self) -> bool:
        """(bool) Whether or not to use dense layers after embedding layers"""
        return self._use_dense_layers

    def call(
        self, inputs: Tuple[tf.Tensor, tf.Tensor], training=None, mask=None
    ) -> tf.Tensor:
        """Method for the forward pass of the neural network.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): A tuple of tensors containing user index and item index, respectively.

        Returns:
            tf.Tensor: Predictions of the final layer.
        """
        user, item = inputs

        if training or mask:
            logger.warning(
                "`training` or `mask` passed as arguments to `call` method..."
            )

        user_emb = self._user_embedding(user)
        item_emb = self._item_embedding(item)

        user_emb = tf.keras.layers.Reshape(target_shape=(self._latent_dimension,))(
            user_emb
        )
        item_emb = tf.keras.layers.Reshape(target_shape=(self._latent_dimension,))(
            item_emb
        )

        for user_dense, item_dense in zip(
            self._dense_layers_user, self._dense_layers_item
        ):
            user_emb = user_dense(user_emb)
            item_emb = item_dense(item_emb)

        return self._dot_product([user_emb, item_emb])

    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Method for retrieving latent factor for a given user.

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
        embedding_layer = list(
            filter(lambda x: self._USER_EMBEDDING_NAME == x.name, self.layers)
        )[0]
        return embedding_layer(tf.constant(user)).numpy()

    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """Method for retrieving latent factor for a given item.

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
        embedding_layer = list(
            filter(lambda x: self._ITEM_EMBEDDING_NAME == x.name, self.layers)
        )[0]
        return embedding_layer(tf.constant(item)).numpy()

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
        rating = self.call((tf.constant([user]), tf.constant([item])))
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
        items = tf.range(
            start=min(self._item_mapping.values()),
            limit=max(self._item_mapping.values()) + 1,
        )
        user = tf.constant(items.shape[0] * [user])
        ratings = self.call((user, items)).numpy()

        idx2item = {idx: item_id for item_id, idx in self._item_mapping.items()}
        output = pd.DataFrame(
            {
                "item_id": [idx2item[idx] for idx in items.numpy().tolist()],
                "score": ratings.squeeze().tolist(),
            }
        )
        output = output.sort_values("score", ascending=False)
        if item_ids_to_exclude:
            logger.info("Filtering out unrelevant items from output...")
            output = output[~output.item_id.isin(item_ids_to_exclude)]
        return output.reset_index(drop=True)
