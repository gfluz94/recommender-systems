__all__ = ["UserVectorAutoEncoder"]

from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf

from recsys.utils.errors import ModelNotFittedYet


class UserVectorAutoEncoder(object):
    """Class that implements an autoencoder for reconstructing binary inputs of items interacted with by the user.

    Parameters:
        latent_space_dimension (int): Dimension of latent space.
        item_mapping (Dict[int, int]): Dictionary containing the mapping from item ids to indices (starting from 0).
        encoder_hidden_layers (Optional[List[int]], optional): List with units for each layer to be added to the encoder. Defaults to None.
        learning_rate (float, optional): Learning rate for the training loop. Defaults to 1e-3.

    Raises:
        ModelNotFittedYet: Raised when inference methods are called before fitting the model.
    """

    def __init__(
        self,
        latent_space_dimension: int,
        item_mapping: Dict[int, int],
        encoder_hidden_layers: Optional[List[int]] = None,
        learning_rate: float = 1e-3,
    ) -> None:
        """Constructor method for UserVectorAutoEncoder.

        Args:
            latent_space_dimension (int): Dimension of latent space.
            item_mapping (Dict[int, int]): Dictionary containing the mapping from item ids to indices (starting from 0).
            encoder_hidden_layers (Optional[List[int]], optional): List with units for each layer to be added to the encoder. Defaults to None.
            learning_rate (float, optional): Learning rate for the training loop. Defaults to 1e-3.

        Raises:
            ModelNotFittedYet: Raised when inference methods are called before fitting the model.
        """
        self._latent_space_dimension = latent_space_dimension
        self._item_mapping = item_mapping
        self._item_mapping_inv = {v: k for k, v in self._item_mapping.items()}
        self._encoder_hidden_layers = (
            encoder_hidden_layers if encoder_hidden_layers else []
        )
        self._learning_rate = learning_rate
        self._input_dim = max(self._item_mapping.values()) + 1

        self._encoder = None
        self._decoder = None
        self._autoencoder = None
        self._fitted = False

    @property
    def latent_space_dimension(self) -> int:
        """(int) Dimension of latent space"""
        return self._latent_space_dimension

    @property
    def item_mapping(self) -> Dict[int, int]:
        """(Dict[int, int]) Dictionary containing the mapping from item ids to indices (starting from 0)"""
        return self._item_mapping

    @property
    def encoder_hidden_layers(self) -> Optional[List[int]]:
        """(Optional[List[int]]) List with units for each layer to be added to the encoder"""
        return self._encoder_hidden_layers

    @property
    def learning_rate(self) -> float:
        """(float) Learning rate for the training loop"""
        return self._learning_rate

    @property
    def encoder(self) -> tf.keras.models.Model:
        """(tf.keras.models.Model) Encoder model"""
        return self._get_encoder()

    @property
    def decoder(self) -> tf.keras.models.Model:
        """(tf.keras.models.Model) Decoder model"""
        return self._get_decoder()

    @property
    def model(self) -> tf.keras.models.Model:
        """(tf.keras.models.Model) Autoencoder model"""
        return self._get_autoencoder()

    @property
    def fitted(self) -> bool:
        """(bool) Whether or not model has already been fitted."""
        return self._fitted

    def _get_encoder(self) -> tf.keras.models.Model:
        """Method that builds encoder and returns it as a singleton.

        Returns:
            tf.keras.models.Model: Encoder model.
        """
        if self._encoder is None:
            x = tf.keras.layers.Input(shape=(self._input_dim,))
            inputs = x
            for layer in self._encoder_hidden_layers:
                x = tf.keras.layers.Dense(layer, activation="relu")(x)
            output = tf.keras.layers.Dense(
                self._latent_space_dimension, activation="relu"
            )(x)
            self._encoder = tf.keras.models.Model(inputs=inputs, outputs=output)
        return self._encoder

    def _get_decoder(self) -> tf.keras.models.Model:
        """Method that builds decoder and returns it as a singleton.

        Returns:
            tf.keras.models.Model: Decoder model.
        """
        if self._decoder is None:
            x = tf.keras.layers.Input(shape=(self._latent_space_dimension,))
            inputs = x
            for layer in self._encoder_hidden_layers[::-1]:
                x = tf.keras.layers.Dense(layer, activation="relu")(x)
            output = tf.keras.layers.Dense(self._input_dim, activation="sigmoid")(x)
            self._decoder = tf.keras.models.Model(inputs=inputs, outputs=output)
        return self._decoder

    def _get_autoencoder(self) -> tf.keras.models.Model:
        """Method that builds the whole autoencoder and returns it as a singleton.

        Returns:
            tf.keras.models.Model: Autoencoder model.
        """
        if self._autoencoder is None:
            self._autoencoder = tf.keras.models.Sequential(
                [self._get_encoder(), self._get_decoder()]
            )
            self._autoencoder.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self._learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(),
            )
        return self._autoencoder

    def fit(
        self,
        training_input: tf.keras.utils.Sequence,
        epochs: int,
        validation_input: Optional[tf.keras.utils.Sequence] = None,
    ) -> None:
        """Method that fits the autoencoder.

        Args:
            training_input (tf.keras.utils.Sequence): Keras sequence containing (X, y) for training.
            epochs (int): Number of epochs for training.
            validation_input (Optional[tf.keras.utils.Sequence], optional): Keras sequence containing (X, y) for validation. Defaults to None.
        """
        self._autoencoder = self._get_autoencoder()
        self._autoencoder.fit(
            training_input, validation_data=validation_input, epochs=epochs
        )
        self._fitted = True

    def predict(
        self,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Method for reconstructing one-hot-encoded user vectors.

        Args:
            inputs (np.ndarray): One-hot-encoded user vectors.

        Raises:
            ModelNotFittedYet: Raised when inference methods are called before fitting the model.

        Returns:
            np.ndarray: Reconstructed inputs.
        """
        if not self._fitted:
            raise ModelNotFittedYet(
                "`.fit()` method must be called before making predictions!"
            )
        if np.ndim(inputs) == 1:
            inputs = np.reshape(inputs, (1, -1))
        return self._autoencoder(inputs).numpy().squeeze()

    def get_user_embedding(
        self,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Method for returning the corresponding embeddings for one-hot-encoded user vectors.

        Args:
            inputs (np.ndarray): One-hot-encoded user vectors.

        Raises:
            ModelNotFittedYet: Raised when inference methods are called before fitting the model.

        Returns:
            np.ndarray: Users' latent factors.
        """
        if not self._fitted:
            raise ModelNotFittedYet(
                "`.fit()` method must be called before making predictions!"
            )
        if np.ndim(inputs) == 1:
            inputs = np.reshape(inputs, (1, -1))
        return self._encoder(inputs).numpy().squeeze()

    def recommend_items(
        self,
        inputs: np.ndarray,
        top_K_items: Optional[int] = -1,
    ) -> List[Dict[int, float]]:
        """Method for making recommendations based on users vectors.

        Args:
            inputs (np.ndarray): One-hot-encoded user vectors.
            top_K_items (Optional[int], optional): Number of top items to be returned. Defaults to -1, representing all items.

        Returns:
            List[Dict[int, float]]: List of dictionaries {item id: score} for each user vector passed as input.
        """
        predictions = self.predict(inputs=inputs)
        if np.ndim(predictions) == 1:
            predictions = predictions.reshape((1, -1))

        sorted_indices = np.argsort(predictions, axis=-1)
        sorted_scores = np.sort(predictions, axis=-1)

        if top_K_items == -1:
            retrieved_items = self._input_dim
        else:
            retrieved_items = top_K_items
        sorted_indices = sorted_indices[:, -retrieved_items:]
        sorted_scores = sorted_scores[:, -retrieved_items:]

        sorted_indices = sorted_indices.tolist()
        sorted_scores = sorted_scores.tolist()

        output = []
        for indices, scores in zip(sorted_indices, sorted_scores):
            rev_indices = indices[::-1]
            rev_scores = scores[::-1]
            output.append(
                {
                    self._item_mapping_inv.get(idx): score
                    for idx, score in zip(rev_indices, rev_scores)
                }
            )

        return output
