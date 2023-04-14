from typing import List, Optional, Tuple
import tensorflow as tf


class SamplingLayer(tf.keras.layers.Layer):
    """Custom layer on top of keras framework to allow for sampling from a normal distribution for the variational autoencoder development."""

    def __init__(self) -> None:
        """Constructor method for SamplingLayer."""
        super(SamplingLayer, self).__init__()

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], *args, **kwargs) -> tf.Tensor:
        """Method to perform the forward pass on the layer - sampling from a normal distribution.

        Args:
            inputs (Tuple[tf.Tensor, tf.Tensor]): Tuple of tensors (latent factors' mean vector, latent factors' log(Var) vector)

        Returns:
            tf.Tensor: Sampling from a normal distribution, given means and log of variances, represeting latent factors.
        """
        _ = args
        _ = kwargs
        mean, log_var = inputs
        batch_dim = tf.shape(mean)[0]
        latent_dim = tf.shape(mean)[1]
        sample = tf.keras.backend.random_normal(shape=(batch_dim, latent_dim))
        return mean + (tf.exp(0.5 * log_var)) * sample


class VariationalAutoEncoder(tf.keras.models.Model):
    """Class that implements a Variational AutoEncoder, on top of keras framework.

    Parameters:
        input_dim (int): Input dimension for the model.
        latent_space_dimension (int): Dimension of latent space.
        encoder_hidden_layers (Optional[List[int]], optional): List with units for each layer to be added to the encoder. Defaults to None.
    """

    def __init__(
        self,
        input_dim: int,
        latent_space_dimension: int,
        encoder_hidden_layers: Optional[List[int]] = None,
    ) -> None:
        """Constructor method for VariationalAutoEncoder.

        Args:
            input_dim (int): Input dimension for the model.
            latent_space_dimension (int): Dimension of latent space.
            encoder_hidden_layers (Optional[List[int]], optional): List with units for each layer to be added to the encoder. Defaults to None.
        """
        super(VariationalAutoEncoder, self).__init__()
        self._input_dim = input_dim
        self._latent_space_dimension = latent_space_dimension
        self._encoder_hidden_layers = encoder_hidden_layers
        self._encoder = self._get_encoder()
        self._decoder = self._get_decoder()

    def _get_encoder(self) -> tf.keras.models.Model:
        """Method that builds the encoder.

        Returns:
            tf.keras.models.Model: Encoder model.
        """
        x = tf.keras.layers.Input(shape=(self._input_dim,))
        inputs = x
        for layer in self._encoder_hidden_layers:
            x = tf.keras.layers.Dense(layer, activation="relu")(x)
        mean_vector = tf.keras.layers.Dense(
            self._latent_space_dimension, activation="relu"
        )(x)
        log_var_vector = tf.keras.layers.Dense(
            self._latent_space_dimension, activation="relu"
        )(x)
        Z = SamplingLayer()([mean_vector, log_var_vector])
        return tf.keras.models.Model(
            inputs=inputs, outputs=[Z, mean_vector, log_var_vector]
        )

    def _get_decoder(self) -> tf.keras.models.Model:
        """Method that builds the decoder.

        Returns:
            tf.keras.models.Model: Decoder model.
        """
        x = tf.keras.layers.Input(shape=(self._latent_space_dimension,))
        inputs = x
        for layer in self._encoder_hidden_layers[::-1]:
            x = tf.keras.layers.Dense(layer, activation="relu")(x)
        output = tf.keras.layers.Dense(self._input_dim, activation="sigmoid")(x)
        return tf.keras.models.Model(inputs=inputs, outputs=output)

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Method to perform the forward pass on the variational autoencoder.

        Args:
            inputs (tf.Tensor): Tensor of one-hot-encoded user vectors.

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Reconstructed input, (latent factors' mean vector and latent factors' log(Var) vector.
        """
        Z, mean_vector, log_var_vector = self._encoder(inputs)
        reconstruction = self._decoder(Z)
        return reconstruction, mean_vector, log_var_vector


class ReconstructionKLLoss(tf.keras.losses.Loss):
    """Loss function on top of keras framework to compute the needed loss for variational autoencoders.
    Basically, we add KL Divergence to the usual reconstruction loss.

    Parameters:
        binary (bool, optional): If true, Binary Cross Entropy is used as reconstruction loss - MSE, otherwise. Defaults to True.
    """

    def __init__(self, binary: bool = True) -> None:
        """Constructor method for ReconstructionKLLoss.

        Args:
            binary (bool, optional): If true, Binary Cross Entropy is used as reconstruction loss - MSE, otherwise. Defaults to True.
        """
        super(ReconstructionKLLoss, self).__init__()
        self._binary = binary
        if self._binary:
            self._reconstruction_loss = tf.keras.losses.BinaryCrossentropy()
        else:
            self._reconstruction_loss = tf.keras.losses.MeanSquaredError()

    def call(
        self, y_true: tf.Tensor, y_pred: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]
    ) -> tf.Tensor:
        """Method to compute the loss.

        Args:
            y_true (tf.Tensor): Tensor containing target values.
            y_pred (Tuple[tf.Tensor, tf.Tensor, tf.Tensor]): Reconstructed input, (latent factors' mean vector and latent factors' log(Var) vector.

        Returns:
            tf.Tensor: Loss value.
        """
        reconstruction, mean_vector, log_var_vector = y_pred
        reconstruction_loss = self._reconstruction_loss(y_true, reconstruction)
        kl = -0.5 * (
            1 + log_var_vector - tf.square(mean_vector) - tf.exp(log_var_vector)
        )
        return reconstruction_loss + tf.reduce_mean(tf.reduce_sum(kl, axis=1))
