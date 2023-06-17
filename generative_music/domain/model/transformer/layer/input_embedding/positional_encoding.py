"""Positional encoding for the Transformer model.

This module implements a custom Keras layer that adds positional encoding
to the input embeddings of the Transformer model.
Positional encoding is used to provide information about the position
of tokens in the sequence to the model.

By adding the positional encoding to the input embeddings,
the Transformer model can better understand the structure of the input sequence
and generalize across different input sequences.
"""
import math

import numpy as np
import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
    """This class is a custom Keras layer that implements positional encoding.

    Positional encoding is used to inject information about the position of tokens
    in the sequence into the model.
    It uses sinusoidal functions to create unique encodings for each position,
    which are then added to the input embeddings.
    The sinusoidal functions have different frequencies,
    making it easier to capture both local and global relationships between tokens.

    The PositionalEncoding layer computes the positional encodings
    and adds them to the input embeddings.
    """

    def __init__(self, embedding_dim: int, max_len: int = 5000):
        """Initialize the Embedding class.

        Args:
            embedding_dim (int): The dimensionality of the embedding vectors.
            max_len (int, optional):
                The maximum length of the input sequences. Default is 5000.
        """
        super(PositionalEncoding, self).__init__()
        self.embedding_dim = embedding_dim

        positional_encoding = np.zeros((max_len, embedding_dim))
        position = np.arange(max_len)[:, np.newaxis]
        # In lower dimensions, div_term is larger (the divisor is smaller),
        # so the values of sin and cos change more easily with the difference in position.
        # This makes it easier to capture "local relationships".
        # In higher dimensions, the opposite is true:
        # the values of sin and cos change less with the difference in position.
        # This makes it easier to capture "global relationships".
        div_term = self._compute_div_term()
        # The unit of angle is radian.
        positional_encoding[:, 0::2] = np.sin(position * div_term)
        positional_encoding[:, 1::2] = np.cos(position * div_term)
        # Expand dimensions to match the batch size.
        positional_encoding = np.expand_dims(positional_encoding, 0)
        self.positional_encoding = tf.constant(positional_encoding, dtype=tf.float32)

    def _compute_div_term(self) -> np.ndarray:
        """Compute the division term for positional encoding.

        This function computes the division term used in the positional encoding formula.
        The division term is calculated as 1 / (10000 ** (2 * i / d_model)),
        where i is the dimension id in the d_model and d_model is the embedding dimension.
        To make the computation efficient, the following conversions are implicit:

        div_term = 1 / (10000 ** (2 * i / d_model)) = (10000 ** (-2 * i / d_model))
        log(div_term) =  log(10000 ** (-2 * i / d_model)) = -2 * i * log(10000) / d_model
        exp(log(div_term)) = div_term = exp(-2 * i * log(10000) / d_model)

        Returns:
            np.ndarray:
                The computed division term,
                which is a 1D array of shape (embedding_dim // 2,).
        """
        div_term = np.exp(
            np.arange(0, self.embedding_dim, 2)
            * -(math.log(10000.0) / self.embedding_dim)
        )
        return div_term

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Compute the positional encoding operation for the input tensor.

        This function adds the positional encoding to the input tensor.

        Args:
            x (tf.Tensor):
                Input tensor with shape (batch_size, seq_len, embedding_dim).

        Returns:
            tf.Tensor:
                The output tensor with shape (batch_size, seq_len, embedding_dim)
                after adding positional encoding.
        """
        x = x + self.positional_encoding[:, : tf.shape(x)[1], :]
        return x
