"""Scaled Embedding layer for neural networks.

This custom Keras layer implements the scaled embedding,
which is used as a first layer in a neural network to handle musical data efficiently.
"""
import tensorflow as tf


class Embedding(tf.keras.layers.Layer):
    """This class is a custom Keras layer that implements the scaled embedding.

    It transforms the input tensor into the embedding dimension
    and scales it by the square root of the embedding_dim.
    This is done to maintain the vector scaling in an appropriate range,
    improving learning stability (avoiding gradient vanishing) and efficiency.
    The Embedding class can be used as a first layer in a neural network
    to handle music data efficiently.
    """

    def __init__(self, vocab_size: int, embedding_dim: int):
        """Initialize the Embedding class.

        Args:
            vocab_size (int):
                The size of the number of unique words or tokens in the dataset
            embedding_dim (int):
                The dimensionality of the embedding vectors.
        """
        super(Embedding, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Compute the scaled embedding for the input tensor.

        The input tensor is transformed into the embedding dimension
        and then scaled by the square root of the embedding_dim.
        This is done to maintain the vector scaling in an appropriate range,
        improving learning stability (avoiding gradient vanishing) and efficiency.

        Args:
            x (tf.Tensor): Input tensor with shape (batch_size, seq_len).

        Returns:
            tf.Tensor:
                The output tensor with shape (batch_size, seq_len, embedding_dim)
                after applying the scaled embedding layer.
        """
        return self.embedding(x) * tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
