"""The final part (Linear and Softmax) of the Decoder module for the Transformer model.

This custom Keras layer implements the Linear and Softmax layers
as the final part of the Decoder in the Transformer model.
It processes the input tensor through a position-wise feed-forward network
and applies the softmax function.
"""
import tensorflow as tf


class Generator(tf.keras.layers.Layer):
    """The final part (Linear and Softmax) of the Decoder in the Transformer model.

    This class is a custom Keras layer that implements the Linear and Softmax layers
    as the final part of the Decoder in the Transformer model.
    It processes the input tensor through a dense network
    and applies the softmax function.
    """

    def __init__(self, vocab_size: int):
        """Initialize the Generator class.

        Args:
            vocab_size (int):
                The size of the vocabulary,
                which determines the output size of the Linear layer.
        """
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.linear = tf.keras.layers.Dense(vocab_size)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Compute the Generator layer for the input tensors.

        This function processes the input tensor through a dense network
        and applies the softmax function.

        Args:
            x (tf.Tensor):
                The input tensor with shape (batch_size, seq_len, d_model).

        Returns:
            tf.Tensor:
                The output tensor with shape (batch_size, seq_len, vocab_size)
                after applying the Linear layer and the softmax function.
        """
        x = self.linear(x)
        x = self.softmax(x)
        return x
