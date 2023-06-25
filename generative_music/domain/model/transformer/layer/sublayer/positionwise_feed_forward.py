"""Positionwise feed-forward module for the Transformer model.

This module consists of two linear layers with a ReLU activation function
and dropout in between. It is responsible for processing the output
from the multi-head attention mechanism, followed by a residual connection.
This module helps in capturing local dependencies
and improving the model's ability to generalize across different input sequences.
"""
from typing import Optional

import tensorflow as tf

from generative_music.domain.model.utils import ActivationFunctions


class PositionwiseFeedForward(tf.keras.layers.Layer):
    """This class is a custom Keras layer that implements position-wise feed-forward.

    It consists of two linear layers with a ReLU activation function and dropout in between.
    The input and output dimensionality should be the same in a standard Transformer model,
    but this implementation allows for different input and output dimensionalities,
    providing flexibility for custom models and adaptations.
    """

    def __init__(
        self,
        output_dim: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        activation: ActivationFunctions = ActivationFunctions.GELU,
    ):
        """Initialize the PositionwiseFeedForward class.

        Args:
            output_dim (int):
                The output dimensionality of the feed-forward layer.
                In a standard Transformer model, this dimension should be equal
                to the input dimension due to the residual connections.
                Since the input of this layer comes from the output of the multi-head attention layer,
                which also uses residual connections,
                the output_dim should be the same as the embedding dimension.
            ff_dim (int):
                The dimensionality of the intermediate layer in the feed-forward network.
                This is the output dimension of the first linear layer (w_1)
                and the input dimension of the second linear layer (w_2).
            dropout_rate (float, optional):
                The dropout rate to be applied between the first and second linear layers.
                Defaults to 0.1.
            activation (ActivationFunctions, optional):
                The activation function to be used in the first linear layer (w_1).
                Defaults to ActivationFunction.GELU, as it is adopted in GPT-2.
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = tf.keras.layers.Dense(ff_dim, activation=activation.value)
        self.w_2 = tf.keras.layers.Dense(output_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Compute the position-wise feed-forward operation for the input tensor.

        Args:
            x (tf.Tensor):
                Input tensor with shape (batch_size, seq_len, dim_qkv).
            training (bool, optional):
                True if the model is in training mode, False if in inference mode.
                If `None`, the mode will be inferred from `self.training`.
                Default is None.

        Returns:
            tf.Tensor:
                The output tensor with shape (batch_size, seq_len, dim_qkv)
                after applying the position-wise feed-forward operation.
        """
        x = self.w_1(x)
        x = self.dropout(x, training=training)
        x = self.w_2(x)
        return x
