"""Attention module for the Transformer model.

This module calculates the attention scores and applies the attention mechanism.
The Multi-Head Attention and Self-Attention will be implemented in the layer.py module.
"""
from typing import Optional, Tuple

import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    """This class is a custom Keras layer that implements the attention mechanism.

    It takes three inputs, query, key, and value, computes the attention weights,
    and applies them to the value to produce the attention output.
    """

    def __init__(self, dropout_rate: Optional[float] = None, **kwargs):
        """Initialize the Attention class.

        Args:
            dropout_rate (float, optional):
                The dropout rate to be applied to the attention weights.
                Defaults to None, meaning no dropout is applied.
            **kwargs:
                Additional keyword arguments to be passed to the constructor
                of the parent class, tf.keras.layers.Layer.
        """
        super(Attention, self).__init__(**kwargs)
        self.dropout_rate = dropout_rate

    def build(self, input_shape: Tuple[int, ...]):
        """Build the Attention layer based on the input shape.

        Args:
            input_shape (Tuple[int, ...]):
                The shape of the input to the Attention layer.
        """
        if self.dropout_rate is not None:
            self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        super(Attention, self).build(input_shape)

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute Scaled Dot-Product Attention.

        Args:
            query (tf.Tensor):
                Query tensor with shape (batch_size, seq_len, depth_q).
            key (tf.Tensor):
                Key tensor with shape (batch_size, seq_len, depth_k).
            value (tf.Tensor):
                Value tensor with shape (batch_size, seq_len, depth_v).
            mask (tf.Tensor, optional):
                Mask tensor with shape (batch_size, seq_len, seq_len).
                If provided, the part of attention scores will be masked.
                Defaults to None.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]:
                The output tensor with shape (batch_size, seq_len, depth_v)
                and the attention weights tensor with shape
                (batch_size, seq_len, seq_len).
        """
        d_k = tf.cast(tf.shape(key)[-1], tf.float32)
        scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(d_k)

        if mask is not None:
            scores -= 1e9 * mask

        attention_weights = tf.nn.softmax(scores, axis=-1)

        if self.dropout_rate is not None:
            attention_weights = self.dropout(attention_weights)

        return tf.matmul(attention_weights, value), attention_weights
