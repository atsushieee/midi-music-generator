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
    An optional mask input can also be provided to selectively ignore certain positions
    in the input sequences during the attention computation.
    """

    def __init__(self, dropout_rate: Optional[float] = 0.1):
        """Initialize the Attention class.

        Args:
            dropout_rate (float, optional):
                The dropout rate to be applied to the attention weights.
                Defaults to 0.1.
        """
        super(Attention, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: Optional[bool] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute Scaled Dot-Product Attention.

        Args:
            query (tf.Tensor):
                Query tensor with shape (batch_size, seq_len, dim_q)
                or (batch_size, num_heads, seq_len, dim_q / num_heads)
                for Multi-head Attention.
            key (tf.Tensor):
                Key tensor with shape (batch_size, seq_len, dim_k)
                or (batch_size, num_heads, seq_len, dim_k / num_heads)
                for Multi-head Attention.
            value (tf.Tensor):
                Value tensor with shape (batch_size, seq_len, dim_v)
                or (batch_size, num_heads, seq_len, dim_v / num_heads)
                for Multi-head Attention.
            mask (tf.Tensor, optional):
                Mask tensor with shape
                (1, seq_len, seq_len) when not using multi-head attention
                or (1, 1, seq_len, seq_len) when using multi-head attention.
                If provided, the part of attention scores will be masked.
                Defaults to None.
            training (bool, optional):
                True if the model is in training mode, False if in inference mode.
                If `None`, the mode will be inferred from `self.training`.
                Default is None.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]:
                The output tensor with shape (batch_size, seq_len, dim_v)
                or (batch_size, num_heads, seq_len, dim_v / num_heads)
                for Multi-head Attention,
                and the attention weights tensor with shape
                (batch_size, seq_len, seq_len).
        """
        d_k = tf.cast(tf.shape(key)[-1], tf.float32)
        scores = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(d_k)

        if mask is not None:
            scores -= 1e9 * mask

        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        return tf.matmul(attention_weights, value), attention_weights
