"""Multi-Head Attention module for the Transformer model.

This module consists of multiple parallel attention layers,
each with its own set of learnable parameters.
It is responsible for capturing different aspects of the input sequence's relationships,
both local and global, by processing the input sequence through multiple attention heads.

Each attention head computes the attention scores
and applies the attention mechanism independently,
allowing the model to focus on different parts of the input sequence simultaneously.
The outputs of these attention heads are then concatenated
and linearly transformed to produce the final output.

The Multi-Head Attention module enables the model to better understand complex relationships
within the input sequence and improve its ability
to generalize across different input sequences by capturing various aspects of the data.
"""
from typing import Optional

import tensorflow as tf

from generative_music.domain.model.transformer.layer.sublayer.multi_head_attention.attention import \
    Attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """This class is a custom Keras layer that implements the multi-headed attention.

    It splits the input query, key, and value into multiple heads,
    and computes the attention weights for each head independently.
    Then, it applies the attention weights to the value
    and concatenates the results from all heads to produce the attention output.

    The input dimensionality does not have to be the same for the first layer
    and the subsequent layers.
    This means that even if the dimensionality of the embedding layer output
    and the query, key, and value inputs are different,
    this class can be used without any issues.
    However, due to the need for residual connections,
    the input and output dimensions are essentially constant for all layers.
    This means that the dimensionality of the embedding layer output
    (referred to as d_model in the paper) is maintained throughout the model.
    """

    def __init__(
        self, num_heads: int, d_model: int, dropout_rate: Optional[float] = 0.1
    ):
        """Initialize the MultiHeadAttention class.

        Args:
            num_heads (int): The number of attention heads.
            d_model (int):
                The common dimensionality of the query, key, and value vectors.
                Note that this implementation assumes that the dimensions
                of query, key, value and output are the same.
                Ideally, d_model should be set during the build process,
                but since the input of the first layer comes from
                the output of the embedding layer
                and may have different dimensions, it is set during instantiation.
            dropout_rate (float, optional):
                The dropout rate to be applied to the attention weights.
                Defaults to 0.1.
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.dim_per_head = d_model // num_heads
        self.num_heads = num_heads
        self.linears = [tf.keras.layers.Dense(d_model) for _ in range(4)]
        self.attention = Attention(dropout_rate)

    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """Split the last dimension of the input tensor into (num_heads, dim_qkv).

        Args:
            x (tf.Tensor):
                The input tensor with shape (batch_size, seq_length, dim_qkv).
            batch_size (int): The number of batch size.

        Returns:
            tf.Tensor:
                A tensor with shape (batch_size, num_heads, seq_length, dim_per_head).
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.dim_per_head))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(
        self,
        query: tf.Tensor,
        key: tf.Tensor,
        value: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: Optional[bool] = None,
    ) -> tf.Tensor:
        """Compute multi-head attention for the given query, key, and value tensors.

        Args:
            query (tf.Tensor):
                Query tensor with shape (batch_size, seq_len, dim_embed or dim_q).
                In the first layer, the dimension is dim_embed,
                while in the subsequent layers, it is dim_q.
                However, due to the need for residual connections,
                the dimensions are essentially the same for all layers.(d_model)
            key (tf.Tensor):
                Key tensor with shape (batch_size, seq_len, dim_embed or dim_k).
                In the first layer, the dimension is dim_embed,
                while in the subsequent layers, it is dim_k.
                However, due to the need for residual connections,
                the dimensions are essentially the same for all layers.(d_model)
            value (tf.Tensor):
                Value tensor with shape (batch_size, seq_len, dim_embed or dim_v).
                In the first layer, the dimension is dim_embed,
                while in the subsequent layers, it is dim_v.
                However, due to the need for residual connections,
                the dimensions are essentially the same for all layers.(d_model)
            mask (tf.Tensor, optional):
                Mask tensor with shape (batch_size, 1, seq_len, seq_len).
                The second dimension (1) corresponds to the number of attention heads
                and is used for broadcasting with the attention logits tensor.
                If provided, the part of attention scores will be masked.
                Defaults to None.
            training (Optional[bool], optional):
                If the layer is being called during training.Defaults to None.

        Returns:
            tf.Tensor:
                The output tensor with shape (batch_size, seq_len, d_model)
                after applying multi-head attention.
        """
        batch_size = tf.shape(query)[0]

        query, key, value = [
            linear(x) for linear, x in zip(self.linears, (query, key, value))
        ]
        query, key, value = [
            self.split_heads(x, batch_size) for x in (query, key, value)
        ]

        scaled_attention, attention_weights = self.attention(
            query, key, value, mask, training
        )

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.num_heads * self.dim_per_head)
        )
        output = self.linears[-1](concat_attention)

        return output
