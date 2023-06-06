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

from generative_music.domain.model.transformer.layer.multi_head_attention.attention import \
    Attention


class MultiHeadedAttention(tf.keras.layers.Layer):
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
    """

    def __init__(self, num_heads: int, dim_qkv: int, attention: Attention):
        """Initialize the MultiHeadedAttention class.

        Args:
            num_heads (int): The number of attention heads.
            dim_qkv (int):
                The common dimensionality of the query, key, and value vectors.
                Note that this implementation assumes that the dimensions
                of query, key, and value are the same.
                Ideally, dim_qkv should be set during the build process,
                but since the input of the first layer comes from
                the output of the embedding layer
                and may have different dimensions, it is set during instantiation.
            attention (Attention): The attention mechanism to be used.
        """
        super(MultiHeadedAttention, self).__init__()
        assert dim_qkv % num_heads == 0
        self.dim_per_head = dim_qkv // num_heads
        self.num_heads = num_heads
        self.linears = [tf.keras.layers.Dense(dim_qkv) for _ in range(4)]
        self.attention = attention

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
    ) -> tf.Tensor:
        """Compute multi-head attention for the given query, key, and value tensors.

        Args:
            query (tf.Tensor):
                Query tensor with shape (batch_size, seq_len, dim_embed or dim_q).
                In the first layer, the dimension is dim_embed,
                while in the subsequent layers, it is dim_q.
            key (tf.Tensor):
                Key tensor with shape (batch_size, seq_len, dim_embed or dim_k).
                In the first layer, the dimension is dim_embed,
                while in the subsequent layers, it is dim_k.
            value (tf.Tensor):
                Value tensor with shape (batch_size, seq_len, dim_embed or dim_v).
                In the first layer, the dimension is dim_embed,
                while in the subsequent layers, it is dim_v.
            mask (tf.Tensor, optional):
                Mask tensor with shape (batch_size, seq_len, seq_len).
                If provided, the part of attention scores will be masked.
                Defaults to None.

        Returns:
            tf.Tensor:
                The output tensor with shape (batch_size, seq_len, dim_v)
                after applying multi-head attention.
        """
        batch_size = tf.shape(query)[0]

        query, key, value = [
            linear(x) for linear, x in zip(self.linears, (query, key, value))
        ]
        query, key, value = [
            self.split_heads(x, batch_size) for x in (query, key, value)
        ]

        scaled_attention, attention_weights = self.attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.num_heads * self.dim_per_head)
        )
        output = self.linears[-1](concat_attention)

        return output
