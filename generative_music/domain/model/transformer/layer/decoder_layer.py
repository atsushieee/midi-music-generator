"""DecoderLayer module for the Transformer model.

This module processes the multi-head attention and positionwise feed-forward.
"""
from typing import Optional

import tensorflow as tf

from generative_music.domain.model.transformer.layer.sublayer import (
    AddAndNorm, MultiHeadAttention, PositionwiseFeedForward)
from generative_music.domain.model.utils import ActivationFunctions


class DecoderLayer(tf.keras.layers.Layer):
    """This class is a custom Keras layer that implements a decoder layer in the Transformer.

    It processes the input tensor through multi-head attention followed by Add & Norm,
    and position-wise feed-forward networks followed by Add & Norm.

    This class is designed to be used multiple times
    as a building block in the Transformer decoder.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        epsilon: float = 1e-6,
        activation: ActivationFunctions = ActivationFunctions.GELU,
    ):
        """Initialize the DecoderLayer class.

        Args:
            d_model (int): The number of dimensions for the model.
            num_heads (int):
                The number of attention heads in the multi-head attention mechanism.
            ff_dim (int):
                The number of dimensions for the feed-forward network's hidden layer.
            dropout_rate (float, optional):
                The dropout rate to be applied. Default is 0.1.
            epsilon (float, optional):
                A small constant for numerical stability
                of the LayerNormalization. Default is 1e-6.
            activation (ActivationFunctions, optional):
                The activation function to be used in the first linear layer (w_1).
                Defaults to ActivationFunction.GELU, as it is adopted in GPT-2.
        """
        super(DecoderLayer, self).__init__()
        self.multi_head_attention_layer = MultiHeadAttention(
            num_heads, d_model, dropout_rate
        )
        self.ff_network_layer = PositionwiseFeedForward(
            d_model, ff_dim, dropout_rate, activation
        )

        self.mha_add_and_norm_layer = AddAndNorm(epsilon, dropout_rate)
        self.ffn_add_and_norm_layer = AddAndNorm(epsilon, dropout_rate)

    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None):
        """Process the input tensor through the decoder layer.

        Args:
            x (tf.Tensor):
                Query tensor with shape (batch_size, seq_len, d_model).
                In the first layer, the d_model is dimension of embedding,
                while in the subsequent layers, it is dimension of the output of previous one.
                However, due to the need for residual connections,
                the dimensions are essentially the same for all layers.
            mask (tf.Tensor, optional):
                Mask tensor with shape (1, 1, seq_len, seq_len).
                The first two dimensions (1, 1) are added to enable broadcasting
                with the attention logits tensor,
                where the first 1 is for batch_size and the second 1 is for num_heads.
                If provided, the part of attention scores will be masked.
                Defaults to None.

        Returns:
            tf.Tensor:
                The output tensor with shape (batch_size, seq_len, d_model)
                after processing through the decoder layer.
        """
        # Apply multi-head attention
        mha_output = self.multi_head_attention_layer(x, x, x, mask)
        mha_aan_out = self.mha_add_and_norm_layer(x, mha_output)
        # Apply position-wise feed forward network
        ffn_output = self.ff_network_layer(mha_aan_out)
        final_output = self.ffn_add_and_norm_layer(mha_aan_out, ffn_output)

        return final_output
