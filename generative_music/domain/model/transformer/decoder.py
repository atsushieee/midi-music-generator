"""Decoder module for the Transformer model.

This module processes the input embedding, decoder layers and generator layer.
"""
from typing import Optional

import tensorflow as tf

from generative_music.domain.model.transformer.layer import (DecoderLayer,
                                                             Generator,
                                                             InputEmbedding)


class Decoder(tf.keras.Model):
    """This class is a custom Keras model that implements a decoder block in the Transformer.

    It processes the input tensor through an input embedding layer,
    multiple decoder layers and a generator layer.
    This class is designed to be used as the main building block in the Transformer decoder
    and inherits from tf.keras.Model to enable model management features
    such as training, evaluation, prediction, and saving.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        vocab_size: int,
        maximum_seq_len: int,
        dropout_rate: float = 0.1,
        epsilon: float = 1e-6,
    ):
        """Initialize the Decoder class.

        Args:
            num_layers (int): The number of the decoder layers for the model.
            d_model (int): The number of dimensions for the model.
            num_heads (int):
                The number of attention heads in the multi-head attention mechanism.
            ff_dim (int):
                The number of dimensions for the feed-forward network's hidden layer.
            vocab_size (int):
                The size of the number of unique words or tokens in the dataset.
            maximum_seq_len (int):
                The maximum length of the input sequences.
            dropout_rate (float, optional):
                The dropout rate to be applied. Default is 0.1.
            epsilon (float, optional):
                A small constant for numerical stability
                of the LayerNormalization. Default is 1e-6.
        """
        super(Decoder, self).__init__()

        self.input_embedding_layer = InputEmbedding(
            vocab_size, d_model, dropout_rate, maximum_seq_len
        )
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, ff_dim, dropout_rate, epsilon)
            for _ in range(num_layers)
        ]
        self.generator_layer = Generator(vocab_size)

    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """Compute the decoder block for the input tensor.

        The input tensor is passed through the input embedding layer,
        decoder layers and generator layer.
        The mask tensor is used in the decoder layers for masking attention scores.

        Args:
            x (tf.Tensor): Input tensor with shape (batch_size, seq_len).
            mask (tf.Tensor, optional):
                Mask tensor with shape (1, 1, seq_len, seq_len).
                The first two dimensions (1, 1) are added to enable broadcasting
                with the attention logits tensor,
                where the first 1 is for batch_size and the second 1 is for num_heads.
                If provided, the part of attention scores will be masked.
                Defaults to None.

        Returns:
            tf.Tensor:
                The output tensor with shape (batch_size, seq_len, vocab_size)
                after applying the input embedding layer, decoder layers and generator layer.
        """
        x = self.input_embedding_layer(x)
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, mask)
        output = self.generator_layer(x)

        return output
