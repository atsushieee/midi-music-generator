"""Input Embedding layer for neural networks.

This custom Keras layer implements the input embedding,
which combines an embedding layer and positional encoding,
and is used as the first layer in a neural network to handle sequence data efficiently.
"""
from typing import Optional

import tensorflow as tf

from generative_music.domain.model.transformer.layer.input_embedding.embedding import \
    Embedding
from generative_music.domain.model.transformer.layer.input_embedding.positional_encoding import \
    PositionalEncoding


class InputEmbedding(tf.keras.layers.Layer):
    """An Input Embedding class that combines an embedding layer and positional encoding.

    This class is responsible for processing input tensors by passing them
    through an embedding layer and adding positional encoding.
    The dimensions of the positional encoding and the embedding must be the same.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dropout_rate: float = 0.1,
        max_len: int = 5000,
    ):
        """Initialize the InputEmbedding class.

        Args:
            vocab_size (int):
                The size of the number of unique words or tokens in the dataset
            embedding_dim (int):
                The dimensionality of the embedding vectors.
            dropout_rate (float, optional):
                The dropout rate to be applied after adding positional encoding.
                Default is 0.1.
            max_len (int, optional):
                The maximum length of the input sequences. Default is 5000.
        """
        super(InputEmbedding, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Compute the input embedding for the input tensor.

        The input tensor is passed through the embedding layer and positional encoding.
        The output tensor is then processed with dropout
        if the layer is being called during training.

        Args:
            x (tf.Tensor): Input tensor with shape (batch_size, seq_len).
            training (Optional[bool], optional): If the layer is being called during training.

        Returns:
            tf.Tensor:
                The output tensor with shape (batch_size, seq_len, embedding_dim)
                after applying the input embedding layer and positional encoding.
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x, training=training)
        return x
