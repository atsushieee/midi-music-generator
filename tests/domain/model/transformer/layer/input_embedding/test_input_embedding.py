"""Tests for the Input Embedding used in the Transformer model."""
import numpy as np
import tensorflow as tf

from generative_music.domain.model.transformer.layer.input_embedding.input_embedding import \
    InputEmbedding


class TestInputEmbedding:
    """A test class for the Input Embedding.

    The class tests if the input tensor is correctly processed
    through the embedding layer and positional encoding.
    """

    def setup_method(self):
        """Initialize the Input Embedding tests.

        This fixture creates predefined vocabulary size, embedding dimension,
        max sequence length, sample sequence length, batch size and dropout rate.
        After that, InputEmbedding class is instantiated.
        """
        # Prepare test data
        vocab_size = 1024
        max_seq_len = 64
        self.embedding_dim = 64
        self.sample_seq_len = 16
        self.batch_size = 4
        self.dropout_rate = 0.5

        self.input_embedding_layer = InputEmbedding(
            vocab_size, self.embedding_dim, self.dropout_rate, max_seq_len
        )
        self.input_tensor = tf.random.uniform(
            shape=(self.batch_size, self.sample_seq_len),
            minval=0,
            maxval=vocab_size,
            dtype=tf.int32,
        )

    def test_output_shape(self):
        """Check the output tensor shapes.

        Tests if the output tensor has the expected shape
        (batch_size, seq_length, embedding_dim)
        after processing through the Input Embedding layer.
        """
        output_tensor = self.input_embedding_layer(self.input_tensor)
        assert output_tensor.shape == (
            self.batch_size,
            self.sample_seq_len,
            self.embedding_dim,
        )

    def test_dropout(self):
        """Check the dropout application in the output tensor.

        Tests if the output tensor with dropout applied has
        elements set to 0 where dropout is applied,
        and elements equal to the corresponding elements in the output tensor
        without dropout divided by (1 - dropout_rate) where dropout is not applied.
        """
        output_tensor_with_dropout = self.input_embedding_layer(
            self.input_tensor, training=True
        )
        output_tensor_without_dropout = self.input_embedding_layer(
            self.input_tensor, training=False
        )

        # Create a boolean mask for elements that are 0 in the output_tensor_with_dropout
        zero_mask = tf.equal(output_tensor_with_dropout, 0)

        scaled_elements = output_tensor_without_dropout / (1 - self.dropout_rate)
        # If zero_mask is True, set the element to 0
        # If zero_mask is False, set the element to the corresponding value in scaled_elements
        modified_output_without_dropout = tf.where(
            zero_mask, tf.zeros_like(output_tensor_without_dropout), scaled_elements
        )
        assert np.allclose(
            modified_output_without_dropout.numpy(),
            output_tensor_with_dropout.numpy(),
            atol=1e-6,
        )
