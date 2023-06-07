"""Tests for the scaled embedding layer used in the Transformer model."""
import pytest
import tensorflow as tf

from generative_music.domain.model.transformer.layer.embedding import Embedding


class TestEmbedding:
    """A test class for the scaled embedding layer.

    The class tests if the input tensor is correctly processed through the
    position-wise feed-forward network and if dropout is applied correctly.
    """

    @pytest.fixture(autouse=True)
    def init_module(self):
        """Initialize the scaled embedding layer tests.

        This fixture creates predefined vocabulary size and embedding dimension.
        After that, Embedding class is instantiated.
        """
        self.vocab_size = 1024
        self.embedding_dim = 64
        self.embedding_layer = Embedding(self.vocab_size, self.embedding_dim)

    def test_embedding_size(self):
        """Check the Embedding layer input and output dimension."""
        assert self.embedding_layer.embedding.input_dim == self.vocab_size
        assert self.embedding_layer.embedding.output_dim == self.embedding_dim

    def test_root_embedding_dim_scaling(self):
        """Check the output scaling by the square root of embedding dimension."""
        # Prepare test data with shape (batch_size, seq_len)
        input_tensor = tf.constant([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
        output_tensor = self.embedding_layer(input_tensor)

        # Check if the output tensor has the same size as the input tensor
        assert output_tensor.shape == input_tensor.shape + (self.embedding_dim,)

        # Check if the output tensor is scaled by the square root of embedding dimension
        unscaled_embedding = self.embedding_layer.embedding(input_tensor)
        expected_output = unscaled_embedding * tf.math.sqrt(
            tf.cast(self.embedding_dim, tf.float32)
        )
        assert tf.reduce_all(tf.math.equal(output_tensor, expected_output))
