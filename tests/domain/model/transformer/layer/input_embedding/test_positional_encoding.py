"""Tests for the Positional encoding used in the Transformer model."""
import numpy as np
import pytest
import tensorflow as tf

from generative_music.domain.model.transformer.layer.input_embedding.positional_encoding import \
    PositionalEncoding


class TestPositionalEncoding:
    """A test class for the Positional encoding .

    The class tests if the input tensor is correctly processed through the
    positional encoding and if dropout is applied correctly.
    """

    @pytest.fixture(autouse=True)
    def init_module(self):
        """Initialize the Positional encoding tests.

        This fixture creates predefined embedding dimension, max sequence length,
        sample sequence length and dropout rate.
        After that, PositionalEncoding class is instantiated.
        """
        embedding_dim = 4
        max_seq_len = 10
        sample_seq_len = 5

        self.positional_encoding_layer = PositionalEncoding(embedding_dim, max_seq_len)
        self.input_tensor = tf.random.normal((1, sample_seq_len, embedding_dim))

    def test_positional_encoding_values(self):
        """Check the positional encoding tensor values.

        Tests if the positional encoding tensor calculated programmatically
        matches the manually calculated expected output tensor.
        """
        # calculation positional encoding tensor
        # with (batch_size, max_seq_len, embedding_dim) manually.
        expected_output_tensor = tf.constant(
            [
                [
                    [0.000000, 1.000000, 0.000000, 1.000000],
                    [0.841471, 0.540302, 0.0099998, 0.999950],
                    [0.909297, -0.416147, 0.019998, 0.999800],
                    [0.141120, -0.989992, 0.029995, 0.999550],
                    [-0.756802, -0.653644, 0.039989, 0.999200],
                    [-0.958924, 0.283662, 0.049979, 0.998750],
                    [-0.279415, 0.960170, 0.059964, 0.998200],
                    [0.656987, 0.753902, 0.069943, 0.997551],
                    [0.989358, -0.145500, 0.079915, 0.996802],
                    [0.412118, -0.911130, 0.089878, 0.995953],
                ]
            ],
            dtype=tf.float32,
        )

        assert np.allclose(
            self.positional_encoding_layer.positional_encoding.numpy(),
            expected_output_tensor.numpy(),
            atol=1e-6,
        )

    def test_input_output_shape(self):
        """Check the input and output tensor shapes.

        Tests if the input tensor and the output tensor have the same shape
        after processing through the positional encoding layer.
        """
        output_tensor = self.positional_encoding_layer(self.input_tensor)
        assert self.input_tensor.shape == output_tensor.shape
