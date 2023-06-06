"""Tests for the Positionwise feed-forward layer used in the Transformer model."""
import pytest
import tensorflow as tf

from generative_music.domain.model.transformer.layer.positionwise_feed_forward import \
    PositionwiseFeedForward


class TestPositionwiseFeedForward:
    """A test class for the Positionwise feed-forward layer.

    The class tests if the input tensor is correctly processed through the
    position-wise feed-forward network and if dropout is applied correctly.
    """

    @pytest.fixture(autouse=True)
    def init_module(self):
        """Initialize the Positionwise feed-forward network layer tests.

        This fixture creates predefined input tensor and other parameters.
        After that, PositionwiseFeedForward class is instantiated.
        """
        self.batch_size = 32
        self.seq_len = 10
        self.output_dim = 64
        input_dim = 64
        ff_dim = 128
        dropout_rate = 0.5

        self.input_tensor = tf.random.uniform(
            (self.batch_size, self.seq_len, input_dim)
        )
        self.positional_feed_forward = PositionwiseFeedForward(
            self.output_dim, ff_dim, dropout_rate
        )

    def test_positionwise_feed_forward_network(self):
        """Check the Positionwise feed-forward network output shape.

        Tests if the Positionwise feed-forward network produces the expected output shape
        after processing the input tensor.
        """
        # Prepare test data
        output_tensor = self.positional_feed_forward(self.input_tensor)
        assert output_tensor.shape == (self.batch_size, self.seq_len, self.output_dim)

    def test_positionwise_feed_forward_dropout(self):
        """Check if the dropout is applied in the Positionwise feed-forward network.

        This test compares the output tensors in training and inference modes
        to check for differences.
        """
        # Prepare test data
        output_tensor_train = self.positional_feed_forward(
            self.input_tensor, training=True
        )
        output_tensor_inference = self.positional_feed_forward(
            self.input_tensor, training=False
        )

        # Check if dropout is applied during training
        dropout_difference = tf.reduce_sum(
            tf.abs(output_tensor_train - output_tensor_inference)
        )
        assert dropout_difference.numpy() > 0
