"""Tests for the Generator layer used in the Transformer model."""
import tensorflow as tf

from generative_music.domain.model.transformer.layer.generator import Generator


class TestGenerator:
    """A test class for the Generator layer.

    The class tests if the output tensor has the expected shape and
    if the softmax probabilities of the Generator's output sum to 1.
    """

    def setup_method(self):
        """Initialize the Generator layer tests.

        This fixture creates a predefined input tensor and other parameters.
        After that, the Generator class is instantiated.
        """
        self.vocab_size = 100
        self.generator = Generator(self.vocab_size)
        # Prepare test data
        d_model = 64
        self.batch_size = 16
        self.seq_length = 20
        self.input_tensor = tf.random.normal(
            (self.batch_size, self.seq_length, d_model)
        )

    def test_output_shape(self):
        """Check the output tensor shapes.

        Tests if the output tensor has the expected shape
        (batch_size, seq_length, vocab_size)
        after processing through the Generator layer.
        """
        output = self.generator(self.input_tensor)
        assert output.shape == (self.batch_size, self.seq_length, self.vocab_size)

    def test_softmax(self):
        """Check the softmax probabilities of the Generator's output.

        Tests if the sum of softmax probabilities for each position in the sequence
        is close to 1 after processing through the Generator layer.
        """
        output = self.generator(self.input_tensor)
        assert tf.reduce_all(
            tf.abs(
                tf.reduce_sum(output, axis=-1)
                - tf.ones((self.batch_size, self.seq_length))
            )
            < 1e-6
        )
