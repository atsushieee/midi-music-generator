"""Tests for the Decoder used in the Transformer model."""
import tensorflow as tf

from generative_music.domain.model.transformer.decoder import Decoder


class TestDecoder:
    """A test class for the Decoder.

    The class tests if the output tensor has the expected shape.
    """

    def setup_method(self):
        """Initialize the Decoder tests.

        This fixture sets up the predefined parameters such as vocabulary size,
        embedding dimension, max sequence length, batch size, and dropout rate.
        It also instantiates the Decoder class with the specified parameters
        and prepares the input data and mask for testing.
        """
        num_layers = 2
        d_model = 64
        num_heads = 4
        ff_dim = 128
        self.vocab_size = 1000
        maximum_seq_len = 50
        self.decoder = Decoder(
            num_layers,
            d_model,
            num_heads,
            ff_dim,
            self.vocab_size,
            maximum_seq_len,
        )

        # Prepare the data
        self.batch_size = 32
        self.seq_length = 10
        self.input_data = tf.random.uniform(
            shape=(self.batch_size, self.seq_length),
            minval=0,
            maxval=self.vocab_size,
            dtype=tf.int32,
        )
        # Create the matrix for mask
        uncasted_mask = 1 - tf.linalg.band_part(
            tf.ones((self.seq_length, self.seq_length)), -1, 0
        )
        self.mask = uncasted_mask[tf.newaxis, tf.newaxis, :, :]

    def test_decoder_output_shape(self):
        """Check the output tensor shapes.

        Tests if the output tensor has the expected shape
        (batch_size, seq_length, vocab_size) after processing through the Decoder.
        """
        output = self.decoder(self.input_data, self.mask)
        assert output.shape == (self.batch_size, self.seq_length, self.vocab_size)
