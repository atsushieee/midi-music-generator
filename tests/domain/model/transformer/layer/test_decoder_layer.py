"""Tests for the DecoderLayer used in the Transformer model."""
import tensorflow as tf

from generative_music.domain.model.transformer.layer.decoder_layer import \
    DecoderLayer


class TestDecoderLayer:
    """A test class for the DecoderLayer.

    The class tests if the output tensor has the expected shape.
    """

    def test_output_shape(self):
        """Check the output tensor shapes.

        Tests if the output tensor has the expected shape
        (batch_size, seq_length, d_model)
        after processing through the DecoderLayer.
        """
        # Prepare test data
        d_model = 128
        ff_dim = 512
        batch_size = 2
        num_heads = 4
        seq_length = 10
        input_tensor = tf.random.normal((batch_size, seq_length, d_model))
        # Create the matrix for mask
        uncasted_mask = 1 - tf.linalg.band_part(
            tf.ones((seq_length, seq_length)), -1, 0
        )
        mask = uncasted_mask[tf.newaxis, tf.newaxis, :, :]

        decoder_layer = DecoderLayer(d_model, num_heads, ff_dim)
        output = decoder_layer(input_tensor, mask)
        assert output.shape == (batch_size, seq_length, d_model)
