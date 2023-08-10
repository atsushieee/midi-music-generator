"""Tests for the MaskGenerator used in sequence processing."""
import tensorflow as tf

from generative_music.domain.dataset_preparation.batch_generation.mask_generator import \
    MaskGenerator


class TestMaskGenerator:
    """A test class for the MaskGenerator.

    The class tests if the masks are correctly generated
    and applied to the input sequences.
    """

    def setup_method(self):
        """Initialize the TestMaskGenerator tests.

        Set up the sequence length and padding ID.
        This method is called before each test function is executed.
        """
        seq_length = tf.constant(4, dtype=tf.int32)
        padding_id = tf.constant(0, dtype=tf.int32)
        self.mask_generator = MaskGenerator(seq_length, padding_id)

    def test_generate_look_ahead_mask(self):
        """Check the generation of the look-ahead mask.

        Tests if the function correctly generates the look-ahead mask
        to prevent the model from attending to future tokens in the sequence.
        """
        expected_mask = tf.constant(
            [[0, 1, 1], [0, 0, 1], [0, 0, 0]],
            dtype=tf.float32,
        )
        assert tf.reduce_all(
            tf.equal(self.mask_generator.look_ahead_mask, expected_mask)
        )

    def test_generate_padding_mask(self):
        """Check the generation of padding mask.

        Tests if the function correctly generates the padding mask
        to mask out the padding tokens in the input sequences.
        """
        input_batch = tf.constant(
            [[1, 2, 3, 0], [4, 5, 6, 7]],
            dtype=tf.int32,
        )
        expected_padding_mask = tf.constant(
            [
                [[[0, 0, 0, 1]]],
                [[[0, 0, 0, 0]]],
            ],
            dtype=tf.float32,
        )
        for eager_mode in [True, False]:
            # Enable/disable eager execution
            tf.config.run_functions_eagerly(eager_mode)
            padding_mask = self.mask_generator._generate_padding_mask(input_batch)
            assert tf.reduce_all(tf.equal(padding_mask, expected_padding_mask))

    def test_generate_combined_mask(self):
        """Check the generation of combined mask.

        Tests if the function correctly generates the combined mask
        which includes both the padding mask and the look-ahead mask.
        """
        input_batch = tf.constant(
            [[1, 2, 0], [1, 0, 0]],
            dtype=tf.int32,
        )
        expected_combined_mask = tf.constant(
            [
                [
                    [
                        [0, 1, 1],
                        [0, 0, 1],
                        [0, 0, 1],
                    ]
                ],
                [
                    [
                        [0, 1, 1],
                        [0, 1, 1],
                        [0, 1, 1],
                    ]
                ],
            ],
            dtype=tf.float32,
        )
        for eager_mode in [True, False]:
            # Enable/disable eager execution
            tf.config.run_functions_eagerly(eager_mode)
            combined_mask = self.mask_generator.generate_combined_mask(input_batch)
            assert tf.reduce_all(tf.equal(combined_mask, expected_combined_mask))
