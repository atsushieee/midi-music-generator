"""Tests for the Batch Generator used in the Transformer model."""
import tensorflow as tf

from generative_music.domain.dataset_preparation.batch_generation.batch_gererator import \
    BatchGenerator


class TestBatchGenerator:
    """A test class for the BatchGenerator.

    The class tests if the input sequences are correctly preprocessed,
    divided into subsequences, and organized into batches,
    and if the generated batches have the correct shapes and data types.
    """

    def setup_method(self):
        """Initialize the TestBatchGenerator tests.

        Set up the data, batch size, sequence length, padding ID and start token ID.
        This method is called before each test function is executed.
        """
        data = [[1, 2, 3, 1], [1, 4, 1, 5, 1, 6, 7, 1, 8], [1, 9, 10, 11, 12, 13]]
        dataset = tf.data.Dataset.from_generator(
            lambda: iter(data),
            output_signature=tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )
        self.batch_size = 2
        self.seq_length = 4
        padding_id = 0
        bar_start_token_id = 1
        buffer_size = len(data)
        self.batch_generator = BatchGenerator(
            dataset,
            self.batch_size,
            self.seq_length,
            padding_id,
            bar_start_token_id,
            buffer_size,
        )

    def test_generate_batches(self):
        """Check the generated batch shapes and data types.

        Tests if the generated batches have the correct shapes(batch_size, seq_length - 1)
        and data types (tf.int32) after processing through the BatchGenerator.
        """
        dataset = self.batch_generator.generate_batches()
        for batch in dataset.take(1):
            src, tgt, mask = batch
            assert src.shape == (self.batch_size, self.seq_length - 1)
            assert tgt.shape == (self.batch_size, self.seq_length - 1)
            assert mask.shape == (
                self.batch_size,
                1,
                self.seq_length - 1,
                self.seq_length - 1,
            )
            assert src.dtype == tf.int32
            assert tgt.dtype == tf.int32
            assert mask.dtype == tf.float32

    def test_process_sequences(self):
        """Check the processed source and target sequences.

        Tests if the source and target sequences are correctly processed
        by removing the last token from the source and the first token from the target.
        """
        sequences = tf.constant([[1, 2, 3, 4]], dtype=tf.int32)
        for eager_mode in [True, False]:
            # Enable/disable eager execution
            tf.config.run_functions_eagerly(eager_mode)
            src, tgt, mask = self.batch_generator._process_sequences(sequences)
            assert tf.reduce_all(tf.equal(src, sequences[:, :-1]))
            assert tf.reduce_all(tf.equal(tgt, sequences[:, 1:]))
