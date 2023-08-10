"""A module for generating masks for sequences in a dataset."""
import tensorflow as tf


class MaskGenerator:
    """A class for generating masks for sequences in a dataset.

    This class creates masks with specified dimensions and properties,
    and applies them to the input sequences during batch generation.
    """

    def __init__(self, subseq_length: tf.Tensor, padding_id: tf.Tensor):
        """Initialize the BatchGenerator instance.

        Args:
            subseq_length (tf.Tensor):
                A tensor containing the length of each sequence in a batch.
            padding_id (tf.Tensor):
                A tensor containing the token ID used for padding.
        """
        self.subseq_length = subseq_length
        self.padding_id = padding_id
        self.look_ahead_mask = self._generate_look_ahead_mask()

    @tf.function
    def generate_combined_mask(self, sequences: tf.Tensor) -> tf.Tensor:
        """Generate the combined mask for the input batch.

        Args:
            sequences (tf.Tensor):
                The input batch of tokenized sequences with shape (batch_size, seq_length).

        Returns:
            tf.Tensor:
                The combined mask as a TensorFlow tensor
                with shape (batch_size, 1, seq_length, seq_length).
                The second dimension (1) corresponds to the number of attention heads
                and is used for broadcasting with the attention logits tensor.
                The combined mask includes both the padding mask and the look-ahead mask,
                which prevents the model from attending to future tokens in the sequences.
        """
        padding_mask = self._generate_padding_mask(sequences)
        combined_mask = tf.maximum(padding_mask, self.look_ahead_mask)
        return combined_mask

    def _generate_look_ahead_mask(self) -> tf.Tensor:
        """Generate the look-ahead mask for the source sequence.

        Returns:
            tf.Tensor:
                The look-ahead mask as a TensorFlow tensor with shape (seq_length, seq_length).
        """
        # Create a look-ahead mask with ones.
        look_ahead_mask = 1 - tf.linalg.band_part(
            tf.ones((self.subseq_length - 1, self.subseq_length - 1)), -1, 0
        )
        return look_ahead_mask

    @tf.function
    def _generate_padding_mask(self, sequences: tf.Tensor) -> tf.Tensor:
        """Generate the padding mask for the input batch.

        Args:
            sequences (tf.Tensor):
                The input batch of tokenized sequences with shape (batch_size, seq_length).

        Returns:
            tf.Tensor:
                The padding mask as a TensorFlow tensor
                with shape (batch_size, 1, 1, seq_length).
                The second and third dimensions (1, 1) are added
                to enable broadcasting with the attention logits tensor,
                where the second 1 is for num_heads and the third 1 is for seq_length.
                The padding mask is used to mask out the padding tokens
                in the input sequences.
        """
        padding_mask = tf.cast(tf.math.equal(sequences, self.padding_id), tf.float32)
        # Expand dimensions to match the shape required for the attention mechanism.
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        return padding_mask
