"""A module for generating batches of sequences."""
from typing import Tuple

import tensorflow as tf

from generative_music.domain.dataset_preparation.batch_generation.mask_generator import \
    MaskGenerator
from generative_music.domain.dataset_preparation.batch_generation.subsequences_extractor import \
    SubsequencesExtractor


class BatchGenerator:
    """A class for generating batches of sequences from a dataset.

    This class creates batches of sequences with specified length,
    and shuffles and pads the training data as needed.
    """

    def __init__(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        seq_length: int,
        padding_id: int,
        bar_start_token_id: int,
        buffer_size: int,
    ):
        """Initialize the BatchGenerator instance.

        Args:
            dataset (tf.data.Dataset):
                A tf.data.Dataset object containing the tokenized MIDI list data.
            batch_size (int): The number of sequences in a batch.
            seq_length (int): The length of each sequence in a batch.
            padding_id (int): The token ID used for padding.
            bar_start_token_id (int):
                The token ID used to indicate
                the start of a new bar (musical measure) in the sequence.
            buffer_size (int): The size of the buffer used for shuffling the dataset.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.subseq_length = tf.constant(seq_length, dtype=tf.int32)
        self.padding_id = tf.constant(padding_id, dtype=tf.int32)
        self.bar_start_token_id = tf.constant(bar_start_token_id, dtype=tf.int32)
        self.buffer_size = buffer_size
        self.mask_generator = MaskGenerator(self.subseq_length, self.padding_id)
        self.subsequences_extractor = SubsequencesExtractor(
            self.subseq_length, self.padding_id, self.bar_start_token_id
        )

    def generate_batches(self) -> tf.data.Dataset:
        """Generate batches of sequences from the input data.

        Returns:
            tf.data.Dataset:
                The dataset containing the generated batches, behaving like a generator.
        """
        # Shuffle the dataset.
        # Adjust buffer_size based on the actual data size and machine specifications.
        dataset = self.dataset.shuffle(buffer_size=self.buffer_size)
        # When batching a dataset with variable shapes,
        # it is necessary to use Dataset.padded_batch.
        dataset = dataset.padded_batch(self.batch_size, padding_values=self.padding_id)
        dataset = dataset.map(
            self._process_sequences,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        # Prefetch optimizes the pipeline by overlapping data preprocessing and model training.
        # AUTOTUNE automatically determines the optimal buffer size, improving training speed.
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    @tf.function
    def _process_sequences(
        self, sequences: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Process sequences to create the source and target sequences.

        This function takes the input sequences and creates two new sequences:
        the source sequence with all elements except the last one,
        and the target sequence with all elements except the first one.
        This is useful for language modeling tasks,
        where the goal is to predict the next token given a sequence of tokens.

        Args:
            sequences (tf.Tensor):
                The input sequence as a TensorFlow tensor with shape
                (batch_size, max_sequence_length).

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
                - The source sequences with shape (batch_size, max_sequence_length - 1)
                - The target sequences with shape (batch_size, max_sequence_length - 1)
                - The combined mask with shape (batch_size, 1, seq_length, seq_length)
        """
        subsequences = self.subsequences_extractor.extract_subsequences(sequences)
        src = subsequences[:, :-1]
        tgt = subsequences[:, 1:]

        combined_mask = self.mask_generator.generate_combined_mask(src)
        return src, tgt, combined_mask
