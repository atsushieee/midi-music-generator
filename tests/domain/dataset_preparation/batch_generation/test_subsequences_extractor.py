"""Tests for the SubsequencesExtractor used in sequences processing."""
from typing import List

import pytest
import tensorflow as tf

from generative_music.domain.dataset_preparation.batch_generation.subsequences_extractor import \
    SubsequencesExtractor


class TestSubsequencesExtractor:
    """A test class for the SubsequencesExtractor.

    The class tests if the subsequences are correctly extracted
    and padded as needed.
    """

    def setup_method(self):
        """Initialize the TestSubsequencesExtractor tests.

        Set up the input data, sequence length, padding ID, and start token ID.
        This method is called before each test function is executed.
        """
        self.data = [[1, 2, 3, 1], [1, 4, 1, 5, 1, 6, 7, 1, 8], [1, 9, 10, 11, 12, 13]]
        self.seq_length = tf.constant(4, dtype=tf.int32)
        self.padding_id = tf.constant(0, dtype=tf.int32)
        start_token_id = tf.constant(1, dtype=tf.int32)
        self.subsequence_extractor = SubsequencesExtractor(
            self.seq_length, self.padding_id, start_token_id
        )

    @pytest.mark.parametrize(
        "sequence_index, expected_subsequences",
        [
            (0, [tf.constant([1, 2, 3, 1], dtype=tf.int32)]),
            (
                1,
                [
                    tf.constant([1, 4, 1, 5], dtype=tf.int32),
                    tf.constant([1, 5, 0, 0], dtype=tf.int32),
                    tf.constant([1, 6, 7, 0], dtype=tf.int32),
                ],
            ),
            (2, [tf.constant([1, 9, 10, 11], dtype=tf.int32)]),
        ],
    )
    def test_extract_subsequence(
        self, sequence_index: int, expected_subsequences: List[tf.Tensor]
    ):
        """Check the extraction of subsequence.

        Tests if the the function correctly extracts a subsequence
        from the input sequence based on the start token index.

        Args:
            sequence_index (int): The index of the input sequence in the data.
            expected_subsequences (List[tf.Tensor]):
                A list of expected subsequences as tensors.
        """
        input_sequence = tf.constant(self.data[sequence_index], dtype=tf.int32)
        for eager_mode in [True, False]:
            # Enable/disable eager execution
            tf.config.run_functions_eagerly(eager_mode)
            extracted_subsequence = self.subsequence_extractor._extract_subsequence(
                input_sequence
            )
            # Check if extracted_subsequence matches any of the expected_subsequences
            match_found = False
            for expected_subsequence in expected_subsequences:
                try:
                    tf.debugging.assert_equal(
                        extracted_subsequence, expected_subsequence
                    )
                    match_found = True
                    break
                except tf.errors.InvalidArgumentError:
                    continue
            assert match_found

    def test_extract_subsequence_no_valid_start_token(self):
        """Check the error handling when no valid start token is found.

        Tests if the function raises an InvalidArgumentError
        when there is no valid start token in the input sequence.
        """
        sequence = tf.constant([0, 2, 3, 4, 5], dtype=tf.int32)
        for eager_mode in [True, False]:
            # Enable/disable eager execution
            tf.config.run_functions_eagerly(eager_mode)
            with pytest.raises(
                tf.errors.InvalidArgumentError,
                match="No valid start token found in the sequence.",
            ):
                self.subsequence_extractor._extract_subsequence(sequence)

    @pytest.mark.parametrize(
        "sequence",
        [
            tf.constant([1, 2, 3, 4], dtype=tf.int32),
            tf.constant([1, 2, 3], dtype=tf.int32),
        ],
    )
    def test_pad_sequence(self, sequence: tf.Tensor):
        """Check the sequence padding.

        Tests if the function correctly pads the input sequence
        to the specified sequence length with padding tokens.

        Args:
            sequence (tf.Tensor): The input sequence as a tensor.
        """
        for eager_mode in [True, False]:
            # Enable/disable eager execution
            tf.config.run_functions_eagerly(eager_mode)
            padded_sequence = self.subsequence_extractor._pad_sequence(sequence)
            assert tf.size(padded_sequence) == self.seq_length
            if tf.size(sequence) >= self.seq_length:
                tf.debugging.assert_equal(padded_sequence, sequence)
            else:
                # Check if the first part of the padded_sequence
                # is the same as the input sequence
                tf.debugging.assert_equal(
                    padded_sequence[: tf.size(sequence)], sequence
                )
                # Check if the remaining part of the padded_sequence
                # is filled with padding tokens
                tf.debugging.assert_equal(
                    padded_sequence[tf.size(sequence) :],
                    tf.fill([self.seq_length - tf.size(sequence)], self.padding_id),
                )

    def test_find_actual_seq_length(self):
        """Check the actual sequence length calculation.

        Tests if the function correctly calculates
        the actual sequence length by excluding padding tokens.
        """
        sequence = tf.constant([1, 2, 3, 0, 0], dtype=tf.int32)
        for eager_mode in [True, False]:
            # Enable/disable eager execution
            tf.config.run_functions_eagerly(eager_mode)
            actual_seq_length = self.subsequence_extractor._find_actual_seq_length(
                sequence
            )
            assert actual_seq_length == 3

    def test_select_start_index(self):
        """Check the start index selection.

        Tests if the function correctly selects the start index
        from the input start indices and actual sequence length.
        """
        start_indices = tf.constant([1], dtype=tf.int32)
        actual_seq_length = tf.constant(4, dtype=tf.int32)
        for eager_mode in [True, True]:
            # Enable/disable eager execution
            tf.config.run_functions_eagerly(eager_mode)
            start_index = self.subsequence_extractor._select_start_index(
                start_indices, actual_seq_length
            )
            assert start_index == 1

    @pytest.mark.parametrize(
        "sequence, start_index, expected_end_index",
        [
            (
                tf.constant([1, 2, 3, 4, 1, 5, 6, 7, 8], dtype=tf.int32),
                tf.constant(0, dtype=tf.int32),
                tf.constant(3, dtype=tf.int32),
            ),
            (
                tf.constant([1, 2, 3, 4, 1, 5, 6, 1, 7, 8], dtype=tf.int32),
                tf.constant(4, dtype=tf.int32),
                tf.constant(6, dtype=tf.int32),
            ),
            (
                tf.constant([1, 2, 3, 4, 1, 5, 6, 7, 8, 9], dtype=tf.int32),
                tf.constant(4, dtype=tf.int32),
                tf.constant(7, dtype=tf.int32),
            ),
        ],
    )
    def test_find_end_index(
        self, sequence: tf.Tensor, start_index: tf.Tensor, expected_end_index: tf.Tensor
    ):
        """Check the end index calculation.

        Tests if the the function correctly calculates the end index
        based on the input sequence and start index.

        Args:
            sequence (tf.Tensor): The input sequence as a tensor.
            start_index (tf.Tensor): The start index as a tensor.
            expected_end_index (tf.Tensor): The expected end index as a tensor.
        """
        for eager_mode in [True, False]:
            # Enable/disable eager execution
            tf.config.run_functions_eagerly(eager_mode)
            assert (
                self.subsequence_extractor._find_end_index(sequence, start_index)
                == expected_end_index
            )

    def test_replace_trailing_values_with_padding(self):
        """Check the replacement of trailing values with padding tokens.

        Tests if the function correctly replaces values in the input sequence
        after the specified index with padding tokens.
        """
        sequence = tf.constant([1, 2, 3, 4, 5], dtype=tf.int32)
        no_padding_end_index = 2
        expected_sequence = tf.constant([1, 2, 3, 0, 0], dtype=tf.int32)
        for eager_mode in [True, False]:
            # Enable/disable eager execution
            tf.config.run_functions_eagerly(eager_mode)
            modified_sequence = (
                self.subsequence_extractor._replace_trailing_values_with_padding(
                    sequence, no_padding_end_index
                )
            )
            assert tf.reduce_all(tf.equal(modified_sequence, expected_sequence))
