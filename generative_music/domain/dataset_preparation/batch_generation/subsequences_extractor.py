"""A module for extracting subsequences from batch sequences."""
import tensorflow as tf


class SubsequencesExtractor:
    """A class for extracting subsequences from batch sequences.

    This class is responsible for breaking input sequences into smaller subsequences,
    and padding them with the specified padding token as needed.
    """

    def __init__(
        self, subseq_length: tf.Tensor, padding_id: tf.Tensor, start_token_id: tf.Tensor
    ):
        """Initialize the SubsequencesExtractor instance.

        Args:
            subseq_length (tf.Tensor):
                A tensor containing the length of each subsequence to be extracted.
            padding_id (tf.Tensor):
                A tensor containing the token ID used for padding.
            start_token_id (tf.Tensor):
                A tensor containing the token ID used
                to indicate the start of a subsequence.
        """
        self.subseq_length = subseq_length
        self.padding_id = padding_id
        self.start_token_id = start_token_id

    @tf.function
    def extract_subsequences(self, sequences: tf.Tensor) -> tf.Tensor:
        """Extract subsequences of the given length from the input sequence.

        Args:
            sequences (tf.Tensor):
                The input sequence as a TensorFlow tensor with shape
                (batch_size, max_sequence_length).

        Returns:
            tf.Tensor: The extracted subsequences as a tensor of token IDs.
        """
        # Use tf.map_fn to apply the _extract_subsequence function
        # to each sequence in the batch.
        result = tf.map_fn(
            lambda seq: self._extract_subsequence(seq),
            sequences,
            dtype=tf.int32,
        )
        return result

    @tf.function
    def _extract_subsequence(self, sequence: tf.Tensor) -> tf.Tensor:
        """Extract a subsequence of the given length from the input sequence.

        If the desired subsequence length is longer than the length of the score,
        use the entire score.
        If it's shorter, truncate the song at the beginning of measures.

        Args:
            sequence (tf.Tensor):
                The input sequence as a TensorFlow tensor of token IDs.

        Returns:
            tf.Tensor: The extracted a subsequence as a tensor of token IDs.
        """
        # Find the indices of start_token_id in the sequence
        start_indices = tf.where(tf.equal(sequence, self.start_token_id))
        # Reshape the start_indices tensor to a 1D list
        start_indices = tf.reshape(start_indices, [-1])
        # Cast start_indices from tf.int64 to tf.int32
        start_indices = tf.cast(start_indices, dtype=tf.int32)

        tf.debugging.assert_greater(
            tf.size(start_indices),
            0,
            message="No valid start token found in the sequence.",
        )
        # Extract the subsequence from the first start index to the end of the sequence
        if self.subseq_length >= tf.shape(sequence)[0]:
            return self._pad_sequence(sequence[start_indices[0] :])

        actual_seq_length = self._find_actual_seq_length(sequence)
        start_index = self._select_start_index(start_indices, actual_seq_length)
        end_index = start_index + self.subseq_length
        # If it's in the middle of a measure, truncate it.
        no_padding_end_index = self._find_end_index(sequence, start_index)
        modified_sequence = self._replace_trailing_values_with_padding(
            sequence, no_padding_end_index
        )
        return modified_sequence[start_index:end_index]

    @tf.function
    def _pad_sequence(self, sequence: tf.Tensor) -> tf.Tensor:
        """Pad the input sequence to the target length with the padding token.

        Args:
            sequence (tf.Tensor):
                The input sequence as a TensorFlow tensor of token IDs.

        Returns:
            tf.Tensor: The padded sequence as a tensor.
        """
        padding = [[self.padding_id, self.subseq_length - tf.shape(sequence)[0]]]
        padded_sequence = tf.pad(sequence, padding)
        return padded_sequence

    @tf.function
    def _find_actual_seq_length(self, sequence: tf.Tensor) -> tf.Tensor:
        """Find the actual length of the input sequence, considering padding tokens.

        Args:
            sequence (tf.Tensor):
                The input sequence as a TensorFlow tensor of token IDs.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]:
                A tuple containing the actual sequence length.
        """
        # If there are any pad_indices, use the first one as the end of the sequence;
        # otherwise, use the original sequence length
        pad_indices = tf.where(tf.equal(sequence, self.padding_id))
        # Reshape the start_indices tensor to a 1D list
        pad_indices = tf.reshape(pad_indices, [-1])
        # Cast start_indices from tf.int64 to tf.int32
        pad_indices = tf.cast(pad_indices, dtype=tf.int32)
        actual_seq_length = tf.cond(
            tf.size(pad_indices) > 0,
            lambda: pad_indices[0],
            lambda: tf.shape(sequence)[0],
        )
        return actual_seq_length

    @tf.function
    def _select_start_index(
        self, start_indices: tf.Tensor, actual_seq_length: tf.Tensor
    ) -> tf.Tensor:
        """Select a random start index from the valid start indices.

        Args:
            start_indices (tf.Tensor):
                A 1D TensorFlow tensor containing the start indices.
            actual_seq_length (tf.Tensor):
                A scalar tensor representing the actual sequence length.

        Returns:
            tf.Tensor: A scalar tensor representing the selected start index.
        """
        valid_start_indices = tf.boolean_mask(
            start_indices,
            tf.greater_equal(actual_seq_length - start_indices, self.subseq_length),
        )
        if tf.size(valid_start_indices) > 0:
            start_index = tf.random.shuffle(valid_start_indices)[0]
        else:
            start_index = start_indices[0]
        return start_index

    @tf.function
    def _find_end_index(self, sequence: tf.Tensor, start_index: tf.Tensor) -> tf.Tensor:
        """Find the end index based on the given sequence and start index.

        The end index is determined with the following priority:
        1. If adding seq_length - 1 to the start_index results in an index
        that is greater than or equal to the last index of the sequence,
        the end index is set to the last index of the sequence.
        2. If the element in the sequence at the position (end_index + 1)
        is equal to the start_token_id, the end index remains unchanged.
        3. Check if there is any occurrence of start_token_id within the subsequence
        from (start_index + 1) to (end_index + 1). If not, return the current end_index.
        4. Otherwise, the end index is set to the last occurrence of start_token_id
        within the subsequence from (start_index + 1) to (end_index + 1).

        Args:
            sequence (tf.Tensor): The input sequence as a tensor.
            start_index (tf.Tensor): The starting index of the subsequence.

        Returns:
            tf.Tensor: The calculated end index.
        """
        end_index = start_index + self.subseq_length - 1
        if end_index >= tf.shape(sequence)[0] - 1:
            return tf.shape(sequence)[0] - 1
        else:
            next_token_is_start = tf.equal(sequence[end_index + 1], self.start_token_id)

        def use_last_matching_index() -> tf.Tensor:
            matching_indices = tf.where(
                tf.equal(sequence[start_index + 1 : end_index + 1], self.start_token_id)
            )
            if tf.size(matching_indices) == 0:
                return end_index
            last_matching_index = tf.reduce_max(matching_indices)
            # Cast start_indices from tf.int64 to tf.int32
            last_matching_index = tf.cast(last_matching_index, dtype=tf.int32)
            return start_index + last_matching_index

        # Update the end index based on whether the next token is the start token or not,
        # and if not, use the last matching index of the start token within the subsequence
        end_index = tf.cond(
            next_token_is_start, lambda: end_index, use_last_matching_index
        )
        return end_index

    @tf.function
    def _replace_trailing_values_with_padding(
        self, sequence: tf.Tensor, no_padding_end_index: tf.Tensor
    ) -> tf.Tensor:
        """Replace the values in the input sequence after the given index with padding tokens.

        Args:
            sequence (tf.Tensor):
                The input sequence as a 1D TensorFlow tensor.
            no_padding_end_index (tf.Tensor):
                A scalar tensor representing the index
                after which the padding should be added.

        Returns:
            tf.Tensor: A 1D tensor with trailing values replaced by padding tokens.
        """
        # Calculate the length of the padding part
        padding_length = tf.shape(sequence)[0] - no_padding_end_index - 1
        # Create a tensor of ones with the calculated padding length
        ones_tensor = tf.ones(padding_length, dtype=tf.int32)
        # Multiply the ones tensor by self.padding_id to create the padding_values tensor
        padding_values = ones_tensor * self.padding_id

        # Slice the input sequence up to no_padding_end_index + 1
        sliced_sequence = sequence[: no_padding_end_index + 1]
        # Concatenate the sliced sequence and padding_values to create the modified_sequence
        modified_sequence = tf.concat([sliced_sequence, padding_values], axis=0)

        # Return the modified_sequence with trailing values replaced by padding tokens
        return modified_sequence
