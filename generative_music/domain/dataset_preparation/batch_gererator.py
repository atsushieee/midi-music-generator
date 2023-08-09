"""A module for generating batches of sequences."""
from typing import List, Tuple

import tensorflow as tf


class BatchGenerator:
    """A class for generating batches of sequences from a dataset.

    This class creates batches of sequences with specified length,
    and shuffles and pads the training data as needed.
    """

    def __init__(
        self,
        data: List[List[int]],
        batch_size: int,
        seq_length: int,
        padding_id: int,
        start_token_id: int,
        vocab_size: int,
    ):
        """Initialize the BatchGenerator instance.

        Args:
            data (List[List[int]]): The input data as a list of tokenized sequences.
            batch_size (int): The number of sequences in a batch.
            seq_length (int): The length of each sequence in a batch.
            padding_id (int): The token ID used for padding.
            start_token_id (int): The token ID used to indicate the start of a sequence.
            vocab_size (int):
                The size of the number of unique words or tokens in the dataset.
        """
        self.data = data
        self.batch_size = batch_size
        self.subseq_length = tf.constant(seq_length, dtype=tf.int32)
        self.padding_id = tf.constant(padding_id, dtype=tf.int32)
        self.start_token_id = tf.constant(start_token_id, dtype=tf.int32)
        self.vocab_size = vocab_size
        self.look_ahead_mask = self._generate_look_ahead_mask()

    def generate_batches(self) -> tf.data.Dataset:
        """Generate batches of sequences from the input data.

        Returns:
            tf.data.Dataset:
                The dataset containing the generated batches, behaving like a generator.
        """
        dataset = tf.data.Dataset.from_generator(
            lambda: iter(self.data), output_types=tf.int32, output_shapes=(None,)
        )
        # Shuffle the dataset.
        # Adjust buffer_size based on the actual data size and machine specifications.
        dataset = dataset.shuffle(buffer_size=len(self.data))
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
        and the target sequence with all elements except the first one,
        which is then one-hot encoded.
        This is useful for language modeling tasks,
        where the goal is to predict the next token given a sequence of tokens.

        Args:
            sequences (tf.Tensor):
                The input sequence as a TensorFlow tensor with shape
                (batch_size, max_sequence_length).

        Returns:
            Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
                The source sequences as a TensorFlow tensor
                with shape (batch_size, max_sequence_length - 1),
                and the one-hot encoded target sequences as a TensorFlow tensor
                with shape (batch_size, max_sequence_length - 1, vocab_size),
                and the combined mask as a TensorFlow tensor
                with shape (batch_size, 1, seq_length, seq_length).
        """
        subsequences = self._extract_subsequences(sequences)
        src = subsequences[:, :-1]
        # One-hot encode tgt
        tgt = subsequences[:, 1:]
        tgt = tf.one_hot(tgt, depth=self.vocab_size, dtype=tf.int32)

        combined_mask = self._generate_combined_mask(src)
        return src, tgt, combined_mask

    @tf.function
    def _extract_subsequences(self, sequences: tf.Tensor) -> tf.Tensor:
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

    @tf.function
    def _generate_combined_mask(self, sequences: tf.Tensor) -> tf.Tensor:
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
