"""TrainDataLoader for loading training data.

The purpose of this file is to provide a class for loading and preparing training data.
This class manages the process of loading data from TensorFlow records
and generating batches of data to be used in training.
"""
from pathlib import Path

import tensorflow as tf

from generative_music.domain.dataset_preparation.batch_generation import \
    BatchGenerator
from generative_music.infrastructure.tfrecords import MidiTFRecordsReader


class TrainDataLoader:
    """A utility for loading training and validation data from TensorFlow records.

    This class provides functionality to load training and validation data from TensorFlow records,
    and generate batches of sequences for model training.
    """

    def __init__(
        self,
        tfrecords_dir: Path,
        batch_size: int,
        seq_length: int,
        padding_id: int,
        bar_start_token_id: int,
        buffer_size: int,
    ):
        """Initialize the DataLoader instance.

        Args:
            tfrecords_dir (Path): The directory containing the TensorFlow records.
            batch_size (int): The number of sequences in a batch.
            seq_length (int): The length of each sequence in a batch.
            padding_id (int): The token ID used for padding.
            bar_start_token_id (int):
                The token ID used to indicate
                the start of a new bar (musical measure) in the sequence.
            buffer_size (int): The size of the buffer used for shuffling the dataset.
        """
        self.tfrecords_dir = tfrecords_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.padding_id = padding_id
        self.bar_start_token_id = bar_start_token_id
        self.buffer_size = buffer_size

    def load_train_data(self) -> tf.data.Dataset:
        """Load the training data from the TensorFlow records and generate batches.

        Returns:
            A tf.data.Dataset object representing the generated batches.
        """
        return self._load_data("train.tfrecords")

    def load_val_data(self) -> tf.data.Dataset:
        """Load the validation data from the TensorFlow records and generate batches.

        Returns:
            A tf.data.Dataset object representing the generated batches.
        """
        return self._load_data("val.tfrecords")

    def _load_data(self, filename) -> tf.data.Dataset:
        """Load the data from the TensorFlow records and generate batches.

        Args:
            filename (str): The name of the TensorFlow records file.

        Returns:
            A tf.data.Dataset object representing the generated batches.
        """
        # Instantiate the MidiTFRecordsReader
        tfrecords_reader = MidiTFRecordsReader()
        # Load the data from the TensorFlow records
        dataset = tfrecords_reader.create_dataset(self.tfrecords_dir / filename)
        # Instantiate the BatchGenerator for the data
        batch_generator = BatchGenerator(
            dataset,
            self.batch_size,
            self.seq_length,
            self.padding_id,
            self.bar_start_token_id,
            self.buffer_size,
        )
        # Generate the datasets
        data = batch_generator.generate_batches()
        return data
