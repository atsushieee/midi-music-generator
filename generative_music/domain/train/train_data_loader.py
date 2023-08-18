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
        batch_size: int,
        seq_length: int,
        padding_id: int,
        bar_start_token_id: int,
        buffer_size: int,
        tfrecords_dir: Path,
        train_basename: str = "train",
        val_basename: str = "validation",
    ):
        """Initialize the DataLoader instance.

        Args:
            batch_size (int): The number of sequences in a batch.
            seq_length (int): The length of each sequence in a batch.
            padding_id (int): The token ID used for padding.
            bar_start_token_id (int):
                The token ID used to indicate
                the start of a new bar (musical measure) in the sequence.
            buffer_size (int): The size of the buffer used for shuffling the dataset.
            tfrecords_dir (Path): The directory containing the TensorFlow records.
            train_basename (str):
                The base name of the training file (without extension).
                Default is "train".
            val_basename (str):
                The base name of the validation file (without extension).
                Default is "validation".
        """
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.padding_id = padding_id
        self.bar_start_token_id = bar_start_token_id
        self.buffer_size = buffer_size
        self.tfrecords_dir = tfrecords_dir
        self.train_basename = train_basename
        self.val_basename = val_basename

    def load_train_data(self) -> tf.data.Dataset:
        """Load the training data from the TensorFlow records and generate batches.

        Returns:
            A tf.data.Dataset object representing the generated batches.
        """
        return self._load_data(f"{self.train_basename}.tfrecords")

    def load_val_data(self) -> tf.data.Dataset:
        """Load the validation data from the TensorFlow records and generate batches.

        Returns:
            A tf.data.Dataset object representing the generated batches.
        """
        return self._load_data(f"{self.val_basename}.tfrecords")

    def _load_data(self, filename: str) -> tf.data.Dataset:
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
