"""Dataset service for processing and writing MIDI data as TensorFlow records.

This module provides a service for tokenizing MIDI data,
splitting it into train, validation, and test sets,
and writing the tokenized data as TensorFlow records.
The DatasetService class can be used to prepare the dataset
for training and evaluation of neural network models.
"""
from pathlib import Path

import yaml

from generative_music.domain.dataset_preparation import DatasetPreparer
from generative_music.domain.midi_data_processor.midi_representation import \
    Config
from generative_music.infrastructure.tfrecords import MidiTFRecordsWriter


class DatasetService:
    """A service for processing and writing MIDI data as TensorFlow records.

    The purpose of this class is to provide a service for tokenizing MIDI data,
    splitting it into train, validation and test sets,
    and writing the tokenized data as TensorFlow records.
    """

    def __init__(
        self,
        data_dir: Path,
        midi_config: Config,
        csv_filepath: Path,
        tfrecords_dir: Path,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        train_basename: str = "train",
        val_basename: str = "validation",
        test_basename: str = "test",
    ):
        """Initialize the DatasetService instance.

        Args:
            data_dir (Path): The directory containing the MIDI files.
            midi_config (Config): The configuration object for MIDI processing.
            csv_filepath (Path):
                The path where the CSV file containing the split information will be saved.
            tfrecords_dir (Path):
                The path to the directory where the TensorFlow records will be written.
            train_ratio (float): The ratio of the dataset to be used for training.
            val_ratio (float): The ratio of the dataset to be used for validation.
            test_ratio (float): The ratio of the dataset to be used for testing.
            train_basename (str):
                The base name of the training file (without extension).
                Default is "train".
            val_basename (str):
                The base name of the validation file (without extension).
                Default is "validation".
            test_basename (str):
                The base name of the test file (without extension).
                Default is "test".
        """
        self.preparer = DatasetPreparer(
            data_dir,
            midi_config,
            csv_filepath,
            train_ratio,
            val_ratio,
            test_ratio,
            train_basename,
            val_basename,
            test_basename,
        )
        self.tf_writer = MidiTFRecordsWriter()
        self.tfrecords_dir = tfrecords_dir
        self.train_basename = train_basename
        self.val_basename = val_basename
        self.test_basename = test_basename

    def process_and_write_tfrecords(self):
        """Process the MIDI data and write it as TensorFlow records.

        Tokenize the MIDI data, split it into train, validation, and test sets,
        and write the tokenized data as TensorFlow records in the specified directory.
        """
        train_tokenized, val_tokenized, test_tokenized = self.preparer.prepare()
        self.tf_writer.write_tfrecords(
            train_tokenized, self.tfrecords_dir / f"{self.train_basename}.tfrecords"
        )
        self.tf_writer.write_tfrecords(
            val_tokenized, self.tfrecords_dir / f"{self.val_basename}.tfrecords"
        )
        self.tf_writer.write_tfrecords(
            test_tokenized, self.tfrecords_dir / f"{self.test_basename}.tfrecords"
        )


if __name__ == "__main__":
    # Load the config file
    with open("generative_music/config/dataset.yml", "r") as f:
        config = yaml.safe_load(f)
    # Extract values from the config
    tfrecords_dir = Path(config["paths"]["tfrecords_dir"])
    midi_data_dir = Path(config["paths"]["midi_data_dir"])
    csv_filepath = Path(config["paths"]["csv_filepath"])
    train_basename = config["dataset_basenames"]["train"]
    val_basename = config["dataset_basenames"]["val"]
    test_basename = config["dataset_basenames"]["test"]
    train_ratio = config["ratios"]["train_ratio"]
    val_ratio = config["ratios"]["val_ratio"]
    test_ratio = config["ratios"]["test_ratio"]

    midi_config = Config()

    dataset_service = DatasetService(
        midi_data_dir,
        midi_config,
        csv_filepath,
        tfrecords_dir,
        train_ratio,
        val_ratio,
        test_ratio,
        train_basename,
        val_basename,
        test_basename,
    )
    dataset_service.process_and_write_tfrecords()
