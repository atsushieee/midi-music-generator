"""Dataset service for processing and writing MIDI data as TensorFlow records.

This module provides a service for tokenizing MIDI data,
splitting it into train, validation, and test sets,
and writing the tokenized data as TensorFlow records.
The DatasetService class can be used to prepare the dataset
for training and evaluation of neural network models.
"""
from pathlib import Path

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
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ):
        """Initialize the DatasetService instance.

        Args:
            data_dir (Path): The directory containing the MIDI files.
            midi_config (Config): The configuration object for MIDI processing.
            train_ratio (float): The ratio of the dataset to be used for training.
            val_ratio (float): The ratio of the dataset to be used for validation.
            test_ratio (float): The ratio of the dataset to be used for testing.
        """
        self.preparer = DatasetPreparer(
            data_dir, midi_config, train_ratio, val_ratio, test_ratio
        )
        self.tf_writer = MidiTFRecordsWriter()

    def process_and_write_tfrecords(self, csv_filepath: Path, tfrecords_dir: Path):
        """Process the MIDI data and write it as TensorFlow records.

        Tokenize the MIDI data, split it into train, validation, and test sets,
        and write the tokenized data as TensorFlow records in the specified directory.

        Args:
            csv_filepath (Path):
                The path where the CSV file containing the split information will be saved.
            tfrecords_dir (Path):
                The path to the directory where the TensorFlow records will be written.
        """
        train_tokenized, val_tokenized, test_tokenized = self.preparer.prepare(
            csv_filepath
        )
        self.tf_writer.write_tfrecords(
            train_tokenized, tfrecords_dir / "train.tfrecords"
        )
        self.tf_writer.write_tfrecords(val_tokenized, tfrecords_dir / "val.tfrecords")
        self.tf_writer.write_tfrecords(test_tokenized, tfrecords_dir / "test.tfrecords")


if __name__ == "__main__":
    data_dir = Path("generative_music/data")
    midi_data_dir = data_dir / "midis"
    csv_filepath = data_dir / "dataset.csv"
    tfrecords_dir = data_dir / "tfrecords"
    midi_config = Config()
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    dataset_service = DatasetService(
        midi_data_dir, midi_config, train_ratio, val_ratio, test_ratio
    )
    dataset_service.process_and_write_tfrecords(csv_filepath, tfrecords_dir)
