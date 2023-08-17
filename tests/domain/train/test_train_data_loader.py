"""Tests for the Train Data Loader function."""
from pathlib import Path
from unittest.mock import Mock

from generative_music.domain.dataset_preparation.batch_generation import \
    BatchGenerator
from generative_music.domain.train.train_data_loader import TrainDataLoader
from generative_music.infrastructure.tfrecords import MidiTFRecordsReader


class TestTrainDataLoader:
    """A test class for the TrainDataLoader class.

    This class is responsible for testing the behavior and output of the _load_data method.
    """

    def test_load_data(self, tmp_path: Path, mocker: Mock):
        """Check the _load_data method behavior of the TrainDataLoader class.

        Tests if the method correctly calls the create_dataset method of MidiTFRecordsReader class
        and the generate_batches method of BatchGenerator class.
        Also checks if it returns the expected results.

        Args:
            tmp_path (Path): Temporary path for the test.
            mocker (Mock): Mock object for testing.
        """
        tfrecords_dir = tmp_path / "tfrecords"
        tfrecords_dir.mkdir()
        data_loader = TrainDataLoader(
            tfrecords_dir=tfrecords_dir,
            batch_size=32,
            seq_length=100,
            padding_id=0,
            bar_start_token_id=1,
            buffer_size=1000,
        )

        mock_dataset = mocker.MagicMock()
        mock_batches = mocker.MagicMock()
        # Mock the create_dataset method of MidiTFRecordsReader
        mock_data_reader = mocker.patch.object(
            MidiTFRecordsReader, "create_dataset", return_value=mock_dataset
        )
        # Mock the generate_batches method of BatchGenerator
        mock_batch_generator = mocker.patch.object(
            BatchGenerator, "generate_batches", return_value=mock_batches
        )
        filename = "train.tfrecords"
        data = data_loader._load_data(filename)
        mock_data_reader.assert_called_once_with(data_loader.tfrecords_dir / filename)
        mock_batch_generator.assert_called_once()
        assert data == mock_batches
