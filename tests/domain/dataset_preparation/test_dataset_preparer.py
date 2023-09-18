"""Tests for a class that prepares dataset using dummy midi files."""
from pathlib import Path

from generative_music.domain.dataset_preparation.dataset_preparer import \
    DatasetPreparer
from generative_music.domain.midi_data_processor.midi_representation import \
    Config


class TestDatasetPreparer:
    """A test class for the DatasetPreparer.

    The class tests if the MIDI files are correctly preprocessed and tokenized.
    """

    def setup_method(self):
        """Initialize the DatasetPreparer tests.

        Set up the MIDI configuration, train/val/test ratios for data splitting.
        This method is called before each test function is executed.
        """
        self.midi_config = Config()
        self.train_ratio = 0.7
        self.val_ratio = 0.2
        self.test_ratio = 0.1

    def test_prepare(self, create_temp_midi_files: Path):
        """Test if the prepare method correctly preprocesses and tokenizes the MIDI files.

        Args:
            create_temp_midi_files (Path):
                The temporary directory path containing　sample MidiFile object
                to be used in the tests.
        """
        tmp_path = create_temp_midi_files
        tmp_csv_filepath = tmp_path / "all_splits.csv"
        dataset_preparer = DatasetPreparer(
            tmp_path,
            self.midi_config,
            tmp_csv_filepath,
            self.train_ratio,
            self.val_ratio,
            self.test_ratio,
        )

        train_data, val_data, test_data = dataset_preparer.prepare()

        # Check if the data is preprocessed and tokenized correctly
        assert all(isinstance(data, list) for data in train_data)
        assert all(isinstance(data, list) for data in val_data)
        assert all(isinstance(data, list) for data in test_data)
        assert all(len(data) > 0 for data in train_data)
        assert all(len(data) > 0 for data in val_data)
        assert all(len(data) > 0 for data in test_data)
