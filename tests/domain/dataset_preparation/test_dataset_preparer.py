"""Tests for a class that prepares dataset using dummy midi files."""
from pathlib import Path

from miditoolkit.midi import parser as midi_parser

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

    def _create_temp_midi_files(
        self, create_sample_midi: midi_parser.MidiFile, dir_path: Path, num_files: int
    ):
        """Creates temporary MIDI files for testing.

        Args:
            create_sample_midi (midi_parser.MidiFile):
                sample MidiFile object to be used in the tests.
            dir_path (Path):
                The directory path where the temporary MIDI files will be created.
            num_files (int): The number of temporary MIDI files to create.
        """
        for i in range(num_files):
            midi_obj = create_sample_midi
            filepath = dir_path / f"tmp_{i:03}.mid"
            midi_obj.dump(str(filepath))

    def test_prepare(self, create_sample_midi: midi_parser.MidiFile, tmp_path: Path):
        """Test if the prepare method correctly preprocesses and tokenizes the MIDI files.

        Args:
            create_sample_midi (midi_parser.MidiFile):
                sample MidiFile object to be used in the tests.
            tmp_path (Path): The temporary directory path provided by the pytest fixture.
        """
        # Create temporary MIDI files
        self._create_temp_midi_files(create_sample_midi, tmp_path, num_files=10)

        dataset_preparer = DatasetPreparer(
            tmp_path,
            self.midi_config,
            self.train_ratio,
            self.val_ratio,
            self.test_ratio,
        )

        tmp_csv_filepath = tmp_path / "all_splits.csv"
        train_data, val_data, test_data = dataset_preparer.prepare(tmp_csv_filepath)

        # Check if the data is preprocessed and tokenized correctly
        assert all(isinstance(data, list) for data in train_data)
        assert all(isinstance(data, list) for data in val_data)
        assert all(isinstance(data, list) for data in test_data)
        assert all(len(data) > 0 for data in train_data)
        assert all(len(data) > 0 for data in val_data)
        assert all(len(data) > 0 for data in test_data)
