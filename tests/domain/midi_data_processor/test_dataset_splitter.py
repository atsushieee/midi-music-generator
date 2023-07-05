"""Tests for the DatasetSplitter used for splitting data into train, val, and test sets."""
import csv
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from generative_music.domain.midi_data_processor.dataset_splitter import \
    DatasetSplitter


def create_test_files(num_files: int) -> Path:
    """Create a specified number of test files in a temporary directory.

    Args:
        num_files (int): The number of test files to create.

    Returns:
        Path: The path to the temporary directory containing the test files.
    """
    temp_dir = Path(tempfile.mkdtemp())
    for i in range(num_files):
        (temp_dir / f"file_{i}.mid").touch()
    return temp_dir


class TestDatasetSplitter:
    """A test class for the DatasetSplitter.

    The class tests if the data is correctly split into train, validation and test sets,
    and if the split files are correctly written to the CSV file.
    """

    def setup_method(self):
        """Initialize the DatasetSplitter tests.

        Set up the test environment by creating test files
        and initializing the DatasetSplitter instance.
        This method is called before each test function is executed.
        """
        self.num_files = 100
        self.data_dir = create_test_files(self.num_files)
        self.train_ratio = 0.7
        self.val_ratio = 0.2
        self.test_ratio = 0.1
        self.splitter = DatasetSplitter(
            self.data_dir, self.train_ratio, self.val_ratio, self.test_ratio
        )

    def teardown_method(self):
        """Clean up the test environment by removing the test data.

        This method is called after each test function is executed.
        """
        shutil.rmtree(self.data_dir)

    def test_is_midi_file(self):
        """Check if the _is_midi_file method can correctly identify MIDI files.

        This test checks the following cases:
        - .midi file
        - .mid file
        - non-MIDI file
        """
        # Test with a .midi file
        midi_file = Path("example.midi")
        assert self.splitter._is_midi_file(midi_file)

        # Test with a .mid file
        mid_file = Path("example.mid")
        assert self.splitter._is_midi_file(mid_file)

        # Test with a non-MIDI file
        non_midi_file = Path("example.txt")
        assert not self.splitter._is_midi_file(non_midi_file)

    def test_split_data(self):
        """Check if the data is correctly split into train, validation and test sets.

        This test checks the number of files in each set, the sum of files in all sets,
        and the consistency between the input ratios and the actual split ratios.
        """
        split_data = self.splitter.split_data()
        assert isinstance(split_data, defaultdict)
        assert set(split_data.keys()) == {"train", "validation", "test"}
        assert sum(len(files) for files in split_data.values()) == self.num_files
        assert len(split_data["train"]) == int(self.num_files * self.train_ratio)
        assert len(split_data["validation"]) == int(self.num_files * self.val_ratio)
        assert len(split_data["test"]) == self.num_files - len(
            split_data["train"]
        ) - len(split_data["validation"])

    def test_create_split_csv(self):
        """Check if the files are correctly split and written to the CSV file.

        This test compares the split_data and written_data dictionaries
        to check for consistency between the split files and the CSV content.
        """
        split_data = self.splitter.split_data()
        output_csv = Path(tempfile.mktemp(suffix=".csv"))
        self.splitter.create_split_csv(split_data, output_csv)
        assert output_csv.is_file()

        written_data = defaultdict(list)
        with output_csv.open(mode="r", newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            header = next(reader)
            assert header == ["filepath", "split"]
            for row in reader:
                filepath, split = row
                assert Path(filepath).is_file()
                assert split in {"train", "validation", "test"}
                written_data[split].append(filepath)

        for split in split_data:
            assert len(split_data[split]) == len(written_data[split])
            assert set(split_data[split]) == set(written_data[split])

        output_csv.unlink()
