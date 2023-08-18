"""A module for splitting a dataset into train, validation and test sets."""
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


class DatasetSplitter:
    """A class for splitting a dataset.

    The dataset is split into train, validation and test sets
    based on the specified ratios.
    """

    def __init__(
        self,
        data_dir: Path,
        csv_filepath: Path,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        train_basename: str = "train",
        val_basename: str = "validation",
        test_basename: str = "test",
    ):
        """Initialize the DatasetSplitter with the data directory and the split ratios.

        Args:
            data_dir (Path):
                The path to the data directory containing the files to be split.
            csv_filepath (Path):
                The path where the CSV file containing the split information will be saved.
            train_ratio (float):
                The ratio of the data to be used for the train set.
            val_ratio (float):
                The ratio of the data to be used for the validation set.
            test_ratio (float):
                The ratio of the data to be used for the test set.
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
        self.output_csv = csv_filepath
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.train_basename = train_basename
        self.val_basename = val_basename
        self.test_basename = test_basename
        self._check_directory_exists(data_dir)
        self.file_list = [
            f for f in data_dir.iterdir() if f.is_file() and self._is_midi_file(f)
        ]
        self._check_file_list_length()

    def split_data(self) -> Dict[str, List[str]]:
        """Split the data into train, validation and test sets based on the specified ratios.

        Returns:
            Dict[str, List[str]]:
                A dictionary containing the file paths for each split (train, val and test).
        """
        random.shuffle(self.file_list)
        num_total_files = len(self.file_list)
        # Recalculate input ratios
        total_ratio = self.train_ratio + self.val_ratio + self.test_ratio
        train_ratio = self.train_ratio / total_ratio
        val_ratio = self.val_ratio / total_ratio
        num_train_files = int(num_total_files * train_ratio)
        num_val_files = int(num_total_files * val_ratio)
        split_data = defaultdict(list)
        for i, file in enumerate(self.file_list):
            if i < num_train_files:
                split_data[self.train_basename].append(str(file))
            elif i < num_train_files + num_val_files:
                split_data[self.val_basename].append(str(file))
            else:
                split_data[self.test_basename].append(str(file))
        return split_data

    def create_split_csv(self, split_data: Dict[str, List[str]]):
        """Create a CSV file containing the file paths for each split.

        Args:
            split_data (Dict[str, List[str]]):
                A dictionary containing the file paths for each split.
        """
        with self.output_csv.open(mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["filepath", "split"])
            for split, files in split_data.items():
                for file in files:
                    writer.writerow([file, split])

    def _check_directory_exists(self, directory: Path):
        """Check if the specified directory exists.

        Args:
            directory (Path): The directory to check.

        Raises:
            ValueError: If the specified directory does not exist.
        """
        if not directory.exists():
            raise ValueError(f"The specified directory '{directory}' does not exist.")

    def _is_midi_file(self, file: Path) -> bool:
        """Check if the given file is a MIDI file.

        Args:
            file (Path): The file to be checked.

        Returns:
            bool: True if the file is a MIDI file, False otherwise.
        """
        return file.suffix.lower() in (".midi", ".mid")

    def _check_file_list_length(self):
        """Check if the file_list length is greater than 0, raise an error if not."""
        if len(self.file_list) <= 0:
            raise ValueError("No MIDI files found in the specified data directory.")
