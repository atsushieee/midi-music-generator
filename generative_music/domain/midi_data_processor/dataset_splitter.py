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
        self, data_dir: Path, train_ratio: float, val_ratio: float, test_ratio: float
    ):
        """Initialize the DatasetSplitter with the data directory and the split ratios.

        Args:
            data_dir (Path):
                The path to the data directory containing the files to be split.
            train_ratio (float):
                The ratio of the data to be used for the train set.
            val_ratio (float):
                The ratio of the data to be used for the validation set.
            test_ratio (float):
                The ratio of the data to be used for the test set.
        """
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.file_list = [
            f for f in data_dir.iterdir() if f.is_file() and self._is_midi_file(f)
        ]

    def _is_midi_file(self, file: Path) -> bool:
        """Check if the given file is a MIDI file.

        Args:
            file (Path): The file to be checked.

        Returns:
            bool: True if the file is a MIDI file, False otherwise.
        """
        return file.suffix.lower() in (".midi", ".mid")

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
                split_data["train"].append(str(file))
            elif i < num_train_files + num_val_files:
                split_data["validation"].append(str(file))
            else:
                split_data["test"].append(str(file))
        return split_data

    def create_split_csv(self, split_data: Dict[str, List[str]], output_csv: Path):
        """Create a CSV file containing the file paths for each split.

        Args:
            split_data (Dict[str, List[str]]):
                A dictionary containing the file paths for each split.
            output_csv (Path): The path to the output CSV file.
        """
        with output_csv.open(mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["filepath", "split"])
            for split, files in split_data.items():
                for file in files:
                    writer.writerow([file, split])
