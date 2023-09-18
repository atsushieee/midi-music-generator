"""A module for calculating the number of steps per epoch for training and validation."""
import math
from pathlib import Path


class EpochStepsCalculator:
    """A class to calculate the number of steps per epoch for training and validation.

    This class counts the number of MIDI files in a directory
    and calculates the total steps needed for each epoch of training and validation
    based on the provided ratios.
    """

    def __init__(
        self, midi_data_dir: Path, train_ratio: float, val_ratio: float, batch_size: int
    ):
        """Initialize the EpochStepsCalculator.

        Args:
            midi_data_dir (Path):
                The path to the directory containing the MIDI files.
            train_ratio (float):
                The ratio of the data to be used for the train set.
            val_ratio (float):
                The ratio of the data to be used for the validation set.
            batch_size (int):
                The batch size for training.
        """
        self.midi_data_dir = midi_data_dir
        self.batch_size = batch_size
        self.midi_file_count = self._count_midi_files()
        self.train_total_steps = self._calculate_total_steps(train_ratio)
        self.val_total_steps = self._calculate_total_steps(val_ratio)

    def _count_midi_files(self) -> int:
        """Count the number of MIDI files in a directory.

        Returns:
            int: The number of MIDI files.
        """
        return len(list(self.midi_data_dir.glob("*.midi"))) + len(
            list(self.midi_data_dir.glob("*.mid"))
        )

    def _calculate_total_steps(self, ratio: float) -> int:
        """Calculate the total steps for each epoch based on the provided ratio.

        Args:
            ratio (float): The ratio to calculate the total steps.

        Returns:
            int: The total steps for each epoch.
        """
        samples = int(self.midi_file_count * ratio)
        total_steps = math.ceil(samples / self.batch_size)
        return total_steps
