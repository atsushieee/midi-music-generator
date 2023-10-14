"""A module for calculating the number of steps per epoch for training and validation."""
import math
from pathlib import Path
from typing import List


class EpochStepsCalculator:
    """A class to calculate the number of steps per epoch for training and validation.

    This class counts the number of MIDI files in a directory
    and calculates the total steps needed for each epoch of training and validation
    based on the provided ratios.
    """

    def __init__(
        self,
        midi_data_dir: Path,
        train_ratio: float,
        val_ratio: float,
        batch_size: int,
        transpose_amounts: List[int] = [0],
        stretch_factors: List[float] = [1.0],
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
            transpose_amounts (List[int], optional):
                A list of integer values to shift the pitch
                of the MIDI files for data augmentation.
                Each integer represents the number of semitones to shift.
                Default is [0], meaning no shift.
            stretch_factors (List[float], optional):
                A list of float values to stretch or shrink the tempo
                of the MIDI files for data augmentation.
                Each float represents the factor by which to stretch the tempo.
                Default is [1.0], meaning no change in tempo.
        """
        self.midi_data_dir = midi_data_dir
        self.batch_size = batch_size
        self.augmentation_factors = len(transpose_amounts) * len(stretch_factors)
        self.midi_file_count = self._count_midi_files()
        self.train_total_steps = self._calculate_total_steps(train_ratio)
        self.val_total_steps = self._calculate_total_steps(val_ratio, False)

    def _count_midi_files(self) -> int:
        """Count the number of MIDI files in a directory.

        Returns:
            int: The number of MIDI files.
        """
        return len(list(self.midi_data_dir.glob("*.midi"))) + len(
            list(self.midi_data_dir.glob("*.mid"))
        )

    def _calculate_total_steps(self, ratio: float, is_training: bool = True) -> int:
        """Calculate the total steps for each epoch based on the provided ratio.

        Args:
            ratio (float): The ratio to calculate the total steps.
            is_training (bool, optional):
                A flag indicating whether the calculation is for the training set.
                If True, the augmentation factor is taken into account in the calculation.
                If False, the calculation assumes no augmentation.
                Default is True.

        Returns:
            int: The total steps for each epoch.
        """
        if is_training:
            samples = int(self.midi_file_count * ratio * self.augmentation_factors)
        else:
            samples = int(self.midi_file_count * ratio)
        total_steps = math.ceil(samples / self.batch_size)
        return total_steps
