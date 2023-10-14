"""A module for generating augmented parameters for MIDI files."""
from typing import List


class DataAugmentedParamGenerator:
    """A class for data augmentation of MIDI files.

    This class provides methods for pitch shifting and time stretching.
    """

    def __init__(self, shift_range: List[int], stretch_range: List[float]):
        """Initialize the DataAugmentor instance.

        Args:
            shift_range (List[int]): The range of pitch shifts to apply.
            stretch_range (List[float]): The range of time stretches to apply.
        """
        self.shift_range = shift_range
        self.stretch_range = stretch_range

    def generate(self):
        """Generate combinations of pitch shifts and time stretches."""
        for shift in self.shift_range:
            for stretch in self.stretch_range:
                yield shift, stretch
