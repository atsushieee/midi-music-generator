"""Tests for the EpochStepsCalculator class that calculates total steps."""
from pathlib import Path

import pytest

from generative_music.domain.train.epoch_steps_calculator import \
    EpochStepsCalculator


class TestEpochStepsCalculator:
    """A test class for the EpochStepsCalculator.

    The class tests if the midi files are correctly counted,
    and if the total steps for training and validation are correctly calculated.
    """

    @pytest.fixture(autouse=True)
    def setup_module(self, create_temp_midi_files: Path):
        """Initialize the TestEpochStepsCalculator tests.

        Set up the test environment by creating test midi files
        and initializing the EpochStepsCalculator instance.
        Args:
            create_temp_midi_files (Path):
                A temporary directory path containing sample MidiFile objects
                to be used in the tests.
        """
        midi_data_dir = create_temp_midi_files
        train_ratio = 0.7
        val_ratio = 0.2
        batch_size = 2
        transpose_amounts = [-1, 0, 1]
        stretch_factors = [0.95, 1.0, 1.05]
        self.epoch_steps_calculator = EpochStepsCalculator(
            midi_data_dir,
            train_ratio,
            val_ratio,
            batch_size,
            transpose_amounts,
            stretch_factors,
        )

    def test_count_midi_files(self):
        """Test if the number of midi files is correctly counted.

        This test checks the number of midi files
        and verifies if it matches the expected value.
        """
        assert self.epoch_steps_calculator._count_midi_files() == 10

    def test_calculate_total_steps(self):
        """Test if the total steps are correctly calculated.

        This test checks the total steps
        and verifies if it matches the expected value.
        """
        assert self.epoch_steps_calculator._calculate_total_steps(0.8) == 36

    def test_train_total_steps(self):
        """Test if the total steps for training are correctly calculated.

        This test checks the total steps for training
        and verifies if it matches the expected value.
        """
        assert self.epoch_steps_calculator.train_total_steps == 32

    def test_val_total_steps(self):
        """Test if the total steps for validation are correctly calculated.

        This test checks the total steps for validation
        and verifies if it matches the expected value.
        """
        assert self.epoch_steps_calculator.val_total_steps == 1
