"""Tests for the DataAugmentedParamGenerator class that generates data augmented params."""
from generative_music.domain.dataset_preparation.data_augmented_param_generator import \
    DataAugmentedParamGenerator


class TestDataAugmentedParamGenerator:
    """A test class for the DataAugmentedParamGenerator.

    The class tests if the pitch shifts and time stretches are correctly initialized,
    and if the combinations of them are correctly generated.
    """

    def test_generate(self):
        """Test if combinations of pitch shifts and time stretches are correctly generated.

        This test checks the combinations of pitch shifts and time stretches
        and verifies if they match the expected values.
        """
        shift_range = [-1, 0, 1]
        stretch_range = [0.9, 1.0, 1.1]

        generator = DataAugmentedParamGenerator(shift_range, stretch_range)

        expected_results = [
            (-1, 0.9),
            (-1, 1.0),
            (-1, 1.1),
            (0, 0.9),
            (0, 1.0),
            (0, 1.1),
            (1, 0.9),
            (1, 1.0),
            (1, 1.1),
        ]
        for expected, actual in zip(expected_results, generator.generate()):
            assert expected == actual
