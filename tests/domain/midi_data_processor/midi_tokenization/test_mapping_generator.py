"""Tests for a class that generates event2id and id2event mappings."""
import json
from pathlib import Path

from generative_music.domain.midi_data_processor.midi_representation import \
    Config
from generative_music.domain.midi_data_processor.midi_tokenization.mapping_generator import \
    MappingGenerator


class TestMappingGenerator:
    """A test class for the MappingGenerator.

    The class is responsible for testing the generation of data and reversed_data dictionaries,
    as well as their correct saving to files.
    It checks if the data and reversed_data dictionaries are created correctly,
    and if the saved data matches the expected values.
    """

    def setup_method(self):
        """Initialize the MappingGenerator tests.

        Set up the test environment by creating a MappingGenerator instance
        and generating the data and reversed_data dictionaries.
        This method is called before each test function is executed.
        """
        self.midi_config = Config()
        self.mapping_generator = MappingGenerator(self.midi_config)
        self.data = self.mapping_generator.data
        self.reversed_data = self.mapping_generator._reverse_data()

    def test_create_data(self):
        """Test if the generated data dictionary contains the expected keys."""
        assert isinstance(self.data, dict)
        assert "Bar_None" in self.data
        assert f"Position_1/{self.midi_config.DEFAULT_FRACTION}" in self.data
        assert "Chord_C:maj" in self.data
        assert "Tempo Class_fast" in self.data
        assert "Tempo Value_0" in self.data
        assert "Note On_0" in self.data
        assert "Note Velocity_0" in self.data
        assert "Note Duration_0" in self.data

    def test_save_data(self, tmp_path: Path):
        """Test if the save_data method correctly saves the data dictionary to a file.

        Args:
            tmp_path (Path): Temporary path for the test file.
        """
        filename = tmp_path / "event2id.json"
        self.mapping_generator._save_data(filename, self.data)
        with open(filename, "r") as infile:
            saved_data = json.load(infile)
        assert saved_data == self.data

    def test_reverse_data(self):
        """Test if the reverse_data method.

        This test checks that the method correctly generates a reversed dictionary
        with values as keys and keys as values.
        """
        for key, value in self.data.items():
            assert self.reversed_data[value] == key

    def test_save_reversed_data(self, tmp_path: Path):
        """Test if the save_data method correctly saves the reversed_data dictionary to a file.

        Args:
            tmp_path (Path): Temporary path for the test file.
        """
        filename = tmp_path / "id2event.json"
        self.mapping_generator._save_data(filename, self.reversed_data)
        with open(filename, "r") as infile:
            saved_data = json.load(infile)
        # Convert the keys of saved_data to integers to match the key types
        saved_data_int_keys = {int(key): value for key, value in saved_data.items()}
        assert saved_data_int_keys == self.reversed_data
