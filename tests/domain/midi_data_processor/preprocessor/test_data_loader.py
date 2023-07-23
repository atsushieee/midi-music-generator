"""Tests for a class that loads MIDI tempo and note event data."""
import os

from generative_music.domain.midi_data_processor.midi_representation import (
    Config, ItemName)
from generative_music.domain.midi_data_processor.preprocessor.data_loader import \
    DataLoader
from tests.domain.midi_data_processor.preprocessor.conftest import \
    create_test_midi


class TestDataLoader:
    """A test class for the DataLoader.

    The class tests if the MIDI tempo and note data are correctly loaded
    and if the loaded data matches the expected values.
    """

    def setup_method(self):
        """Initialize the DataLoader tests.

        Set up the test environment by creating test midi file
        and initializing the DataLoader instance.
        This method is called before each test function is executed.
        """
        self.test_midi_path = create_test_midi()
        self.data_loader = DataLoader(self.test_midi_path, Config())

    def teardown_method(self):
        """Clean up the test environment by removing the test file.

        This method is called after each test function is executed.
        """
        os.remove(self.test_midi_path)

    def test_read_note_items(self):
        """Test if the note items are correctly read and match the expected values.

        This test checks the number of note item
        and the consistency between the expected note items
        (start, end, velocity, pitch) and the actual note items.
        """
        # Read note items and check if they match the expected values
        note_items = self.data_loader.read_note_items()

        expected_note_items = [
            (0, 480, 60, 62),
            (360, 959, 64, 66),
            (960, 1440, 67, 69),
            (1440, 1920, 71, 73),
            (1680, 1920, 60, 61),
        ]

        for item, (
            expected_start,
            expected_end,
            expected_velocity,
            expected_pitch,
        ) in zip(note_items, expected_note_items):
            assert item.name == ItemName.NOTE
            assert item.start == expected_start
            assert item.end == expected_end
            assert item.velocity == expected_velocity
            assert item.pitch == expected_pitch

    def test_read_tempo_items(self):
        """Test if the tempo items are correctly read and match the expected values.

        This test checks the number of tempo items
        and the consistency between the expected tempo items (start, tempo)
        and the actual tempo items.
        """
        # Read tempo items and check if they match the expected values
        tempo_items = self.data_loader.read_tempo_items()

        expected_tempo_items = [
            (0, 120),
            (480, 80),
            (960, 60),
            (1440, 60),
            (1920, 140),
        ]

        for item, (expected_start, expected_tempo) in zip(
            tempo_items, expected_tempo_items
        ):
            assert item.name == ItemName.TEMPO
            assert item.start == expected_start
            assert item.tempo == expected_tempo
