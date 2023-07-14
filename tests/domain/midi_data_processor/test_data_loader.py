"""Tests for a class that loads MIDI tempo and note event data."""
import os
import tempfile

from miditoolkit.midi import parser as midi_parser

from generative_music.domain.midi_data_processor.data_elements import ItemName
from generative_music.domain.midi_data_processor.data_loader import DataLoader


def create_test_midi():
    """Create a test MIDI file with specified tempo changes and notes.

    Returns:
        str: The path of the temporary MIDI file created.
    """
    midi_obj = midi_parser.MidiFile()
    midi_obj.tempo_changes = [
        midi_parser.TempoChange(120, 0),
        midi_parser.TempoChange(100, 480),
        midi_parser.TempoChange(80, 720),
        midi_parser.TempoChange(120, 900),
        midi_parser.TempoChange(100, 1320),
        midi_parser.TempoChange(140, 1435),
    ]
    midi_obj.instruments.append(midi_parser.Instrument(0))
    midi_obj.instruments[0].notes = [
        midi_parser.Note(60, 62, 0, 480),
        midi_parser.Note(64, 66, 360, 960),
        midi_parser.Note(67, 69, 960, 1440),
        midi_parser.Note(71, 73, 1440, 1920),
        midi_parser.Note(60, 61, 1675, 1915),
    ]
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
    temp_file.close()
    # Dump MIDI data to the temporary file
    midi_obj.dump(temp_file.name)

    return temp_file.name


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
        self.data_loader = DataLoader(
            self.test_midi_path, note_resolution=240, tempo_resolution=240
        )

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
        assert len(note_items) == 5

        expected_note_items = [
            (0, 480, 60, 62),
            (240, 840, 64, 66),
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
        assert len(tempo_items) == 6

        expected_tempo_items = [
            (0, 120),
            (480, 100),
            (720, 80),
            (960, 120),
            (1200, 100),
            (1440, 140),
        ]

        for item, (expected_start, expected_tempo) in zip(
            tempo_items, expected_tempo_items
        ):
            assert item.name == ItemName.TEMPO
            assert item.start == expected_start
            assert item.tempo == expected_tempo
