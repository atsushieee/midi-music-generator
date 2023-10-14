"""Tests for a class that extracts chord information from note data."""
from generative_music.domain.midi_data_processor.midi_representation import (
    Config, Item, ItemName)
from generative_music.domain.midi_data_processor.preprocessor.chord_extractor import \
    ChordExtractor


class TestChordExtractor:
    """A test class for the ChordExtractor.

    The class is responsible for extracting chord information from note data.
    It checks if the note data is correctly processed
    and if the extracted chord data matches the expected values.
    """

    def setup_method(self):
        """Initialize the ChordExtractor tests.

        Set up the test environment by creating test notes
        and initializing the ChordExtractor instance.
        This method is called before each test function is executed.

        Note:
            Due to necessary adjustments that need to be made to the referred logic,
            not all chord patterns are covered in these tests.
        """
        self.test_note_items = [
            # C:maj (root: C, quality: maj)
            Item(name=ItemName.NOTE, start=0, end=1920, velocity=48, pitch=60),  # C
            Item(name=ItemName.NOTE, start=0, end=1920, velocity=48, pitch=64),  # E
            Item(name=ItemName.NOTE, start=0, end=1920, velocity=48, pitch=67),  # G
            # D:min (root: D, quality: min), half-bar length
            Item(name=ItemName.NOTE, start=1920, end=2880, velocity=48, pitch=62),  # D
            Item(name=ItemName.NOTE, start=1920, end=2880, velocity=48, pitch=65),  # F
            Item(name=ItemName.NOTE, start=1920, end=2880, velocity=48, pitch=69),  # A
            # G:dim (root: G, quality: dim), half-bar length
            Item(name=ItemName.NOTE, start=2880, end=3840, velocity=48, pitch=67),  # G
            Item(name=ItemName.NOTE, start=2880, end=3840, velocity=48, pitch=70),  # A#
            Item(name=ItemName.NOTE, start=2880, end=3840, velocity=48, pitch=73),  # C#
            # E:dom (root: E, quality: dom) with an outsider (-1), one-bar length
            Item(name=ItemName.NOTE, start=3840, end=5760, velocity=48, pitch=64),  # E
            Item(name=ItemName.NOTE, start=3840, end=5760, velocity=48, pitch=68),  # G#
            Item(name=ItemName.NOTE, start=3840, end=5760, velocity=48, pitch=71),  # B
            Item(name=ItemName.NOTE, start=3840, end=5760, velocity=48, pitch=74),  # D
            Item(name=ItemName.NOTE, start=3840, end=5760, velocity=48, pitch=78),  # F#
        ]
        self.chord_extractor = ChordExtractor(Config())

    def test_extract_chords(self):
        """Test if the chord items are correctly extracted and match the expected values.

        This test checks the number of chord items
        and the consistency between the expected chord items (start, end, pitch)
        and the actual chord items.
        """
        chord_items = self.chord_extractor.extract(self.test_note_items)
        assert len(chord_items) == 4

        expected_chord_items = [
            (0, 1920, "C:maj"),
            (1920, 2880, "D:min"),
            (2880, 3840, "G:dim"),
            (3840, 5760, "E:dom"),
        ]

        for item, (expected_start, expected_end, expected_pitch) in zip(
            chord_items, expected_chord_items
        ):
            assert item.name == ItemName.CHORD
            assert item.start == expected_start
            assert item.end == expected_end
            assert item.pitch == expected_pitch

    def test_transpose_items(self):
        """Test if the chord items are correctly transposed and match the expected values.

        This test checks the number of transposed chord items
        and the consistency between the expected transposed chord items
        (start, end, pitch) and the actual transposed chord items.
        """
        chord_items = self.chord_extractor.extract(self.test_note_items)
        transposed_chord_items = self.chord_extractor.transpose_items(chord_items, -2)
        expected_chord_items = [
            (0, 1920, "A#:maj"),
            (1920, 2880, "C:min"),
            (2880, 3840, "F:dim"),
            (3840, 5760, "D:dom"),
        ]
        for chord_item, (expected_start, expected_end, expected_pitch) in zip(
            transposed_chord_items, expected_chord_items
        ):
            assert chord_item.start == expected_start
            assert chord_item.end == expected_end
            assert chord_item.pitch == expected_pitch
