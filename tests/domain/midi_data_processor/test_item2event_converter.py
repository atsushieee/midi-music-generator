"""Tests for a class that converts item to event data."""
from generative_music.domain.midi_data_processor.data_elements import (
    EventName, Item, ItemName)
from generative_music.domain.midi_data_processor.item2event_converter import \
    Item2EventConverter


class TestItem2EventConverter:
    """A test class for the Item2EventConverter.

    The class is responsible for testing the conversion Items to Event instances.
    It checks if the MIDI items are correctly processed
    and if the resulting Event instances match the expected values.
    """

    def setup_method(self):
        """Initialize the Item2EventConverter tests.

        Set up the test environment by creating test items
        and initializing the Item2EventConverter instance.
        This method is called before each test function is executed.
        """
        self.test_items = [
            # Tempo
            Item(name=ItemName.TEMPO, start=0, tempo=120),
            Item(name=ItemName.TEMPO, start=1920, tempo=240),
            Item(name=ItemName.TEMPO, start=2400, tempo=25),
            Item(name=ItemName.TEMPO, start=2880, tempo=90),
            Item(name=ItemName.TEMPO, start=3840, tempo=150),
            # Chords
            Item(name=ItemName.CHORD, start=0, end=1920, pitch="C:maj"),
            Item(name=ItemName.CHORD, start=1920, end=2880, pitch="D:min"),
            Item(name=ItemName.CHORD, start=2880, end=3840, pitch="G:dim"),
            Item(name=ItemName.CHORD, start=3840, end=5760, pitch="E:dom"),
            # Notes
            Item(name=ItemName.NOTE, start=0, end=1920, velocity=43, pitch=60),  # C
            Item(name=ItemName.NOTE, start=0, end=1910, velocity=40, pitch=64),  # E
            Item(name=ItemName.NOTE, start=0, end=1960, velocity=48, pitch=67),  # G
            Item(name=ItemName.NOTE, start=1920, end=2880, velocity=65, pitch=62),  # D
            Item(name=ItemName.NOTE, start=3840, end=5760, velocity=48, pitch=64),  # E
            Item(name=ItemName.NOTE, start=3840, end=5790, velocity=48, pitch=68),  # G#
        ]
        max_time = self.test_items[-1].end
        self.item2event_converter = Item2EventConverter(self.test_items, max_time)

    def test_convert_items_to_events(self):
        """Test if the items are correctly converted to events and match the expected values.

        This test checks the number of events
        and the consistency between the expected events (name, time, value, text)
        and the actual events.
        """
        events = self.item2event_converter.convert_items_to_events()
        assert len(events) == 50
        expected_events = [
            (EventName.BAR, 0, None),
            (EventName.POSITION, 0, "1/16"),
            (EventName.TEMPO_CLASS, 0, "mid"),
            (EventName.TEMPO_VALUE, 0, 30),
            (EventName.POSITION, 0, "1/16"),
            (EventName.CHORD, 0, "C:maj"),
            (EventName.POSITION, 0, "1/16"),
            (EventName.NOTE_VELOCITY, 0, 10),
            (EventName.NOTE_ON, 0, 60),
            (EventName.NOTE_DURATION, 0, 31),
            (EventName.POSITION, 0, "1/16"),
            (EventName.NOTE_VELOCITY, 0, 10),
            (EventName.NOTE_ON, 0, 64),
            (EventName.NOTE_DURATION, 0, 31),
            (EventName.POSITION, 0, "1/16"),
            (EventName.NOTE_VELOCITY, 0, 12),
            (EventName.NOTE_ON, 0, 67),
            (EventName.NOTE_DURATION, 0, 32),
            (EventName.BAR, 1920, None),
            (EventName.POSITION, 1920, "1/16"),
            (EventName.TEMPO_CLASS, 1920, "fast"),
            (EventName.TEMPO_VALUE, 1920, 59),
            (EventName.POSITION, 1920, "1/16"),
            (EventName.CHORD, 1920, "D:min"),
            (EventName.POSITION, 1920, "1/16"),
            (EventName.NOTE_VELOCITY, 1920, 16),
            (EventName.NOTE_ON, 1920, 62),
            (EventName.NOTE_DURATION, 1920, 15),
            (EventName.POSITION, 2400, "5/16"),
            (EventName.TEMPO_CLASS, 2400, "slow"),
            (EventName.TEMPO_VALUE, 2400, 0),
            (EventName.POSITION, 2880, "9/16"),
            (EventName.TEMPO_CLASS, 2880, "mid"),
            (EventName.TEMPO_VALUE, 2880, 0),
            (EventName.POSITION, 2880, "9/16"),
            (EventName.CHORD, 2880, "G:dim"),
            (EventName.BAR, 3840, None),
            (EventName.POSITION, 3840, "1/16"),
            (EventName.TEMPO_CLASS, 3840, "fast"),
            (EventName.TEMPO_VALUE, 3840, 0),
            (EventName.POSITION, 3840, "1/16"),
            (EventName.CHORD, 3840, "E:dom"),
            (EventName.POSITION, 3840, "1/16"),
            (EventName.NOTE_VELOCITY, 3840, 12),
            (EventName.NOTE_ON, 3840, 64),
            (EventName.NOTE_DURATION, 3840, 31),
            (EventName.POSITION, 3840, "1/16"),
            (EventName.NOTE_VELOCITY, 3840, 12),
            (EventName.NOTE_ON, 3840, 68),
            (EventName.NOTE_DURATION, 3840, 31),
        ]
        for event, (expected_name, expected_time, expected_value) in zip(
            events, expected_events
        ):
            assert event.name == expected_name
            assert event.time == expected_time
            assert event.value == expected_value
