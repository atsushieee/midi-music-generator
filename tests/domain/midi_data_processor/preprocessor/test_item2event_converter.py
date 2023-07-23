"""Tests for a class that converts item to event data."""
from generative_music.domain.midi_data_processor.midi_representation import (
    Config, EventName, Item, ItemName)
from generative_music.domain.midi_data_processor.preprocessor.item2event_converter import \
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
        self.midi_config = Config()
        self.second_bar = self.midi_config.TICKS_PER_BAR
        self.third_bar = self.midi_config.TICKS_PER_BAR * 2
        self.fourth_bar = self.midi_config.TICKS_PER_BAR * 3
        self.second_beat = int(self.midi_config.TICKS_PER_BAR / 4)
        self.third_beat = int(self.midi_config.TICKS_PER_BAR / 2)
        self.test_items = [
            # Tempo
            Item(name=ItemName.TEMPO, start=0, tempo=120),
            Item(name=ItemName.TEMPO, start=self.second_bar, tempo=240),
            Item(
                name=ItemName.TEMPO, start=self.second_bar + self.second_beat, tempo=25
            ),
            Item(
                name=ItemName.TEMPO, start=self.second_bar + self.third_beat, tempo=90
            ),
            Item(name=ItemName.TEMPO, start=self.third_bar, tempo=150),
            # Chords
            Item(name=ItemName.CHORD, start=0, end=self.second_bar, pitch="C:maj"),
            Item(
                name=ItemName.CHORD,
                start=self.second_bar,
                end=self.second_bar + self.third_beat,
                pitch="D:min",
            ),
            Item(
                name=ItemName.CHORD,
                start=self.second_bar + self.third_beat,
                end=self.third_bar,
                pitch="G:dim",
            ),
            Item(
                name=ItemName.CHORD,
                start=self.third_bar,
                end=self.fourth_bar,
                pitch="E:dom",
            ),
            # Notes
            Item(
                name=ItemName.NOTE, start=0, end=self.second_bar, velocity=43, pitch=60
            ),  # C
            Item(
                name=ItemName.NOTE,
                start=0,
                end=self.second_bar - 10,
                velocity=40,
                pitch=64,
            ),  # E
            Item(
                name=ItemName.NOTE,
                start=0,
                end=self.second_bar + 10,
                velocity=48,
                pitch=67,
            ),  # G
            Item(
                name=ItemName.NOTE,
                start=self.second_bar,
                end=self.second_bar + self.third_beat,
                velocity=65,
                pitch=62,
            ),  # D
            Item(
                name=ItemName.NOTE,
                start=self.third_bar,
                end=self.fourth_bar,
                velocity=48,
                pitch=64,
            ),  # E
            Item(
                name=ItemName.NOTE,
                start=self.third_bar,
                end=self.fourth_bar + 30,
                velocity=48,
                pitch=68,
            ),  # G#
        ]
        max_time = self.test_items[-1].end
        self.item2event_converter = Item2EventConverter(
            self.test_items, max_time, self.midi_config
        )

    def test_convert_items_to_events(self):
        """Test if the items are correctly converted to events and match the expected values.

        This test checks the number of events
        and the consistency between the expected events (name, time, value, text)
        and the actual events.
        """
        events = self.item2event_converter.convert_items_to_events()
        # Position
        first_beat = f"1/{self.midi_config.DEFAULT_FRACTION}"
        second_beat = f"{int(self.midi_config.DEFAULT_FRACTION / 4) + 1}/{self.midi_config.DEFAULT_FRACTION}"
        third_beat = f"{int(self.midi_config.DEFAULT_FRACTION / 2) + 1}/{self.midi_config.DEFAULT_FRACTION}"
        expected_events = [
            (EventName.BAR, 0, None),
            (EventName.POSITION, 0, first_beat),
            (EventName.TEMPO_CLASS, 0, "mid"),
            (EventName.TEMPO_VALUE, 0, 30),
            (EventName.POSITION, 0, first_beat),
            (EventName.CHORD, 0, "C:maj"),
            (EventName.POSITION, 0, first_beat),
            (EventName.NOTE_VELOCITY, 0, 10),
            (EventName.NOTE_ON, 0, 60),
            (EventName.NOTE_DURATION, 0, self.midi_config.NUM_NOTE_DURATIONS / 2 - 1),
            (EventName.POSITION, 0, first_beat),
            (EventName.NOTE_VELOCITY, 0, 10),
            (EventName.NOTE_ON, 0, 64),
            (EventName.NOTE_DURATION, 0, self.midi_config.NUM_NOTE_DURATIONS / 2 - 1),
            (EventName.POSITION, 0, first_beat),
            (EventName.NOTE_VELOCITY, 0, 12),
            (EventName.NOTE_ON, 0, 67),
            (EventName.NOTE_DURATION, 0, self.midi_config.NUM_NOTE_DURATIONS / 2 - 1),
            (EventName.BAR, self.second_bar, None),
            (EventName.POSITION, self.second_bar, first_beat),
            (EventName.TEMPO_CLASS, self.second_bar, "fast"),
            (EventName.TEMPO_VALUE, self.second_bar, 59),
            (EventName.POSITION, self.second_bar, first_beat),
            (EventName.CHORD, self.second_bar, "D:min"),
            (EventName.POSITION, self.second_bar, first_beat),
            (EventName.NOTE_VELOCITY, self.second_bar, 16),
            (EventName.NOTE_ON, self.second_bar, 62),
            (
                EventName.NOTE_DURATION,
                self.second_bar,
                self.midi_config.NUM_NOTE_DURATIONS / 4 - 1,
            ),
            (EventName.POSITION, self.second_bar + self.second_beat, second_beat),
            (EventName.TEMPO_CLASS, self.second_bar + self.second_beat, "slow"),
            (EventName.TEMPO_VALUE, self.second_bar + self.second_beat, 0),
            (EventName.POSITION, self.second_bar + self.third_beat, third_beat),
            (EventName.TEMPO_CLASS, self.second_bar + self.third_beat, "mid"),
            (EventName.TEMPO_VALUE, self.second_bar + self.third_beat, 0),
            (EventName.POSITION, self.second_bar + self.third_beat, third_beat),
            (EventName.CHORD, self.second_bar + self.third_beat, "G:dim"),
            (EventName.BAR, self.third_bar, None),
            (EventName.POSITION, self.third_bar, first_beat),
            (EventName.TEMPO_CLASS, self.third_bar, "fast"),
            (EventName.TEMPO_VALUE, self.third_bar, 0),
            (EventName.POSITION, self.third_bar, first_beat),
            (EventName.CHORD, self.third_bar, "E:dom"),
            (EventName.POSITION, self.third_bar, first_beat),
            (EventName.NOTE_VELOCITY, self.third_bar, 12),
            (EventName.NOTE_ON, self.third_bar, 64),
            (
                EventName.NOTE_DURATION,
                self.third_bar,
                self.midi_config.NUM_NOTE_DURATIONS / 2 - 1,
            ),
            (EventName.POSITION, self.third_bar, first_beat),
            (EventName.NOTE_VELOCITY, self.third_bar, 12),
            (EventName.NOTE_ON, self.third_bar, 68),
            (
                EventName.NOTE_DURATION,
                self.third_bar,
                self.midi_config.NUM_NOTE_DURATIONS / 2 - 1,
            ),
        ]
        for event, (expected_name, expected_time, expected_value) in zip(
            events, expected_events
        ):
            assert event.name == expected_name
            assert event.time == expected_time
            assert event.value == expected_value
