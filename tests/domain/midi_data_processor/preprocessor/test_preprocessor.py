"""Tests for a class that preprocess MIDI file."""
import os

from generative_music.domain.midi_data_processor.midi_representation import (
    Event, EventName)
from generative_music.domain.midi_data_processor.preprocessor.preprocessor import \
    Preprocessor
from tests.domain.midi_data_processor.preprocessor.conftest import \
    create_test_midi


class TestPreprocessor:
    """A test class for the Preprocessor.

    The class tests if the MIDI tempo and note data are correctly processed
    and if the generated events match the expected values.
    """

    def setup_method(self):
        """Initialize the Preprocessor tests.

        Set up the test environment by creating a test MIDI file
        and initializing the Preprocessor instance.
        This method is called before each test function is executed.
        """
        self.test_midi_path = create_test_midi()
        self.preprocessor = Preprocessor(
            self.test_midi_path, note_resolution=240, tempo_resolution=240
        )

    def teardown_method(self):
        """Clean up the test environment by removing the test file.

        This method is called after each test function is executed.
        """
        os.remove(self.test_midi_path)

    def test_process(self):
        """Test if the Preprocessor's process method generates the correct events.

        This test checks if the process method generates a list of events
        with the expected events including BAR, POSITION, NOTE_ON, CHORD, and TEMPO_CLASS events.
        """
        events = self.preprocessor.process()

        assert events is not None
        assert len(events) > 0
        assert isinstance(events[0], Event)

        # Check if the first event is a BAR event
        assert events[0].name == EventName.BAR
        assert events[0].time == 0

        # Check if there are any POSITION events
        position_events = [
            event for event in events if event.name == EventName.POSITION
        ]
        assert len(position_events) > 0

        # Check if there are any NOTE_ON, CHORD, or TEMPO events
        note_on_events = [event for event in events if event.name == EventName.NOTE_ON]
        chord_events = [event for event in events if event.name == EventName.CHORD]
        print(chord_events)
        tempo_events = [
            event for event in events if event.name == EventName.TEMPO_CLASS
        ]
        assert len(note_on_events) > 0
        assert len(chord_events) > 0
        assert len(tempo_events) > 0
