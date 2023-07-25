"""Tests for a class that preprocess MIDI file."""
from pathlib import Path

import pytest
from miditoolkit.midi import parser as midi_parser

from generative_music.domain.midi_data_processor.midi_representation import (
    Config, Event, EventName)
from generative_music.domain.midi_data_processor.preprocessor.preprocessor import \
    Preprocessor


class TestPreprocessor:
    """A test class for the Preprocessor.

    The class tests if the MIDI tempo and note data are correctly processed
    and if the generated events match the expected values.
    """

    @pytest.fixture(autouse=True)
    def setup_module(self, create_sample_midi: midi_parser.MidiFile, tmp_path: Path):
        """Initialize the Preprocessor tests.

        Set up the test environment by creating a test MIDI file
        and initializing the Preprocessor instance.

        Args:
            create_sample_midi (midi_parser.MidiFile):
                sample MidiFile object to be used in the tests.
            tmp_path (Path): The temporary directory path provided by the pytest fixture.
        """
        test_midi_obj = create_sample_midi
        self.test_midi_filepath = tmp_path / "tmp.mid"
        test_midi_obj.dump(str(self.test_midi_filepath))
        self.preprocessor = Preprocessor(self.test_midi_filepath, Config())

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
        tempo_events = [
            event for event in events if event.name == EventName.TEMPO_CLASS
        ]
        assert len(note_on_events) > 0
        assert len(chord_events) > 0
        assert len(tempo_events) > 0
