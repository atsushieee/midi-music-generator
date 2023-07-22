"""Tests for a class that writes midi file based on events data."""
from pathlib import Path

import miditoolkit
import pytest

from generative_music.domain.midi_data_processor.config import Config
from generative_music.domain.midi_data_processor.data_elements import (
    Event, EventName)
from generative_music.domain.midi_data_processor.postprocessor.data_writer import \
    DataWriter


class TestDataWriter:
    """A test class for the DataWriter.

    This class contains tests for the DataWriter class,
    which is responsible for writing MIDI data from events.
    The tests cover the main functionality of the DataWriter class,
    such as writing MIDI files with the correct notes, chords, and tempos.
    """

    def setup_method(self):
        """Initialize the DataWriter tests.

        Set up the test environment by creating test items
        and initializing the DataWriter instance.
        This method is called before each test function is executed.
        """
        self.midi_config = Config()
        self.first_note_velocity_id = 10
        self.first_note_duration_id = 31
        self.test_events = [
            Event(EventName.BAR, 0, None),
            Event(EventName.POSITION, 0, "1/16"),
            Event(EventName.TEMPO_CLASS, 0, self.midi_config.default_tempo_name[1]),
            Event(EventName.TEMPO_VALUE, 0, 30),
            Event(EventName.POSITION, 0, "1/16"),
            Event(EventName.CHORD, 0, "C:maj"),
            Event(EventName.POSITION, 0, "1/16"),
            Event(EventName.NOTE_VELOCITY, 0, self.first_note_velocity_id),
            Event(EventName.NOTE_ON, 0, 60),
            Event(EventName.NOTE_DURATION, 0, self.first_note_duration_id),
            Event(EventName.POSITION, 480, "5/16"),
            Event(EventName.NOTE_VELOCITY, 480, 11),
            Event(EventName.NOTE_ON, 480, 64),
            Event(EventName.NOTE_DURATION, 480, 23),
            Event(EventName.BAR, 1920, None),
            Event(EventName.POSITION, 1920, "1/16"),
            Event(EventName.TEMPO_CLASS, 1920, self.midi_config.default_tempo_name[2]),
            Event(EventName.TEMPO_VALUE, 1920, 30),
            Event(EventName.POSITION, 1920, "1/16"),
            Event(EventName.CHORD, 1920, "D:min"),
            Event(EventName.POSITION, 1920, "1/16"),
            Event(EventName.NOTE_VELOCITY, 1920, 8),
            Event(EventName.NOTE_ON, 1920, 62),
            Event(EventName.NOTE_DURATION, 1920, 31),
            Event(EventName.POSITION, 2880, "9/16"),
            Event(EventName.NOTE_VELOCITY, 2880, 9),
            Event(EventName.NOTE_ON, 2880, 65),
            Event(EventName.NOTE_DURATION, 2880, 15),
        ]
        self.data_writer = DataWriter(self.midi_config)

    def test_write_midi_file(self, tmp_path: Path):
        """Test the write_midi_file method of the DataWriter class.

        This test checks if the write_midi_file method generates the correct MIDI file
        with the expected notes, chords, and tempos based on the provided test_events.

        Args:
            tmp_path (Path): Temporary path for the test file.
        """
        output_path = tmp_path / "test_output.mid"
        self.data_writer.write_midi_file(self.test_events, output_path)
        assert output_path.exists()

        # Load the generated MIDI file and check its contents
        midi = miditoolkit.midi.parser.MidiFile(str(output_path))
        assert len(midi.instruments) == 1
        assert len(midi.tempo_changes) == 2
        assert midi.tempo_changes[0].time == 0
        assert (
            midi.tempo_changes[0].tempo
            == self.midi_config.default_tempo_intervals[1].start + 30
        )
        assert midi.tempo_changes[1].time == 1920
        assert midi.tempo_changes[1].tempo == pytest.approx(
            self.midi_config.default_tempo_intervals[2].start + 30, abs=1e-3
        )
        assert len(midi.markers) == 2
        assert midi.markers[0].time == 0
        assert midi.markers[0].text == "C:maj"
        assert midi.markers[1].time == 1920
        assert midi.markers[1].text == "D:min"
        assert len(midi.instruments[0].notes) == 4
        # Add assertions for note properties (start, end, pitch, velocity)
        # for each note in midi.instruments[0].notes
        # Example for the first note:
        assert midi.instruments[0].notes[0].start == 0
        assert (
            midi.instruments[0].notes[0].end
            == self.midi_config.default_duration_bins[self.first_note_duration_id]
        )
        assert midi.instruments[0].notes[0].pitch == 60
        assert (
            midi.instruments[0].notes[0].velocity
            == self.midi_config.default_velocity_bins[self.first_note_velocity_id]
        )
