"""Tests for a class that writes midi file based on events data."""
from pathlib import Path

import miditoolkit
import pytest

from generative_music.domain.midi_data_processor.midi_representation import (
    Config, Event, EventName)
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
        # Position
        first_beat = f"1/{self.midi_config.DEFAULT_FRACTION}"
        second_beat = f"{self.midi_config.DEFAULT_FRACTION/4+1}/{self.midi_config.DEFAULT_FRACTION}"
        third_beat = f"{self.midi_config.DEFAULT_FRACTION/2+1}/{self.midi_config.DEFAULT_FRACTION}"
        self.first_note_velocity_id = 10
        self.first_note_duration_id = 15
        self.test_events = [
            Event(EventName.BAR, 0, None),
            Event(EventName.POSITION, 0, first_beat),
            Event(EventName.TEMPO_CLASS, 0, self.midi_config.DEFAULT_TEMPO_NAMES[1]),
            Event(EventName.TEMPO_VALUE, 0, 30),
            Event(EventName.POSITION, 0, first_beat),
            Event(EventName.CHORD, 0, "C:maj"),
            Event(EventName.POSITION, 0, first_beat),
            Event(EventName.NOTE_VELOCITY, 0, self.first_note_velocity_id),
            Event(EventName.NOTE_ON, 0, 60),
            Event(EventName.NOTE_DURATION, 0, self.first_note_duration_id),
            Event(
                EventName.POSITION, int(self.midi_config.TICKS_PER_BAR / 4), second_beat
            ),
            Event(EventName.NOTE_VELOCITY, int(self.midi_config.TICKS_PER_BAR / 4), 11),
            Event(EventName.NOTE_ON, int(self.midi_config.TICKS_PER_BAR / 4), 64),
            Event(EventName.NOTE_DURATION, int(self.midi_config.TICKS_PER_BAR / 4), 11),
            Event(EventName.BAR, self.midi_config.TICKS_PER_BAR, None),
            Event(EventName.POSITION, self.midi_config.TICKS_PER_BAR, first_beat),
            Event(
                EventName.TEMPO_CLASS,
                self.midi_config.TICKS_PER_BAR,
                self.midi_config.DEFAULT_TEMPO_NAMES[2],
            ),
            Event(EventName.TEMPO_VALUE, self.midi_config.TICKS_PER_BAR, 30),
            Event(EventName.POSITION, self.midi_config.TICKS_PER_BAR, first_beat),
            Event(EventName.CHORD, self.midi_config.TICKS_PER_BAR, "D:min"),
            Event(EventName.POSITION, self.midi_config.TICKS_PER_BAR, first_beat),
            Event(EventName.NOTE_VELOCITY, self.midi_config.TICKS_PER_BAR, 8),
            Event(EventName.NOTE_ON, self.midi_config.TICKS_PER_BAR, 62),
            Event(EventName.NOTE_DURATION, self.midi_config.TICKS_PER_BAR, 15),
            Event(
                EventName.POSITION,
                int(self.midi_config.TICKS_PER_BAR * 1.5),
                third_beat,
            ),
            Event(
                EventName.NOTE_VELOCITY, int(self.midi_config.TICKS_PER_BAR * 1.5), 9
            ),
            Event(EventName.NOTE_ON, int(self.midi_config.TICKS_PER_BAR * 1.5), 65),
            Event(
                EventName.NOTE_DURATION, int(self.midi_config.TICKS_PER_BAR * 1.5), 7
            ),
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
            == self.midi_config.DEFAULT_TEMPO_INTERVALS[1].start + 30
        )
        assert midi.tempo_changes[1].time == self.midi_config.TICKS_PER_BAR
        assert midi.tempo_changes[1].tempo == pytest.approx(
            self.midi_config.DEFAULT_TEMPO_INTERVALS[2].start + 30, abs=1e-3
        )
        assert len(midi.markers) == 2
        assert midi.markers[0].time == 0
        assert midi.markers[0].text == "C:maj"
        assert midi.markers[1].time == self.midi_config.TICKS_PER_BAR
        assert midi.markers[1].text == "D:min"
        assert len(midi.instruments[0].notes) == 4
        # Add assertions for note properties (start, end, pitch, velocity)
        # for each note in midi.instruments[0].notes
        # Example for the first note:
        assert midi.instruments[0].notes[0].start == 0
        assert (
            midi.instruments[0].notes[0].end
            == self.midi_config.DEFAULT_DURATION_BINS[self.first_note_duration_id]
        )
        assert midi.instruments[0].notes[0].pitch == 60
        assert (
            midi.instruments[0].notes[0].velocity
            == self.midi_config.DEFAULT_VELOCITY_BINS[self.first_note_velocity_id]
        )
