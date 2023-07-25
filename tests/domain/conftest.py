"""The common Functions Module."""
import pytest
from miditoolkit.midi import parser as midi_parser


@pytest.fixture
def create_sample_midi() -> midi_parser.MidiFile:
    """Create a test MIDI file with specified tempo changes and notes.

    Returns:
        midi_parser.MidiFile: The MidiFile object created.
    """
    midi_obj = midi_parser.MidiFile()
    midi_obj.tempo_changes = [
        midi_parser.TempoChange(120, 0),
        midi_parser.TempoChange(100, 480),
        midi_parser.TempoChange(80, 720),
        midi_parser.TempoChange(100, 1020),
        midi_parser.TempoChange(60, 1080),
        midi_parser.TempoChange(140, 1860),
    ]
    midi_obj.instruments.append(midi_parser.Instrument(0))
    midi_obj.instruments[0].notes = [
        midi_parser.Note(60, 62, 0, 480),
        midi_parser.Note(64, 66, 361, 960),
        midi_parser.Note(67, 69, 960, 1440),
        midi_parser.Note(71, 73, 1440, 1920),
        midi_parser.Note(60, 61, 1675, 1915),
    ]
    return midi_obj
