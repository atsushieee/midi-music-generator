"""The common Functions Module."""
from pathlib import Path

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


@pytest.fixture
def create_temp_midi_files(
    create_sample_midi: midi_parser.MidiFile, tmp_path: Path
) -> Path:
    """Create temporary MIDI files for testing.

    Args:
        create_sample_midi (midi_parser.MidiFile):
            A sample MIDI file object used for creating the temporary files.
        tmp_path (Path):
            The temporary directory path provided by pytest
            where the MIDI files will be created.

    Returns:
        Path: The path to the directory containing the temporary MIDI files.
    """
    num_files = 10
    for i in range(num_files):
        midi_obj = create_sample_midi
        filepath = tmp_path / f"tmp_{i:03}.mid"
        midi_obj.dump(str(filepath))
    return tmp_path
