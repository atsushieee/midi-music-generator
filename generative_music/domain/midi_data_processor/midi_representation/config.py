"""MIDI configuration for music generation models."""
from pathlib import Path

import numpy as np


class Config:
    """Configuration for MIDI event representation in music generation models."""

    # Midi events-IDs token mapping
    MAPPING_DATA_FOLDER = Path("generative_music/data")
    EVENT2ID_FILEPATH = MAPPING_DATA_FOLDER / "event2id.json"
    ID2EVENT_FILEPATH = MAPPING_DATA_FOLDER / "id2event.json"
    # Midi Info: Postion
    TICKS_PER_BAR = 1920
    MIN_RESOLUTION = 120
    DEFAULT_FRACTION = int(TICKS_PER_BAR / MIN_RESOLUTION)
    # Midi Info: Tempo
    TEMPO_RESOLUTION = 480
    # Must: len(DEFAULT_TEMPO_NAMES) == len(DEFAULT_TEMPO_INTERVALS)
    DEFAULT_TEMPO_NAMES = ["slow", "mid", "fast"]
    DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]
    NUM_TEMPO_INTERVAL = max([len(interval) for interval in DEFAULT_TEMPO_INTERVALS])
    # Midi Info: Note
    NUM_NOTE_PITCHES = 128
    DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32 + 1, dtype=int)
    NUM_NOTE_VELOCITIES = len(DEFAULT_VELOCITY_BINS)
    # The max duration is the length of 2 bars,
    # and default_duration_bins are generated within the range of min_resolution
    # to the tick count of 2 bars.
    DEFAULT_DURATION_BINS = np.arange(
        MIN_RESOLUTION, TICKS_PER_BAR * 2 + 1, MIN_RESOLUTION, dtype=int
    )
    NUM_NOTE_DURATIONS = len(DEFAULT_DURATION_BINS)
    # Chord info
    CHORD_RESOLUTION = 480
    PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    CHORD_TYPES = ["maj", "min", "dom", "aug", "dim"]
    CHORD_MAPS = {
        "maj": [0, 4],
        "min": [0, 3],
        "dim": [0, 3, 6],
        "aug": [0, 4, 8],
        "dom": [0, 4, 7, 10],
    }
    # define chord insiders (scoring: +1)
    CHORD_INSIDERS = {"maj": [7], "min": [7], "dim": [9]}
    # define chord outsiders (scoring: -1)
    CHORD_OUTSIDERS_1 = {
        "maj": [2, 5, 9],
        "min": [2, 5, 8],
        "dim": [2, 5, 10],
        "aug": [2, 5, 9],
        "dom": [2, 5, 9],
    }
    # define chord outsiders (scoring: -2)
    CHORD_OUTSIDERS_2 = {
        "maj": [1, 3, 6, 8, 10],
        "min": [1, 4, 6, 9, 11],
        "dim": [1, 4, 7, 8, 11],
        "aug": [1, 3, 6, 7, 10],
        "dom": [1, 3, 6, 8, 11],
    }
