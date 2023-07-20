"""MIDI configuration for music generation models."""
from pathlib import Path

import numpy as np


class Config:
    """Configuration for MIDI event representation in music generation models."""

    # Midi events-IDs token mapping
    MAPPING_DATA_FOLDER = Path("generative_music/data")
    EVENT2ID_FILEPATH = MAPPING_DATA_FOLDER / "event2id.json"
    ID2EVENT_FILEPATH = MAPPING_DATA_FOLDER / "id2event.json"
    # Chord info
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
    CHORD_INSIDERS = {"maj": [7], "min": [7], "dim": [9], "aug": [], "dom": []}
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

    def __init__(self, ticks_per_bar: int = 1920, min_resolution: int = 120):
        """Initialize the configuration with the given parameters.

        Args:
            ticks_per_bar (int):
                The number of ticks per bar in the MIDI representation. Defalut is 1920.
            min_resolution (int):
                The minimum resolution for the MIDI representation.Default is 120.
        """
        # Midi info
        self.default_fraction = ticks_per_bar // min_resolution
        self.default_tempo_name = ["slow", "mid", "fast"]
        self.default_tempo_intervals = [range(30, 90), range(90, 150), range(150, 210)]
        self._validate_tempo_data()
        self.num_tempo_interval = max(
            [len(interval) for interval in self.default_tempo_intervals]
        )
        self.num_note_pitches = 128
        self.default_velocity_bins = np.linspace(0, 128, 32 + 1, dtype=int)
        self.num_note_velocities = len(self.default_velocity_bins)
        self.default_duration_bins = np.arange(60, 3841, 60, dtype=int)
        self.num_note_durations = len(self.default_duration_bins)

    def _validate_tempo_data(self):
        """Validate that the lengths of default_tempo_name and default_tempo_intervals are the same.

        Raises:
            ValueError:
                If the length of default_tempo_name and default_tempo_intervals is not the same.
        """
        if len(self.default_tempo_name) != len(self.default_tempo_intervals):
            raise ValueError(
                "The length of default_tempo_name and default_tempo_intervals must be the same."
            )
