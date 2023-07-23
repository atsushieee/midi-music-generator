"""A module to generate mappings for MIDI events and IDs for DNN learning."""
import json
from pathlib import Path
from typing import Dict

from generative_music.domain.midi_data_processor.midi_representation import (
    Config, EventName)


class MappingGenerator:
    """A class that generates event2id and id2event mappings for MIDI data processing."""

    def __init__(self, midi_config: Config):
        """Initialize the MappingGenerator instance and create the data dictionary.

        Args:
            midi_config (Config): The Configuration for MIDI representation.
        """
        self.midi_config = midi_config
        self.data = self._create_data()

    def save_data(self, filename: Path):
        """Save the data dictionary to a file.

        Args:
            filename (Path): The filename for the output file.
        """
        self._save_data(filename, self.data)

    def save_reversed_data(self, filename: Path):
        """Save the data dictionary to a file.

        Args:
            filename (Path): The filename for the output file.
        """
        reversed_data = self._reverse_data()
        self._save_data(filename, reversed_data)

    def _create_data(self) -> Dict[str, int]:
        """Create the data dictionary.

        This dictionary contains keys for various MIDI events
        and their corresponding IDs.

        Returns:
            Dict[str, int]:
                The data dictionary with string keys and integer values.
        """
        keys = []
        # Bar values
        keys.append(f"{EventName.BAR.value}_None")
        # Position values
        for i in range(self.midi_config.DEFAULT_FRACTION):
            keys.append(
                f"{EventName.POSITION.value}_{i+1}/{self.midi_config.DEFAULT_FRACTION}"
            )
        # Chord values
        keys.append(f"{EventName.CHORD.value}_N:N")
        for chord in self.midi_config.PITCH_CLASSES:
            for chord_type in self.midi_config.CHORD_TYPES:
                keys.append(f"{EventName.CHORD.value}_{chord}:{chord_type}")
        # Tempo Class values
        for tempo_class in self.midi_config.DEFAULT_TEMPO_NAMES:
            keys.append(f"{EventName.TEMPO_CLASS.value}_{tempo_class}")
        # Tempo Value values
        for i in range(self.midi_config.NUM_TEMPO_INTERVAL):
            keys.append(f"{EventName.TEMPO_VALUE.value}_{i}")
        # Note On values
        for i in range(self.midi_config.NUM_NOTE_PITCHES):
            keys.append(f"{EventName.NOTE_ON.value}_{i}")
        # Note Velocity values
        for i in range(self.midi_config.NUM_NOTE_VELOCITIES):
            keys.append(f"{EventName.NOTE_VELOCITY.value}_{i}")
        # Note Duration values
        for i in range(self.midi_config.NUM_NOTE_DURATIONS):
            keys.append(f"{EventName.NOTE_DURATION.value}_{i}")

        # Assign values to the keys in the dictionary
        data = {key: idx for idx, key in enumerate(keys)}
        return data

    def _reverse_data(self) -> Dict[int, str]:
        """Generate a reversed dictionary with values as keys and keys as values.

        Returns:
            Dict[int, str]:
                The reversed dictionary with integer keys and string values.
        """
        return {v: k for k, v in self.data.items()}

    def _save_data(self, filename: Path, data: Dict):
        """Save the given data dictionary to a file.

        Args:
            filename (Path): The filename for the output file.
            data (Dict): The data dictionary to save.
        """
        with open(filename, "w") as outfile:
            json.dump(data, outfile, ensure_ascii=False, indent=2)
