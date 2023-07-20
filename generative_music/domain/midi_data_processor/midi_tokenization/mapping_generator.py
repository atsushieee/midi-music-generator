"""A module to generate mappings for MIDI events and IDs for DNN learning."""
import json
from pathlib import Path
from typing import Dict


class MappingGenerator:
    """A class that generates event2id and id2event mappings for MIDI data processing."""

    def __init__(self):
        """Initialize the MappingGenerator instance and create the data dictionary."""
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
        keys.append("Bar_None")
        # Position values
        for i in range(1, 17):
            keys.append(f"Position_{i}/16")
        # Chord values
        keys.append("Chord_N:N")
        chords = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        chord_types = ["maj", "min", "dom", "aug", "dim"]
        for i, chord in enumerate(chords):
            for j, chord_type in enumerate(chord_types):
                keys.append(f"Chord_{chord}:{chord_type}")
        # Tempo Class values
        tempo_classes = ["fast", "mid", "slow"]
        for i, tempo_class in enumerate(tempo_classes):
            keys.append(f"Tempo Class_{tempo_class}")
        # Tempo Value values
        for i in range(60):
            keys.append(f"Tempo Value_{i}")
        # Note On values
        for i in range(128):
            keys.append(f"Note On_{i}")
        # Note Velocity values
        for i in range(33):
            keys.append(f"Note Velocity_{i}")
        # Note Duration values
        for i in range(64):
            keys.append(f"Note Duration_{i}")

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
