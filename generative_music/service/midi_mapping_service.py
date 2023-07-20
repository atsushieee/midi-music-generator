"""MIDI event-ID and ID-event mappings service for music generation models.

The purpose of this file is to provide a service
for generating and saving MIDI event-ID and ID-event mappings.
The generated mappings can be used for tokenization and detokenization
in neural network models.
"""
from pathlib import Path

from generative_music.domain.midi_data_processor.config import Config
from generative_music.domain.midi_data_processor.midi_tokenization import \
    MappingGenerator


class MidiMappingService:
    """A service for generating and saving MIDI event-ID mappings."""

    def __init__(self):
        """Initialize the MidiMappingService instance."""
        self.mapping_generator = MappingGenerator()

    def save_mapping_data(self, filename: Path):
        """Save the MIDI event-ID mapping data to a file.

        Args:
            filename (Path): The file path where the mapping data will be saved.
        """
        self.mapping_generator.save_data(filename)

    def save_reversed_mapping_data(self, filename: Path):
        """Save the reversed MIDI ID-event mapping data to a file.

        Args:
            filename (Path): The file path where the mapping data will be saved.
        """
        self.mapping_generator.save_reversed_data(filename)


if __name__ == "__main__":
    midi_mapping_service = MidiMappingService()
    midi_mapping_service.save_mapping_data(Config.EVENT2ID_FILEPATH)
    midi_mapping_service.save_reversed_mapping_data(Config.ID2EVENT_FILEPATH)
