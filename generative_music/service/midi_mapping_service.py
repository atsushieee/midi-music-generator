"""MIDI event-ID and ID-event mappings service for music generation models.

The purpose of this file is to provide a service
for generating and saving MIDI event-ID and ID-event mappings.
The generated mappings can be used for tokenization and detokenization
in neural network models.
"""
from generative_music.domain.midi_data_processor.midi_representation import \
    Config
from generative_music.domain.midi_data_processor.midi_tokenization import \
    MappingGenerator


class MidiMappingService:
    """A service for generating and saving MIDI event-ID mappings."""

    def __init__(self, midi_config: Config):
        """Initialize the MidiMappingService instance.

        Args:
            midi_config (Config): The Configuration for MIDI representation.
        """
        self.midi_config = midi_config
        self.mapping_generator = MappingGenerator(midi_config)

    def save_mapping_data(self):
        """Save the MIDI event-ID mapping data to a file."""
        self.mapping_generator.save_data(self.midi_config.EVENT2ID_FILEPATH)

    def save_reversed_mapping_data(self):
        """Save the reversed MIDI ID-event mapping data to a file."""
        self.mapping_generator.save_reversed_data(self.midi_config.ID2EVENT_FILEPATH)


if __name__ == "__main__":
    midi_config = Config()
    midi_mapping_service = MidiMappingService(midi_config)
    midi_mapping_service.save_mapping_data()
    midi_mapping_service.save_reversed_mapping_data()
