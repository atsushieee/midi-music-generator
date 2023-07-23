"""A module for preprocessing midi data."""
from pathlib import Path
from typing import List

from generative_music.domain.midi_data_processor.midi_representation import (
    Config, Event)
from generative_music.domain.midi_data_processor.preprocessor.chord_extractor import \
    ChordExtractor
from generative_music.domain.midi_data_processor.preprocessor.data_loader import \
    DataLoader
from generative_music.domain.midi_data_processor.preprocessor.item2event_converter import \
    Item2EventConverter


class Preprocessor:
    """A class for loading and processing MIDI files.

    This class handles the loading and processing of MIDI files.
    It provides methods for reading note and tempo information,
    extracting chords, and converting the combined items to events.
    The generated events are formatted into an appropriate format
    for easier handling and analysis.
    """

    def __init__(self, midi_file_path: Path, midi_config: Config):
        """Initialize the Preprocessor instance with the given MIDI file.

        Args:
            midi_file_path (Path): The path to the MIDI file to be processed.
            midi_config (Config): The Configuration for MIDI representation.
        """
        self.midi_file_path = midi_file_path
        self.midi_config = midi_config

    def process(self) -> List[Event]:
        """Process the MIDI file and generate a list of events.

        This method reads and processes note and tempo items from the MIDI file,
        extracts chords from note items, combines note, chord and tempo items,
        and converts the combined items to events.

        Returns:
            List[Event]: A list of events generated from the MIDI file.
        """
        # Initialize DataLoader, ChordExtractor, and Item2EventConverter
        data_loader = DataLoader(self.midi_file_path, self.midi_config)
        chord_extractor = ChordExtractor(self.midi_config)

        # Read and process notes and tempo items
        note_items = data_loader.read_note_items()
        tempo_items = data_loader.read_tempo_items()

        # Extract chords from note items
        chord_items = chord_extractor.extract(note_items)

        # Combine note, chord, and tempo items
        combined_items = tempo_items + chord_items + note_items

        # Initialize Item2EventConverter with combined items and max_time
        max_time = max(
            item.end if item.end is not None else item.start for item in note_items
        )
        item2event_converter = Item2EventConverter(
            combined_items, max_time, self.midi_config
        )

        # Convert combined items to events
        events = item2event_converter.convert_items_to_events()

        return events
