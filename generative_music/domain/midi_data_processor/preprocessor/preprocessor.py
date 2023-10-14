"""A module for preprocessing midi data."""
import copy
from pathlib import Path
from typing import List, Tuple

from generative_music.domain.midi_data_processor.midi_representation import (
    Config, Event, Item)
from generative_music.domain.midi_data_processor.preprocessor.chord_extractor import \
    ChordExtractor
from generative_music.domain.midi_data_processor.preprocessor.item2event_converter import \
    Item2EventConverter
from generative_music.domain.midi_data_processor.preprocessor.midi_data_loader import \
    MidiDataLoader


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
        # Initialize DataLoader, ChordExtractor
        self._load_midi_original_note_and_tempo()

    def generate_events(
        self, is_augmented: bool = False, shift: int = 0, stretch: float = 1.0
    ) -> List[Event]:
        """Generate a list of events from the MIDI data.

        This method applies pitch shift and tempo stretching if specified,
        extracts chords from note items, combines note, chord and tempo items,
        and converts the combined items to events.

        Args:
            is_augmented (bool, optional):
                Whether to apply data augmentation. Defaults to False.
            shift (int, optional):
                The amount to shift the pitch of the notes.
                Positive values shift the pitch up, while negative values shift it down.
                Defaults to 0, meaning no shift.
            stretch (float, optional):
                The factor by which to stretch or shrink the tempo.
                Values greater than 1.0 will speed up the tempo,
                while values less than 1.0 will slow it down.
                Defaults to 1.0, meaning no change in tempo.

        Returns:
            List[Event]: A list of events generated from the MIDI file.
        """
        note_items = self.note_items
        tempo_items = self.tempo_items
        chord_items = self.chord_items
        # If is_augmented is True,
        # apply augmentation to the note, tempo and chord items
        if is_augmented and (shift != 0 or stretch != 1.0):
            note_items, tempo_items, chord_items = self._apply_augmentation(
                shift, stretch
            )

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

    def _load_midi_original_note_and_tempo(self):
        """Load note and tempo data from the MIDI file.

        This method reads note and tempo items from the MIDI file
        and stores them internally.
        """
        data_loader = MidiDataLoader(self.midi_file_path, self.midi_config)
        # Read notes and tempo items
        self.note_items = data_loader.read_note_items()
        self.tempo_items = data_loader.read_tempo_items()
        # Extract chords from note items
        self.chord_extractor = ChordExtractor(self.midi_config)
        self.chord_items = self.chord_extractor.extract(self.note_items)

    def _apply_augmentation(
        self, shift: int, stretch: float
    ) -> Tuple[List[Item], List[Item], List[Item]]:
        """Apply pitch shift and tempo stretching to the note, tempo and chord items.

        Args:
            shift (int):
                The amount to shift the pitch of the notes.
                Positive values shift the pitch up, while negative values shift it down.
            stretch (float):
                The factor by which to stretch or shrink the tempo.
                Values greater than 1.0 will speed up the tempo,
                while values less than 1.0 will slow it down.

        Returns:
            Tuple[List[Item], List[Item], List[Item]]:
                A tuple of a list of note items
                and a list of tempo items and a list of chord items.
        """
        note_items, chord_items = self._apply_shift(shift)
        tempo_items = self._apply_stretch(stretch)

        return note_items, tempo_items, chord_items

    def _apply_shift(self, shift: int) -> Tuple[List[Item], List[Item]]:
        """Apply pitch shift to the note and chord items.

        If the shift is 0, it returns the copied note and chord items without any changes.

        Args:
            shift (int): The amount to shift the pitch of the notes and chords.

        Returns:
            Tuple[List[Item], List[Item]]:
                A tuple of a list of shifted note items and a list of shifted chord items.
        """
        note_items = copy.deepcopy(self.note_items)
        chord_items = copy.deepcopy(self.chord_items)
        if shift == 0:
            return note_items, chord_items
        for note_item in note_items:
            if isinstance(note_item.pitch, int):
                note_item.pitch = note_item.pitch + shift
        chord_items = self.chord_extractor.transpose_items(chord_items, shift)
        return note_items, chord_items

    def _apply_stretch(self, stretch: float) -> List[Item]:
        """Apply tempo stretching to the tempo items.

        If the stretch is 1.0, it returns the copied tempo items without any changes.

        Args:
            stretch (float): The factor by which to stretch or shrink the tempo.

        Returns:
            List[Item]: A list of stretched tempo items.
        """
        tempo_items = copy.deepcopy(self.tempo_items)
        if stretch == 1.0:
            return tempo_items
        for tempo_item in tempo_items:
            if isinstance(tempo_item.tempo, int):
                tempo_item.tempo = round(tempo_item.tempo * stretch)
        return tempo_items
