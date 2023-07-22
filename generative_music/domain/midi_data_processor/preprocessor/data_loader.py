"""A module for loading midi data.

This code is based on the following implementation:
Source: https://github.com/YatingMusic/remi/blob/master/utils.py
Author: Yu-Siang (Remy) Huang
License: https://github.com/YatingMusic/remi/blob/master/LICENSE
"""

from typing import List, Optional

import miditoolkit
import numpy as np

from generative_music.domain.midi_data_processor.midi_representation import (
    Item, ItemName)


class DataLoader:
    """A class for loading and processing MIDI files.

    This class handles the loading and processing of MIDI files
    using the miditoolkit library.
    It provides methods for reading note and tempo information.
    The note and tempo information is also formatted
    into an appropriate format for easier handling and analysis.
    """

    def __init__(self, file_path, note_resolution=120, tempo_resolution=480):
        """Initialize the DataLoader with the given file path and resolution.

        Args:
            file_path (str): The path to the MIDI file to be loaded.
            note_resolution (int):
                The resolution (ticks per beat) of the note of MIDI file.
                Default is 120.
            tempo_resolution (int):
                The resolution (ticks per beat) of the tempo of MIDI file.
                Default is 480.
        """
        self.file_path = file_path
        self.midi_obj = miditoolkit.midi.parser.MidiFile(self.file_path)
        self.note_resolution = note_resolution
        self.tempo_resolution = tempo_resolution

    def read_note_items(self, track: Optional[int] = None) -> List[Item]:
        """Read and quantize note items from the MIDI file.

        This method reads note events from the MIDI file
        for a specified track or for all combined tracks if no track is specified.
        The note items are then quantized to align them to the specified grid resolution.
        The purpose of this method is to read and extract
        and preprocess note data from the MIDI file, which can be used
        for facilitating subsequent data processing and machine learning tasks.

        Args:
            track (int, optional):
                The track number to read note events from.
                If None, all tracks will be combined. Defaults to None.

        Returns:
            List[Item]:
                A list of note items from the specified track or all combined tracks.

        Raises:
            ValueError: If the specified track number is invalid.
        """
        if track is None:
            # Combine all tracks
            combined_notes = []
            for instrument in self.midi_obj.instruments:
                combined_notes.extend(instrument.notes)
            note_items = self._create_note_items(combined_notes)
        else:
            # Read notes from the specified track
            if track < 0 or track >= len(self.midi_obj.instruments):
                raise ValueError(
                    f"Invalid track number: {track}. Available tracks: 0 to {len(self.midi_obj.instruments) - 1}"
                )
            notes = self.midi_obj.instruments[track].notes
            note_items = self._create_note_items(notes)

        # Quantize note_items before returning
        return self._quantize_items(note_items, self.note_resolution)

    def read_tempo_items(self) -> List[Item]:
        """Read and quantize tempo changes items from the MIDI file.

        This method reads tempo changes from the MIDI file
        and quantizes the tempo items to align them to the specified grid resolution.
        The new tempo items will have values at each resolution step,
        regardless of whether the tempo has changed or not. This is important
        because it allows the model to learn that the tempo remains constant
        in sections where it does not change.
        The purpose of this method is to read, extract
        and preprocess tempo data from the MIDI file, which can be used
        for facilitating subsequent data processing and machine learning tasks.

        Returns:
            List[Item]:
                A list of items quantized to the specified grid resolution,
                with tempo values at each resolution step.
        """
        tempo_changes = self.midi_obj.tempo_changes
        tempo_items = [
            Item(name=ItemName.TEMPO, start=tempo.time, tempo=int(tempo.tempo))
            for tempo in tempo_changes
        ]
        # Quantize note_items before returning
        quantized_tempo = self._quantize_items(tempo_items, self.tempo_resolution)
        return self._process_tempo_items(quantized_tempo)

    def _create_note_items(
        self, notes: List[miditoolkit.midi.containers.Note]
    ) -> List[Item]:
        """Create note items from the given note events.

        Args:
            notes (List[miditoolkit.midi.containers.Note]):
                A list of note events read from the MIDI file using miditoolkit.

        Returns:
            List[Item]: A list of note items created from the note events.
        """
        notes.sort(key=lambda x: (x.start, x.pitch))
        note_items = [
            Item(
                name=ItemName.NOTE,
                start=note.start,
                end=note.end,
                velocity=note.velocity,
                pitch=note.pitch,
            )
            for note in notes
        ]
        return note_items

    def _quantize_items(self, items: List[Item], resolution: int) -> List[Item]:
        """Quantize items to align them to a specified grid resolution.

        This method takes a list of items and generates a new list of items
        aligned to the specified grid resolution.
        The quantization process adjusts the start and end positions of the items,
        making them fit into a grid with the given resolution.
        This is useful for facilitating subsequent data processing
        and machine learning tasks.

        Args:
            items (List[Item]): A list of items read from the MIDI file.
            resolution (int): The number of the grid resolution (tick).

        Returns:
            List[Item]:
                A list of items quantized to the specified grid resolution.
        """
        max_tick = items[-1].start
        if max_tick % resolution != 0:
            max_tick = ((max_tick // resolution) + 1) * resolution
        # grid
        grids = np.arange(0, max_tick + 1, resolution, dtype=int)
        # process
        for item in items:
            index = np.argmin(abs(grids - item.start))
            shift = grids[index] - item.start
            item.start += shift
            if item.end is not None:
                item.end += shift
        return items

    def _process_tempo_items(self, tempo_items: List[Item]) -> List[Item]:
        """Process tempo items to generate tempo items at a specific resolution.

        This method takes a list of tempo items
        and generates a new list of tempo items
        at the specified resolution.
        The new tempo items will have values at each resolution step,
        regardless of whether the tempo has changed or not.
        This is important because it allows the model to learn
        that the tempo remains constant in sections where it does not change.

        Args:
            tempo_items (List[Item]):
                A list of tempo items read from the MIDI file.

        Returns:
            List[Item]:
                A list of items quantized to the specified grid resolution,
                with tempo values at each resolution step.
        """
        max_tick = tempo_items[-1].start
        wanted_ticks = np.arange(0, max_tick + 1, self.tempo_resolution)
        output = []
        current_tempo_item_index = 0
        for tick in wanted_ticks:
            while (current_tempo_item_index + 1) < len(
                tempo_items
            ) and tick >= tempo_items[current_tempo_item_index + 1].start:
                current_tempo_item_index += 1

            output.append(
                Item(
                    name=ItemName.TEMPO,
                    start=tick,
                    tempo=tempo_items[current_tempo_item_index].tempo,
                )
            )
        return output
