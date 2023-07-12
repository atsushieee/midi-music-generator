"""A module for loading midi data.

This code is based on the following implementation:
Source: https://github.com/YatingMusic/remi/blob/master/utils.py
Author: Yu-Siang (Remy) Huang
License: https://github.com/YatingMusic/remi/blob/master/LICENSE
"""

from typing import List, Optional

import miditoolkit
import numpy as np

from generative_music.domain.midi_data_processor.data_elements import (
    Item, ItemName)


class DataLoader:
    """A class for loading and processing MIDI files.

    This class handles the loading and processing of MIDI files
    using the miditoolkit library.
    It provides methods for reading note and tempo information.
    The tempo information is also formatted
    into an appropriate format for easier handling and analysis.
    """

    def __init__(self, file_path, resolution=480):
        """Initialize the DataLoader with the given file path and resolution.

        Args:
            file_path (str): The path to the MIDI file to be loaded.
            resolution (int):
                The resolution (ticks per beat) of the MIDI file. Default is 480.
        """
        self.file_path = file_path
        self.midi_obj = miditoolkit.midi.parser.MidiFile(self.file_path)
        self.resolution = resolution

    def read_note_items(self, track: Optional[int] = None) -> List[Item]:
        """Read note items from the MIDI file.

        This method reads note events from the MIDI file
        for a specified track or for all combined tracks if no track is specified.
        The purpose of this method is to read and extract note data from the MIDI file,
        which can be used for further analysis and processing based on music theory.

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
            return self._create_note_items(combined_notes)

        # Read notes from the specified track
        if track < 0 or track >= len(self.midi_obj.instruments):
            raise ValueError(
                f"Invalid track number: {track}. Available tracks: 0 to {len(self.midi_obj.instruments) - 1}"
            )
        notes = self.midi_obj.instruments[track].notes
        return self._create_note_items(notes)

    def read_tempo_items(self) -> List[Item]:
        """Read tempo items from the MIDI file and generate tempo items at self.resolution.

        This method reads tempo changes from the MIDI file
        and generates tempo items at the specified resolution
        using the _process_tempo_items method.

        Returns:
            List[Item]: A list of tempo items with the specified resolution.
        """
        tempo_changes = self.midi_obj.tempo_changes
        tempo_items = [
            Item(name=ItemName.TEMPO, start=tempo.time, tempo=int(tempo.tempo))
            for tempo in tempo_changes
        ]
        return self._process_tempo_items(tempo_items)

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
        print(type(notes[0]))
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

    def _process_tempo_items(self, tempo_items: List[Item]) -> List[Item]:
        """Process tempo items to generate tempo items at a specific resolution.

        This method takes a list of tempo items
        and generates a new list of tempo items at the specified resolution.
        The purpose of this method is to preprocess tempo data,
        transforming it into an appropriate format
        for further analysis and processing based on music theory.

        Args:
            tempo_items (List[Item]):
                A list of tempo items read from the MIDI file.

        Returns:
            List[Item]: A list of tempo items with the specified resolution.
        """
        max_tick = tempo_items[-1].start
        if max_tick % self.resolution != 0:
            max_tick = ((max_tick // self.resolution) + 1) * self.resolution
        wanted_ticks = np.arange(0, max_tick + 1, self.resolution)
        output = []
        current_tempo_item_index = 0
        for tick in wanted_ticks:
            while (current_tempo_item_index + 1) < len(
                tempo_items
            ) and tick >= tempo_items[current_tempo_item_index + 1].start:
                current_tempo_item_index += 1
            if (current_tempo_item_index + 1) < len(tempo_items) and abs(
                tick - tempo_items[current_tempo_item_index + 1].start
            ) < self.resolution // 2:
                current_tempo = tempo_items[current_tempo_item_index + 1].tempo
            else:
                current_tempo = tempo_items[current_tempo_item_index].tempo
            output.append(
                Item(
                    name=ItemName.TEMPO,
                    start=tick,
                    tempo=current_tempo,
                )
            )
        return output
