"""A module to write midi data based on events info.

This code is based on the following implementation:
Source: https://github.com/YatingMusic/remi/blob/master/utils.py
Author: Yu-Siang (Remy) Huang
License: https://github.com/YatingMusic/remi/blob/master/LICENSE
"""
from pathlib import Path
from typing import List, Tuple

import miditoolkit

from generative_music.domain.midi_data_processor.midi_representation import (
    Config, Event, EventName)


class DataWriter:
    """This class is responsible for writing MIDI data from events.

    The DataWriter takes a list of Event instances representing a musical composition
    and writes a MIDI file with the corresponding notes, chords, and tempos.
    """

    def __init__(self, midi_config: Config):
        """Initialize the DataWriter class.

        Args:
            midi_config (Config): The Configuration for MIDI representation.
        """
        self.midi_config = midi_config

    def write_midi_file(self, events: List[Event], output_path: Path):
        """Write a MIDI file based on the given events.

        Args:
            events (List[Event]):
                A list of Event instances representing the musical composition.
            output_path (Path): The path to the output MIDI file.
        """
        temp_notes, chords, tempos = self._process_events(events)
        notes = self._get_specific_time_for_notes(temp_notes)

        midi = miditoolkit.midi.parser.MidiFile()
        # Four Four Time signature
        midi.ticks_per_beat = int(self.midi_config.TICKS_PER_BAR / 4)
        inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
        inst.notes = notes
        midi.instruments.append(inst)

        tempo_changes = []
        for start_time, bpm in tempos:
            tempo_changes.append(
                miditoolkit.midi.containers.TempoChange(bpm, start_time)
            )
        midi.tempo_changes = tempo_changes

        for c in chords:
            midi.markers.append(
                miditoolkit.midi.containers.Marker(text=c[1], time=c[0])
            )
        midi.dump(output_path)

    def _process_events(
        self, events: List[Event]
    ) -> Tuple[
        List[Tuple[int, int, int, int]], List[Tuple[int, str]], List[Tuple[int, int]]
    ]:
        """Process the input events and extract notes, chords, and tempos.

        Args:
            events (List[Event]):
                A list of Event instances representing the musical composition.

        Returns:
            Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, str]], List[Tuple[int, int]]]:
                A tuple containing three lists:
                - notes: A list of tuples (time, velocity, pitch, duration)
                - chords: A list of tuples (time, chord)
                - tempos: A list of tuples (time, tempo)
        """
        notes: List[Tuple[int, int, int, int]] = []
        chords: List[Tuple[int, str]] = []
        tempos: List[Tuple[int, int]] = []
        # We loop until len(events) - 3 because we are checking for a sequence of 4 events
        # at a time (POSITION, NOTE_VELOCITY, NOTE_ON, NOTE_DURATION).
        # If the last event sequence is a chord,it will not form a complete sequence of 4 events
        # but we don't need to consider it for forming notes.
        for i in range(len(events) - 3):
            time = events[i].time
            if (
                events[i].name == EventName.POSITION
                and events[i + 1].name == EventName.NOTE_VELOCITY
                and events[i + 2].name == EventName.NOTE_ON
                and events[i + 3].name == EventName.NOTE_DURATION
            ):
                velocity_index = events[i + 1].value
                velocity = int(self.midi_config.DEFAULT_VELOCITY_BINS[velocity_index])
                pitch = events[i + 2].value
                if not isinstance(pitch, int):
                    continue
                duration_index = events[i + 3].value
                duration = int(self.midi_config.DEFAULT_DURATION_BINS[duration_index])
                notes.append((time, velocity, pitch, duration))
            elif (
                events[i].name == EventName.POSITION
                and events[i + 1].name == EventName.CHORD
            ):
                chord = events[i + 1].value
                if not isinstance(chord, str):
                    continue
                chords.append((time, chord))
            elif (
                events[i].name == EventName.POSITION
                and events[i + 1].name == EventName.TEMPO_CLASS
                and events[i + 2].name == EventName.TEMPO_VALUE
            ):
                tempo_class_value = events[i + 1].value
                tempo_value = events[i + 2].value
                for j, tempo_name in enumerate(self.midi_config.DEFAULT_TEMPO_NAMES):
                    if not isinstance(tempo_value, int):
                        break
                    if tempo_class_value == tempo_name:
                        tempo = (
                            self.midi_config.DEFAULT_TEMPO_INTERVALS[j].start
                            + tempo_value
                        )
                        tempos.append((time, tempo))
                        break
        return notes, chords, tempos

    def _get_specific_time_for_notes(
        self, temp_notes: List[Tuple[int, int, int, int]]
    ) -> List[miditoolkit.Note]:
        """Convert the Event class format to miditoolkit.Note instances.

        Args:
            temp_notes (List[Tuple[int, int, int, int]]):
                A list of tuples (start_time, velocity, pitch, duration)

        Returns:
            List[miditoolkit.Note]:
                A list of miditoolkit.Note instances with the specified start and end times.
        """
        notes = []
        for note in temp_notes:
            start_time, velocity, pitch, duration = note
            end_time = start_time + duration
            notes.append(miditoolkit.Note(velocity, pitch, start_time, end_time))
        return notes
