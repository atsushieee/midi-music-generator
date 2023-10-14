"""A module for extracting chords from midi data.

This code is based on the following implementation:
Source: https://github.com/YatingMusic/remi/blob/master/chord_recognition.py
Author: Yu-Siang (Remy) Huang
License: https://github.com/YatingMusic/remi/blob/master/LICENSE
"""
from copy import deepcopy
from typing import Dict, List, Tuple

import miditoolkit
import numpy as np

from generative_music.domain.midi_data_processor.midi_representation import (
    Config, Item, ItemName)


# [TODO] Implement a new logic for the ChordExtractor class
class ChordExtractor:
    """A class for extracting chords from MIDI files.

    This class handles the extraction of chord information from notes data.
    It provides methods for analyzing note items
    and determining the most likely chords based on the notes being played.
    The extracted chords are formatted into a human-readable format
    for easier understanding and further analysis.
    """

    def __init__(self, midi_config: Config):
        """Initialize the ChordExtractor for extracting chords from midi data.

        Args:
            midi_config (Config): The Configuration for MIDI representation.
        """
        self.midi_config = midi_config

    def extract(self, notes: List[Item]) -> List[Item]:
        """Extract chords from a list of notes with a given resolution.

        This method reads the notes, converts them to a pianoroll representation
        and explores chord possibilities with different window widths.
        It then selects the best chords using a greedy algorithm
        and returns a list of chord items.

        Args:
            notes (List[Item]): A list of note items to extract chords from.

        Returns:
            List[Item]: A list of chord items extracted from the input notes.
        """
        # read
        max_tick = max([0 if n.end is None else n.end for n in notes])
        pianoroll = self._note2pianoroll(
            notes=notes,
            max_tick=max_tick,
            ticks_per_beat=self.midi_config.CHORD_RESOLUTION,
        )
        # get lots of candidates
        candidates: Dict[int, Dict[int, Tuple[str, str, str, int]]] = {}
        # the shortest: 2 beat, longest: 4 beat
        # Explore chord possibilities w ith 4 (1 bar) or 2 (1/2 bar) window widths
        for interval in [4, 2]:
            for start_tick in range(0, max_tick, self.midi_config.CHORD_RESOLUTION):
                # set target pianoroll
                end_tick = int(
                    self.midi_config.CHORD_RESOLUTION * interval + start_tick
                )
                if end_tick > max_tick:
                    end_tick = max_tick
                _pianoroll = pianoroll[start_tick:end_tick, :]
                # find chord
                root_note, quality, bass_note, score = self._find_chord(
                    pianoroll=_pianoroll
                )
                # save
                if start_tick not in candidates:
                    candidates[start_tick] = {}
                    candidates[start_tick][end_tick] = (
                        root_note,
                        quality,
                        bass_note,
                        score,
                    )
                else:
                    if end_tick not in candidates[start_tick]:
                        candidates[start_tick][end_tick] = (
                            root_note,
                            quality,
                            bass_note,
                            score,
                        )
        # greedy
        chords = self._select_chords_greedily(candidates=candidates, max_tick=max_tick)
        # convert to CHORD Items
        output = []
        for chord in chords:
            # chordのbass音は省略して、append
            output.append(
                Item(
                    name=ItemName.CHORD,
                    start=chord[0],
                    end=chord[1],
                    pitch=chord[2].split("/")[0],
                )
            )
        return output

    def transpose_items(self, chord_items: List[Item], shift: int) -> List[Item]:
        """Transpose the pitch of the chord items based on the shift value.

        Args:
            chord_items (List[Item]): The list of chord items to be transposed.
            shift (int): The amount to shift the pitch of the chords.

        Returns:
            List[Item]: A list of transposed chord items.
        """
        transposed_items = deepcopy(chord_items)
        for item in transposed_items:
            if not isinstance(item.pitch, str):
                continue
            # Split the pitch into note and chord type
            note, chord_type = item.pitch.split(":")
            # Skip if note is 'N'
            if note == "N":
                continue
            # Get the index of the current note in the pitch classes
            index = self.midi_config.PITCH_CLASSES.index(note)
            # Calculate the new index by adding the shift and wrapping around the list length
            new_index = (index + shift) % len(self.midi_config.PITCH_CLASSES)
            # Update the pitch of the item
            item.pitch = self.midi_config.PITCH_CLASSES[new_index] + ":" + chord_type
        return transposed_items

    def _note2pianoroll(
        self, notes: List[Item], max_tick: int, ticks_per_beat: int
    ) -> np.ndarray:
        """Convert a list of note events to a piano roll representation.

        Args:
            notes (List[Item]): A list of note items.
            max_tick (int): The maximum tick value in the note items.
            ticks_per_beat (int): The number of ticks per beat in the MIDI file.

        Returns:
            np.ndarray:
                A piano roll representation of the input note events
                with size (max_ticks, 128).
                Each element represents the velocity of the corresponding note
                at a given time.
        """
        return miditoolkit.pianoroll.parser.notes2pianoroll(
            note_stream_ori=notes, max_tick=max_tick, ticks_per_beat=ticks_per_beat
        )

    def _calculate_relative_pitch_sequences(
        self, chroma: np.ndarray
    ) -> Dict[int, List[int]]:
        """Find the relative pitch sequence for each root note in the chroma array.

        This method calculates the relative pitch sequence for each of the 12 tones,
        using a given tone as the base (0)
        and determining the positions of the other sounding notes in the chroma array.

        Args:
            chroma (np.ndarray):
                A chroma array representing the presence (1) or absence (0)
                of each of the 12 pitches.

        Returns:
            Dict[int, List[int]]:
                A dictionary mapping each root note to its relative pitch sequence.
        """
        candidates = {}
        for index in range(len(chroma)):
            if chroma[index]:
                root_note = index
                _chroma = np.roll(chroma, -root_note)
                sequence = np.where(_chroma == 1)[0]
                candidates[root_note] = list(sequence)
        return candidates

    def _evaluate_chord_scores_and_qualities(
        self, candidates: Dict[int, List[int]]
    ) -> Tuple[Dict[int, int], Dict[int, str]]:
        """Calculate the scores and qualities for each root note and its corresponding sequence.

        This method evaluates the scores and qualities of each candidate root note
        and its corresponding sequence based on the presence or absence of specific intervals.

        Args:
            candidates (Dict[int, List[int]]):
                A dictionary mapping each root note to its relative pitch sequence.

        Returns:
            Tuple[Dict[int, int], Dict[int, str]]:
                Two dictionaries containing the scores and qualities for each root note.
        """
        scores = {}
        qualities = {}
        for root_note, sequence in candidates.items():
            if 3 not in sequence and 4 not in sequence:
                scores[root_note] = -100
                qualities[root_note] = "None"
            elif 3 in sequence and 4 in sequence:
                scores[root_note] = -100
                qualities[root_note] = "None"
            else:
                # decide quality
                if 3 in sequence:
                    if 6 in sequence:
                        quality = "dim"
                    else:
                        quality = "min"
                elif 4 in sequence:
                    if 8 in sequence:
                        quality = "aug"
                    else:
                        if 7 in sequence and 10 in sequence:
                            quality = "dom"
                        else:
                            quality = "maj"
                # decide score
                maps = self.midi_config.CHORD_MAPS.get(quality)
                if maps is None:
                    continue
                _notes = [n for n in sequence if n not in maps]
                score = 0
                for n in _notes:
                    outsiders_1 = self.midi_config.CHORD_OUTSIDERS_1.get(quality, [])
                    outsiders_2 = self.midi_config.CHORD_OUTSIDERS_2.get(quality, [])
                    insiders = self.midi_config.CHORD_INSIDERS.get(quality, [])

                    if n in outsiders_1:
                        score -= 1
                    elif n in outsiders_2:
                        score -= 2
                    elif n in insiders:
                        score += 1
                scores[root_note] = score
                qualities[root_note] = quality
        return scores, qualities

    def _find_chord(self, pianoroll: np.ndarray) -> Tuple[str, str, str, int]:
        """Find the chord information from a pianoroll representation.

        This method analyzes the given pianoroll
        to determine the chord's root note, quality, bass note and score.

        Args:
            pianoroll (np.ndarray):
                A piano roll representation of the input note events
                with size (max_ticks, 128).
                Each element represents the velocity of the corresponding note
                at a given time.

        Returns:
            Tuple[str, str, str, int]:
                A tuple containing the root note, quality, bass note
                and score of the chord.
        """
        # Number of ticks * 12 (number of notes in one octave) array
        chroma = miditoolkit.pianoroll.utils.tochroma(pianoroll=pianoroll)
        # 12 one-dimensional arrays
        chroma = np.sum(chroma, axis=0)
        # Throw away all the information about the velocity and time.
        # I'm not sure this is a good idea!?
        chroma = np.array([1 if c else 0 for c in chroma])
        if np.sum(chroma) == 0:
            return "N", "N", "N", 0
        else:
            candidates = self._calculate_relative_pitch_sequences(chroma=chroma)
            scores, qualities = self._evaluate_chord_scores_and_qualities(
                candidates=candidates
            )
            # bass note
            sorted_notes = []
            for i, v in enumerate(np.sum(pianoroll, axis=0)):
                if v > 0:
                    sorted_notes.append(int(i % 12))
            bass_note = sorted_notes[0]
            # root note
            __root_note = []
            _max = max(scores.values())
            for _root_note, score in scores.items():
                if score == _max:
                    __root_note.append(_root_note)
            if len(__root_note) == 1:
                root_note = __root_note[0]
            else:
                # TODO: what should i do
                for n in sorted_notes:
                    if n in __root_note:
                        root_note = n
                        break
            # quality
            quality = qualities.get(root_note, "")
            # score
            score = scores.get(root_note, -100)
            return (
                self.midi_config.PITCH_CLASSES[root_note],
                quality,
                self.midi_config.PITCH_CLASSES[bass_note],
                score,
            )

    def _select_chords_greedily(
        self, candidates: Dict[int, Dict[int, Tuple[str, str, str, int]]], max_tick: int
    ) -> List[Tuple[int, int, str]]:
        """Select chords greedily based on the highest score at each start time.

        This method iterates through the candidates and selects the best chord at each tick
        based on score and duration.
        It also filters out any ":None" chords and adjusts the duration accordingly.

        The process follows these steps:
        1. Select the chord with the maximum score at the current start time.
        2. Set the end time of the selected chord as the new start time and repeat.

        Args:
            candidates (Dict[int, Dict[int, Tuple[str, str, str, int]]]):
                A dictionary containing chord candidates for each tick.
            max_tick (int): The maximum tick value to consider when selecting chords.

        Returns:
            List[Tuple[int, int, str]]:
                A list of selected chords
                with their start tick, end tick and chord information.
        """
        chords: List[Tuple[int, int, str]] = []
        # start from 0
        start_tick = 0
        while start_tick < max_tick:
            _candidates = candidates.get(start_tick, {})
            _candidates_items = sorted(
                _candidates.items(), key=lambda x: (x[1][-1], x[0])
            )
            # choose
            end_tick, (root_note, quality, bass_note, _) = _candidates_items[-1]
            if root_note == bass_note:
                chord = f"{root_note}:{quality}"
            else:
                chord = f"{root_note}:{quality}/{bass_note}"
            chords.append((start_tick, end_tick, chord))
            start_tick = end_tick
        # remove :None
        temp = chords
        while temp and isinstance(temp[0][-1], str) and ":None" in temp[0][-1]:
            try:
                temp[1] = (temp[0][0], temp[1][1], temp[1][2])
                del temp[0]
            except IndexError:
                print("NO CHORD")
                return []
        temp2 = []
        for chord_info in temp:
            if isinstance(chord_info[-1], str) and ":None" not in chord_info[-1]:
                temp2.append(chord_info)
            else:
                temp2[-1] = (temp2[-1][0], chord_info[1], temp2[-1][2])
        return temp2
