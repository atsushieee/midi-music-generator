"""A module to convert item to event for DNN learning.

This code is based on the following implementation:
Source: https://github.com/YatingMusic/remi/blob/master/utils.py
Author: Yu-Siang (Remy) Huang
License: https://github.com/YatingMusic/remi/blob/master/LICENSE
"""
from typing import List, Union

import numpy as np

from generative_music.domain.midi_data_processor.midi_representation import (
    Config, Event, EventName, Item, ItemName)


class Item2EventConverter:
    """A class for converting Item to Event objects for subsequent learning tasks.

    This class handles the conversion of Item objects
    such as notes and other musical elements into Event objects
    that can be used for further processing, analysis, and learning tasks.
    It provides methods for analyzing item properties, such as velocity and duration,
    and maps them to appropriate event representations based on defined rules.

    The converted events are formatted in a way that makes it easier to work with
    and understand the underlying musical information,
    allowing for more effective learning and pattern recognition in subsequent tasks.
    """

    def __init__(self, items: List[Item], max_time: int, midi_config: Config):
        """Initialize the Item2EventConverter to convert item to event data.

        Args:
            items (List[Item]): A list of Item objects to be converted.
            max_time (int): The maximum time.
            midi_config (Config): The Configuration for MIDI representation.
        """
        self.items = items
        self.max_time = max_time
        self.midi_config = midi_config

    def convert_items_to_events(self) -> List[Event]:
        """Convert a list of items into a list of events.

        Returns:
            List[Event]: A list of events generated from the input items.
        """
        events = []
        n_downbeat = 0
        groups = self._group_items()
        for i in range(len(groups)):
            n_downbeat += 1
            if ItemName.NOTE not in [
                item.name for item in groups[i][1:-1] if isinstance(item, Item)
            ]:
                continue

            bar_st = groups[i][0]
            bar_et = groups[i][-1]
            events.append(
                Event(
                    name=EventName.BAR,
                    time=(n_downbeat - 1) * self.midi_config.TICKS_PER_BAR,
                )
            )
            for item in groups[i][1:-1]:
                if not isinstance(item, Item):
                    continue
                if not isinstance(bar_st, np.int64):
                    continue
                if not isinstance(bar_et, np.int64):
                    continue
                # position
                flags = np.linspace(
                    bar_st, bar_et, self.midi_config.DEFAULT_FRACTION, endpoint=False
                )
                index = np.argmin(abs(flags - item.start))
                events.append(
                    Event(
                        name=EventName.POSITION,
                        time=item.start,
                        value=f"{index + 1}/{self.midi_config.DEFAULT_FRACTION}",
                    )
                )
                if item.name == ItemName.NOTE:
                    item_events = self._create_note_events(item)
                elif item.name == ItemName.CHORD:
                    item_events = self._create_chord_events(item)
                elif item.name == ItemName.TEMPO:
                    item_events = self._create_tempo_events(item)
                events.extend(item_events)
        return events

    def _group_items(self) -> List[List[Union[int, Item]]]:
        """Group the items by downbeat (1 bar).

        Returns:
            List[List[Union[int, Item]]]:
                A list of lists, where each inner list contains items within a downbeat.
        """
        self.items.sort(key=lambda x: x.start)
        downbeats = np.arange(
            0,
            self.max_time + self.midi_config.TICKS_PER_BAR,
            self.midi_config.TICKS_PER_BAR,
        )
        groups = []
        for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
            insiders = []
            for item in self.items:
                if (item.start >= db1) and (item.start < db2):
                    insiders.append(item)
            overall = [db1] + insiders + [db2]
            groups.append(overall)
        return groups

    def _create_note_events(self, item: Item) -> List[Event]:
        """Create note events from the given item.

        Args:
            item (Item): An item with name ItemName.NOTE.

        Returns:
            List[Event]: A list of note events created from the input item.
        """
        note_events = []
        # velocity
        velocity = item.velocity if item.velocity is not None else 0
        velocity_index = int(
            np.searchsorted(
                self.midi_config.DEFAULT_VELOCITY_BINS, velocity, side="right"
            )
            - 1
        )
        note_events.append(
            Event(name=EventName.NOTE_VELOCITY, time=item.start, value=velocity_index)
        )
        # pitch
        note_events.append(
            Event(name=EventName.NOTE_ON, time=item.start, value=item.pitch)
        )
        # duration
        duration = 0 if item.end is None else item.end - item.start
        index = np.argmin(abs(self.midi_config.DEFAULT_DURATION_BINS - duration))
        note_events.append(
            Event(name=EventName.NOTE_DURATION, time=item.start, value=int(index))
        )
        return note_events

    def _create_chord_events(self, item: Item) -> List[Event]:
        """Create chord events from the given item.

        Args:
            item (Item): An item with name ItemName.CHORD.

        Returns:
            List[Event]: A list of chord events created from the input item.
        """
        chord_events = []
        chord_events.append(
            Event(name=EventName.CHORD, time=item.start, value=item.pitch)
        )
        return chord_events

    def _create_tempo_events(self, item: Item) -> List[Event]:
        """Create tempo events from the given item.

        Args:
            item (Item): An item with name ItemName.TEMPO.

        Returns:
            List[Event]: A list of tempo events created from the input item.
        """
        if item.tempo is None:
            return []
        tempo_events = []
        tempo = item.tempo
        # less DEFAULT_TEMPO_INTERVALS[0].start or more DEFAULT_TEMPO_INTERVALS[2].stop
        # is cut off
        if tempo in self.midi_config.DEFAULT_TEMPO_INTERVALS[0]:
            tempo_style = Event(
                EventName.TEMPO_CLASS,
                item.start,
                self.midi_config.DEFAULT_TEMPO_NAMES[0],
            )
            tempo_value = Event(
                EventName.TEMPO_VALUE,
                item.start,
                tempo - self.midi_config.DEFAULT_TEMPO_INTERVALS[0].start,
            )
        elif tempo in self.midi_config.DEFAULT_TEMPO_INTERVALS[1]:
            tempo_style = Event(
                EventName.TEMPO_CLASS,
                item.start,
                self.midi_config.DEFAULT_TEMPO_NAMES[1],
            )
            tempo_value = Event(
                EventName.TEMPO_VALUE,
                item.start,
                tempo - self.midi_config.DEFAULT_TEMPO_INTERVALS[1].start,
            )
        elif tempo in self.midi_config.DEFAULT_TEMPO_INTERVALS[2]:
            tempo_style = Event(
                EventName.TEMPO_CLASS,
                item.start,
                self.midi_config.DEFAULT_TEMPO_NAMES[2],
            )
            tempo_value = Event(
                EventName.TEMPO_VALUE,
                item.start,
                tempo - self.midi_config.DEFAULT_TEMPO_INTERVALS[2].start,
            )
        elif tempo < self.midi_config.DEFAULT_TEMPO_INTERVALS[0].start:
            tempo_style = Event(
                EventName.TEMPO_CLASS,
                item.start,
                self.midi_config.DEFAULT_TEMPO_NAMES[0],
            )
            tempo_value = Event(EventName.TEMPO_VALUE, item.start, 0)
        elif tempo > self.midi_config.DEFAULT_TEMPO_INTERVALS[2].stop:
            tempo_style = Event(
                EventName.TEMPO_CLASS,
                item.start,
                self.midi_config.DEFAULT_TEMPO_NAMES[2],
            )
            tempo_value = Event(
                EventName.TEMPO_VALUE,
                item.start,
                self.midi_config.DEFAULT_TEMPO_INTERVALS[2].stop
                - self.midi_config.DEFAULT_TEMPO_INTERVALS[2].start
                - 1,
            )
        tempo_events.append(tempo_style)
        tempo_events.append(tempo_value)
        return tempo_events
