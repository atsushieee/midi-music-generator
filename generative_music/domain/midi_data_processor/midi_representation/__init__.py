"""Midi data representation package.

This package contains various modules and classes for MIDI data representation,
including MIDI items, events and configuration for music generation models.
"""

from generative_music.domain.midi_data_processor.midi_representation.config import \
    Config
from generative_music.domain.midi_data_processor.midi_representation.data_elements import (
    Event, EventName, Item, ItemName)

__all__ = ["ItemName", "Item", "EventName", "Event", "Config"]
