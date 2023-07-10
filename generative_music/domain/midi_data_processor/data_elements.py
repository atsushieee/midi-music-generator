"""A module for representing MIDI events and their properties."""
from enum import Enum
from typing import Optional


class ItemName(Enum):
    """An enumeration representing the types of MIDI events.

    Args:
        NOTE: A MIDI note event.
        TEMPO: A MIDI tempo event.
        CHORD: A MIDI chord event.
    """

    NOTE = "Note"
    TEMPO = "Tempo"
    CHORD = "Chord"


class Item:
    """A class for representing MIDI events and their properties.

    The class represents a MIDI event with attributes
    such as name, start, end, velocity, and pitch.
    It provides an interface for handling MIDI events in a structured manner.
    """

    def __init__(
        self,
        name: ItemName,
        start: int,
        end: Optional[int] = None,
        velocity: Optional[int] = None,
        pitch: Optional[int] = None,
    ):
        """Initialize a MIDI event and its properties.

        Args:
            name (ItemName): The type of the MIDI event.
            start (int): The start time of the event.
            end (Optional[int]): The end time of the event, if applicable.
            velocity (Optional[int]): The velocity of the event, if applicable.
            pitch (Optional[int]): The pitch of the event, if applicable.

        Raises:
            ValueError: If the provided name is not a valid ItemName Enum member.
        """
        if not isinstance(name, ItemName):
            raise ValueError(
                "Invalid name. Allowed values are ItemName.NOTE, ItemName.TEMPO, and ItemName.CHORD."
            )
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch
