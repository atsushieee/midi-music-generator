"""A module for representing MIDI items and events dor DNN learning."""
from enum import Enum
from typing import Optional, Union


class ItemName(Enum):
    """An enumeration representing different types of MIDI items."""

    NOTE = "Note"
    TEMPO = "Tempo"
    CHORD = "Chord"


class Item:
    """A class for representing MIDI items and their properties.

    The class represents a MIDI item with attributes
    such as name, start, end, velocity and pitch.
    It provides an interface for handling MIDI events in a structured manner.
    """

    def __init__(
        self,
        name: ItemName,
        start: int,
        end: Optional[int] = None,
        velocity: Optional[int] = None,
        pitch: Optional[Union[int, str]] = None,
        tempo: Optional[int] = None,
    ):
        """Initialize a MIDI item and its properties.

        Args:
            name (ItemName): The type of the MIDI item.
            start (int): The start time of the item.
            end (Optional[int]): The end time of the item, if applicable.
            velocity (Optional[int]): The velocity of the item, if applicable.
            pitch (Optional[int]): The pitch of the item, if applicable.
            tempo (Optional[int]): The tempo of the item, if applicable.

        Raises:
            ValueError: If the provided name is not a valid ItemName.
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
        self.tempo = tempo

    def __repr__(self) -> str:
        """Return a string representation of the Item instance.

        Returns:
            str: A string representation of the Item instance.
        """
        return f"""
            Item(name={self.name}, start={self.start}, end={self.end},
            velocity={self.velocity}, pitch={self.pitch}, tempo={self.tempo})
        """


class EventName(Enum):
    """An enumeration representing different types of events."""

    BAR = "Bar"
    POSITION = "Position"
    NOTE_VELOCITY = "Note Velocity"
    NOTE_ON = "Note On"
    NOTE_DURATION = "Note Duration"
    CHORD = "Chord"
    TEMPO_CLASS = "Tempo Class"
    TEMPO_VALUE = "Tempo Value"


class Event:
    """A class for representing MIDI items based on predefined rules for learning purposes.

    The class represents a MIDI item with attributes such as name, time, value and text.
    It provides an interface for handling MIDI events in a structured manner
    following a set of predetermined rules for learning purposes.
    """

    def __init__(
        self,
        name: EventName,
        time: int,
        value: Optional[Union[int, str]] = None,
    ):
        """Initialize an Event instance with the given name, time, value and text.

        Args:
            name (EventName): The name of the event.
            time (int): The time at which the event occurs. Defaults to None.
            value (Optional[Union[int, str]]):
                A numeric or string value associated with the event.
                Defaults to None.

        Raises:
            ValueError: If the provided name is not a valid EventName.
        """
        if not isinstance(name, EventName):
            raise ValueError("Invalid name. Allowed values are enum items.")
        self.name = name
        self.time = time
        self.value = value

    def __repr__(self) -> str:
        """Return a string representation of the Event instance.

        Returns:
            str: A string representation of the Event instance.
        """
        return f"Event(name={self.name}, time={self.time}, value={self.value})"
