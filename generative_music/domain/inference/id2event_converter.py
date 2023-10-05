"""A module to convert event IDs into music events for music creation."""
import re

from generative_music.domain.midi_data_processor.midi_representation import (
    Config, Event)
from generative_music.domain.midi_data_processor.midi_tokenization import \
    Tokenizer


class Id2EventConverter:
    """This class is responsible for converting event IDs into music events.

    The Id2EventConverter uses a Tokenizer to convert event IDs into their corresponding events,
    and implements methods for updating the time attribute based on the event's position.
    """

    def __init__(self, tokenizer: Tokenizer, config: Config):
        """Initialize Id2EventConverter.

        Args:
            tokenizer (Tokenizer): An instance of Tokenizer to perform tokenization.
            config (Config): The Configuration for MIDI representation.
        """
        self.tokenizer = tokenizer
        self.config = config
        self.time = 0

    def convert_id_to_events(self, event_id: int, current_generated_bar: int) -> Event:
        """Convert an event ID into a music event.

        Args:
            event_id (int): The ID of the event to be converted.
            current_generated_bar (int): The current bar that is being generated.

        Returns:
            Event: The event corresponding to the given ID.
        """
        event_name = self.tokenizer.id2event[str(event_id)]
        if "Position" in event_name:
            self._update_time(event_name, current_generated_bar)
        return self.tokenizer.detokenize(event_id, self.time)

    def _update_time(self, event_name: str, current_generated_bar: int):
        """Update the current time based on the position number in the event name.

        Args:
            event_name (str): The name of the event.
            current_generated_bar (int): The current bar that is being generated.

        Raises:
            ValueError: If no number is found in the event name.
        """
        match = re.search(r"\d+", event_name)
        if match:
            position_num = int(match.group())
        else:
            raise ValueError("No number found in the provided string")
        self.time = (
            self.config.TICKS_PER_BAR * current_generated_bar
            + self.config.MIN_RESOLUTION * (position_num - 1)
        )
