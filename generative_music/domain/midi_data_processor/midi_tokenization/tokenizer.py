"""A module to tokenize event2id and detokenize id2event for DNN learning."""
import json

from generative_music.domain.midi_data_processor.midi_representation import (
    Config, Event, EventName)


class Tokenizer:
    """This class is responsible for tokenizing and detokenizing events.

    The Tokenizer maps events to unique integer IDs and vice versa.
    It allows for easy conversion between events and their corresponding IDs,
    which is useful for machine learning models that require numerical inputs.
    """

    def __init__(self, config: Config):
        """Initialize the tokenizer with token mapping from the given Config instance.

        Args:
            config (Config):
                An instance of the Config class containing token mapping file paths.
        """
        with open(config.EVENT2ID_FILEPATH, "r") as f:
            self.event2id = json.load(f)

        with open(config.ID2EVENT_FILEPATH, "r") as f:
            self.id2event = json.load(f)

    def tokenize(self, event: Event) -> int:
        """Convert a given event to its corresponding ID.

        Args:
            event (Event): The event to be tokenized.

        Returns:
            int: The ID corresponding to the given event.

        Raises:
            KeyError: If the given event is not found in the event2id mapping.
        """
        event_key = f"{event.name.value}_{event.value}"
        if event_key not in self.event2id:
            raise KeyError(
                f"The event '{event_key}' is not found in the event2id mapping."
            )
        return self.event2id[event_key]

    def detokenize(self, id: int, time: int) -> Event:
        """Convert a given ID to its corresponding event.

        Args:
            id (int): The ID to be detokenized.
            time (int): starting event time (tick).

        Returns:
            Event: The event corresponding to the given ID.

        Raises:
            KeyError: If the given ID is not found in the id2event mapping.
        """
        id_str = str(id)
        if id_str not in self.id2event:
            raise KeyError(f"The ID '{id}' is not found in the id2event mapping.")
        event_str = self.id2event[id_str]
        name, value = event_str.split("_", 1)
        return Event(name=EventName(name), time=time, value=value)
