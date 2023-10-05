"""Tests for a class that converts event IDs into music events."""
import re

import pytest

from generative_music.domain.inference.id2event_converter import \
    Id2EventConverter
from generative_music.domain.midi_data_processor.midi_representation import (
    Config, Event)
from generative_music.domain.midi_data_processor.midi_tokenization import \
    Tokenizer


class TestId2EventConverter:
    """A test class for the Id2EventConverter.

    The class is responsible for testing the Id2EventConverter's functionality,
    which includes converting event IDs into music events and updating the time attribute.
    """

    def setup_method(self):
        """Initialize the Id2EventConverter tests.

        Set up initializing the Id2EventConverter instance.
        This method is called before each test function is executed.
        """
        config = Config()
        tokenizer = Tokenizer(config)
        self.converter = Id2EventConverter(tokenizer, config)

    def test_convert_id_to_events(self):
        """Test if the event ID is correctly converted into a music event.

        This test checks if the output of the method is an instance of Event.
        """
        event_id = 0
        current_generated_bar = 0
        event = self.converter.convert_id_to_events(event_id, current_generated_bar)
        assert isinstance(event, Event)

    def test_update_time(self):
        """Test if the time attribute is correctly updated based on the position number.

        This test checks if the time attribute of the converter is correctly updated,
        and if a ValueError is raised when no number is found in the event name.
        """
        event_name = "Position_9/16"
        current_generated_bar = 2
        self.converter._update_time(event_name, current_generated_bar)
        expected_time = (
            self.converter.config.TICKS_PER_BAR * current_generated_bar
            + self.converter.config.MIN_RESOLUTION
            * (int(re.search(r"\d+", event_name).group()) - 1)
        )
        assert self.converter.time == expected_time
        with pytest.raises(ValueError):
            self.converter._update_time("Position", current_generated_bar)
