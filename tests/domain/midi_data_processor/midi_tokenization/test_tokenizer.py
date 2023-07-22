"""Tests for a class that tokenize event2id and detokenize id2event."""
from generative_music.domain.midi_data_processor.midi_representation import (
    Config, Event, EventName)
from generative_music.domain.midi_data_processor.midi_tokenization.tokenizer import \
    Tokenizer


class TestTokenizer:
    """A test class for the Tokenizer.

    The class is responsible for testing the tokenization and detokenization of events.
    It checks if the events are correctly tokenized and detokenized,
    and if the results match the expected values.
    """

    def setup_method(self):
        """Initialize the Tokenizer tests.

        Set up the test environment by creating test items
        and initializing the Config instance.
        This method is called before each test function is executed.
        """
        config = Config()
        self.tokenizer = Tokenizer(config)

    def test_tokenize(self):
        """Test if the tokenization correctly converts an Event instance to an integer."""
        event = Event(EventName.POSITION, 0, "1/16")
        event_id = self.tokenizer.tokenize(event)
        assert isinstance(event_id, int)

    def test_detokenize(self):
        """Test if the detokenization correctly converts an integer to an Event instance."""
        event_id = 42
        time = 0
        event = self.tokenizer.detokenize(event_id, time)
        assert isinstance(event, Event)
        assert event.time == time

    def test_tokenize_detokenize_consistency(self):
        """Test if the tokenization and detokenization are consistent."""
        event = Event(EventName.POSITION, 1920, "1/16")
        event_id = self.tokenizer.tokenize(event)
        detokenized_event = self.tokenizer.detokenize(event_id, event.time)
        assert event.name == detokenized_event.name
        assert event.time == detokenized_event.time
        assert event.value == detokenized_event.value
