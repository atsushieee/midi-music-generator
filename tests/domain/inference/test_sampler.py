"""Tests for a class that generates initial event IDs and samples event IDs with temperature."""
import numpy as np

from generative_music.domain.inference.sampler import Sampler
from generative_music.domain.midi_data_processor.midi_representation import \
    Config
from generative_music.domain.midi_data_processor.midi_tokenization import \
    Tokenizer


class TestSampler:
    """A test class for the Sampler.

    The class is responsible for testing the Sampler's functionality,
    which includes generating initial event IDs and sampling with temperature.
    """

    def setup_method(self):
        """Initialize the Sampler tests.

        Set up initializing the Sampler instance.
        This method is called before each test function is executed.
        """
        tokenizer = Tokenizer(Config())
        self.sampler = Sampler(tokenizer)

    def test_generate_initial_event_ids(self):
        """Test if the initial event IDs are correctly generated and are integers.

        This test checks if the output of the 'generate_initial_event_ids' method is a list,
        and if all elements in the list are instances of the integer type.
        """
        event_ids = self.sampler.generate_initial_event_ids()
        assert isinstance(event_ids, list)
        assert all(isinstance(id_, int) for id_ in event_ids)

    def test_sample_with_temperature(self):
        """Test if the method correctly samples an event ID.

        This test checks if the sampled event ID is an integer,
        and is consistent with the expected output given the fixed random seed.
        """
        logits = np.array([0.1, 0.2, 0.7])
        temperature = 0.5
        topk = 2
        # Fix Seed to allow random elements to be reproduced.
        np.random.seed(0)
        event_id = self.sampler.sample_with_temperature(logits, temperature, topk)
        assert isinstance(event_id, int)
        assert event_id == 2
