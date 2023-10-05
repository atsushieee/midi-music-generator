"""A module to generate and sample music event IDs for music creation."""
from typing import List

import numpy as np

from generative_music.domain.midi_data_processor.midi_tokenization import \
    Tokenizer


class Sampler:
    """This class is responsible for generating and sampling music events.

    The Sampler uses a Tokenizer to convert events into unique integer IDs,
    and implements methods for generating initial event IDs
    and sampling new events based on given logits.
    """

    def __init__(self, tokenizer: Tokenizer):
        """Initialize Sampler.

        Args:
            tokenizer (Tokenizer): An instance of Tokenizer to perform tokenization.
        """
        self.tokenizer = tokenizer

    def generate_initial_event_ids(self) -> List[int]:
        """Generate initial event IDs.

        Returns:
            List[int]: List of generated initial event IDs.
        """
        event_ids = [self.tokenizer.event2id["Bar_None"]]
        tempo_classes = [
            v for k, v in self.tokenizer.event2id.items() if "Tempo Class" in k
        ]
        tempo_values = [
            v for k, v in self.tokenizer.event2id.items() if "Tempo Value" in k
        ]
        chords = [v for k, v in self.tokenizer.event2id.items() if "Chord" in k]
        event_ids.append(self.tokenizer.event2id["Position_1/16"])
        event_ids.append(np.random.choice(chords))
        event_ids.append(self.tokenizer.event2id["Position_1/16"])
        event_ids.append(np.random.choice(tempo_classes))
        event_ids.append(np.random.choice(tempo_values))

        # Convert all elements in the list to int
        event_ids = [int(id_) for id_ in event_ids]
        return event_ids

    def sample_with_temperature(
        self, logits: np.ndarray, temperature: float, topk: int
    ) -> int:
        """Perform temperature sampling.

        Args:
            logits (np.ndarray):
                A 1-D numpy array of logits, which are the raw outputs (before softmax).
                Each value in the array corresponds to the log-odds
                that a particular event will be selected.
                The shape of the array is expected to be (n,),
                where n is the number of possible events.
            temperature (float): Temperature parameter for sampling.
            topk (int): Parameter to select from the top k candidates.

        Returns:
            int: ID of the event selected by sampling.
        """
        probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))

        sorted_index = np.argsort(probs)[::-1]
        candi_index = sorted_index[:topk]
        candi_probs = [probs[i] for i in candi_index]
        # normalize probs
        candi_probs /= sum(candi_probs)
        # choose by predicted probs
        prediction = int(np.random.choice(candi_index, size=1, p=candi_probs)[0])
        return prediction
