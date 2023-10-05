"""A module to manage parameters for generating music events."""


class GenerationParameters:
    """This class is responsible for validating and storing the parameters required for generating music events.

    It checks the validity of parameters such as the target number of bars to generate,
    the temperature for sampling, and the number of top candidates to consider for sampling.
    """

    def __init__(
        self,
        n_target_bar: int,
        temperature: float,
        topk: int,
        num_event_ids: int,
        padding_id: int,
        max_length: int,
    ):
        """Initialize GenerationParameters with given parameters.

        Args:
            n_target_bar (int):
                The target number of bars to generate.
                Must be an integer greater than or equal to 1.
            temperature (float):
                The temperature for generation.
                Must be a float greater than or equal to 0.0.
            topk (int):
                The number of top candidates to consider for generation.
                Must be an integer between 1 and max_word_id.
            num_event_ids (int): TThe length of event_ids.
            padding_id (int): The id used for padding the sequence.
            max_length (int): The maximum length of the sequence.

        Raises:
            ValueError: If any of the parameters do not meet their conditions.
        """
        self.n_target_bar = n_target_bar
        self.temperature = temperature
        self.topk = topk
        self.num_event_ids = num_event_ids
        self.padding_id = padding_id
        self.max_length = max_length
        self._check_validation()

    def _check_validation(self):
        """Check the validity of the parameters.

        Raises:
            ValueError: If any of the parameters do not meet their conditions.
        """
        if not isinstance(self.n_target_bar, int) or self.n_target_bar < 1:
            raise ValueError(
                "n_target_bar must be an integer greater than or equal to 1."
            )
        if not (0.0 <= self.temperature):
            raise ValueError("temperature must be greater than or equal to 0.0.")
        if not isinstance(self.topk, int) or not (1 <= self.topk <= self.num_event_ids):
            raise ValueError(
                f"topk must be an integer between 1 and {self.num_event_ids}."
            )
        if not isinstance(self.padding_id, int) or self.padding_id < 0:
            raise ValueError("padding_id must be an integer.")
        if not isinstance(self.max_length, int) or self.max_length < 1:
            raise ValueError(
                "max_length must be an integer greater than or equal to 1."
            )
