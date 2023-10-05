"""Tests for a class that initializes and validates parameters for inference."""
import pytest

from generative_music.domain.inference.generation_parameters import \
    GenerationParameters


class TestGenerationParameters:
    """A test class for the GenerationParameters.

    The class is responsible for testing the GenerationParameters' functionality,
    which includes initializing with given parameters and validating the parameters.
    """

    def setup_method(self):
        """Initialize the GenerationParameters tests.

        Set up initializing the GenerationParameters instance.
        This method is called before each test function is executed.
        """
        self.valid_n_target_bar = 10
        self.valid_temperature = 0.5
        self.valid_topk = 5
        self.valid_num_event_ids = 100
        self.valid_padding_id = 0
        self.valid_max_length = 50

    def test_initialization(self):
        """Test if the GenerationParameters class is correctly initialized.

        This test checks if the output parameters of the class match the input parameters.
        """
        # 正常なパラメータで初期化できることを確認
        params = GenerationParameters(
            self.valid_n_target_bar,
            self.valid_temperature,
            self.valid_topk,
            self.valid_num_event_ids,
            self.valid_padding_id,
            self.valid_max_length,
        )
        assert params.n_target_bar == self.valid_n_target_bar
        assert params.temperature == self.valid_temperature
        assert params.topk == self.valid_topk
        assert params.num_event_ids == self.valid_num_event_ids
        assert params.padding_id == self.valid_padding_id
        assert params.max_length == self.valid_max_length

    def test_validation(self):
        """Test if the parameters are correctly validated.

        This test checks if a ValueError is raised when an invalid parameter is given.
        """
        # n_target_bar is a negative value.
        with pytest.raises(ValueError):
            GenerationParameters(
                -1,
                self.valid_temperature,
                self.valid_topk,
                self.valid_num_event_ids,
                self.valid_padding_id,
                self.valid_max_length,
            )
        # temperature is a negative value.
        with pytest.raises(ValueError):
            GenerationParameters(
                self.valid_n_target_bar,
                -1.0,
                self.valid_topk,
                self.valid_num_event_ids,
                self.valid_padding_id,
                self.valid_max_length,
            )
        # topk is 0.
        with pytest.raises(ValueError):
            GenerationParameters(
                self.valid_n_target_bar,
                self.valid_temperature,
                0,
                self.valid_num_event_ids,
                self.valid_padding_id,
                self.valid_max_length,
            )
        # topk is larger than num_event_ids.
        with pytest.raises(ValueError):
            GenerationParameters(
                self.valid_n_target_bar,
                self.valid_temperature,
                self.valid_num_event_ids + 1,
                self.valid_num_event_ids,
                self.valid_padding_id,
                self.valid_max_length,
            )
        # padding_id is a negative value.
        with pytest.raises(ValueError):
            GenerationParameters(
                self.valid_n_target_bar,
                self.valid_temperature,
                self.valid_topk,
                self.valid_num_event_ids,
                -1,
                self.valid_max_length,
            )
        # max_length is a negative value.
        with pytest.raises(ValueError):
            GenerationParameters(
                self.valid_n_target_bar,
                self.valid_temperature,
                self.valid_topk,
                self.valid_num_event_ids,
                self.valid_padding_id,
                -1,
            )
