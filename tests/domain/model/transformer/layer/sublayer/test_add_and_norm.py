"""Tests for the Add&Norm layer used in the Transformer model."""
import numpy as np
import pytest
import tensorflow as tf

from generative_music.domain.model.transformer.layer.sublayer.add_and_norm import \
    AddAndNorm


class TestAddAndNorm:
    """A test class for the Add&Norm layer.

    The class tests if the input tensor is correctly processed through the
    Add&Norm layer and if dropout is applied correctly.
    """

    @pytest.fixture(autouse=True)
    def init_module(self):
        """Initialize the Add&Norm layer tests.

        This fixture creates predefined dropout rate.
        After that, AddAndNorm class is instantiated.
        """
        self.dropout_rate = 0.5
        self.add_norm_layer = AddAndNorm(dropout_rate=self.dropout_rate)
        # Prepare test data
        self.prev_input_tensor = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.prev_output_tensor = tf.constant([[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]])

    def test_add_norm_value(self):
        """Check the addition of two inputs and normalization tensor values.

        Tests if the output tensor from the AddAndNorm layer is
        the same as the manually calculated normalized tensor.
        """
        added = self.prev_input_tensor + self.prev_output_tensor
        mean, variance = tf.nn.moments(added, axes=-1, keepdims=True)
        # For tf.keras.layers.LayerNormalization,
        # the default initial values are gamma=1 and beta=0.
        # So, if the initial values of beta and gamma are changed,
        # this calculation will also need to be updated.
        normalized = (added - mean) / tf.sqrt(variance + 1e-6)
        # Ensure the output from the AddAndNorm class is
        # the same as the manually calculated normalized output.
        assert np.allclose(
            self.add_norm_layer(
                self.prev_input_tensor, self.prev_output_tensor
            ).numpy(),
            normalized.numpy(),
            atol=1e-6,
        )

    def test_output_tensor_with_dropout(self):
        """Check the dropout application in the previous output tensor.

        Tests if the previous output tensor with dropout applied has
        elements set to 0 where dropout is applied,
        and elements equal to the corresponding elements in the previous output tensor
        without dropout divided by (1 - dropout_rate) where dropout is not applied.
        """
        _, prev_output_tensor_with_dropout = self.add_norm_layer(
            self.prev_input_tensor,
            self.prev_output_tensor,
            training=True,
            return_prev_layer_output=True,
        )
        _, prev_output_tensor_without_dropout = self.add_norm_layer(
            self.prev_input_tensor,
            self.prev_output_tensor,
            training=False,
            return_prev_layer_output=True,
        )

        # Create a boolean mask for elements that are 0 in the output_tensor_with_dropout
        zero_mask = tf.equal(prev_output_tensor_with_dropout, 0)

        scaled_elements = prev_output_tensor_without_dropout / (1 - self.dropout_rate)
        # If zero_mask is True, set the element to 0
        # If zero_mask is False, set the element to the corresponding value in scaled_elements
        modified_prev_output_without_dropout = tf.where(
            zero_mask,
            tf.zeros_like(prev_output_tensor_without_dropout),
            scaled_elements,
        )
        assert np.allclose(
            modified_prev_output_without_dropout.numpy(),
            prev_output_tensor_with_dropout.numpy(),
            atol=1e-6,
        )
