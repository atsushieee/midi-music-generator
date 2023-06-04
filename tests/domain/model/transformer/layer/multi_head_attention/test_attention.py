"""Tests for the Attention layer used in the Transformer model."""
from typing import List

import numpy as np
import pytest
import tensorflow as tf

from generative_music.domain.model.transformer.layer.multi_head_attention.attention import \
    Attention

test_data = [
    (
        [
            [0.880797, 0.119203],
            [0.001501, 0.998498],
        ],
        [
            [0.238406, 5.284782, 1.072826, 2.761594],
            [1.996998, 0.009007, 8.986489, 1.003002],
        ],
        None,
    ),
    (
        [
            [1.0, 0.0],
            [0.001501, 0.998498],
        ],
        [
            [0, 6, 0, 3],
            [1.996998, 0.009007, 8.986489, 1.003002],
        ],
        [[0, 1], [0, 0]],
    ),
]


class TestAttention:
    """A test class for the Attention layer.

    The class tests if the output tensor and attention weights tensor
    are equal to the expected tensors, both with and without dropout applied.
    """

    @pytest.fixture
    def init_module(self):
        """Initialize the input tensors for the attention layer tests.

        This fixture creates predefined query, key, and value tensors.
        """
        # Create predefined input tensors
        # batch_size, seq_len, d_model = 1, 2, 4
        self.query = tf.constant([[[3, 0, 1, 2], [0, 4, 2, 1]]], dtype=tf.float32)
        self.key = tf.constant([[[2, 0, 1, 1], [0, 3, 1, 2]]], dtype=tf.float32)
        self.value = tf.constant([[[0, 6, 0, 3], [2, 0, 9, 1]]], dtype=tf.float32)

    @pytest.mark.usefixtures("init_module")
    @pytest.mark.parametrize(
        "attention_weights_expected, output_expected, mask", test_data
    )
    def test_attention_without_dropout(
        self,
        attention_weights_expected: List[List[float]],
        output_expected: List[List[float]],
        mask: List[List[int]],
    ):
        """Test the attention layer without dropout applied.

        This test checks if the output tensor and attention weights tensor
        are equal to the expected tensors.
        Args:
            attention_weights_expected (List[List[float]]):
                The expected attention weights without dropout.
            output_expected (List[List[float]]):
                The expected output tensor without dropout.
            mask (List[List[float]]):
                The mask to be applied, if any.
                Use 1 for positions to be masked and 0 for positions to be kept.
        """
        # Convert expected values to tensors to compare with code results
        expected_attention_weights_tensor = tf.constant(
            attention_weights_expected, dtype=tf.float32
        )
        expected_output_tensor = tf.constant(output_expected, dtype=tf.float32)
        dropout_rate = None
        attention_layer = Attention(dropout_rate=dropout_rate)
        if mask is not None:
            mask = tf.constant(mask, dtype=tf.float32)
        output, attention_weights = attention_layer(
            self.query, self.key, self.value, mask
        )
        # Check if the output tensor and attention weights tensor
        # are equal to the expected tensors
        assert np.allclose(output.numpy(), expected_output_tensor.numpy(), atol=1e-6)
        assert np.allclose(
            attention_weights.numpy(),
            expected_attention_weights_tensor.numpy(),
            atol=1e-6,
        )

    @pytest.mark.usefixtures("init_module")
    @pytest.mark.parametrize("attention_weights_expected, _, mask", test_data)
    def test_attention_with_dropout(self, attention_weights_expected, _, mask):
        """Test the attention layer with dropout applied.

        This test checks if the attention weights tensor
        is equal to the expected tensors.
        Args:
            attention_weights_expected (List[List[float]]):
                The expected attention weights with dropout.
            _ (List[List[float]]):
                A placeholder for the output values, which are not tested.
            mask (List[List[float]]):
                The mask to be applied, if any.
                Use 1 for positions to be masked and 0 for positions to be kept.
        """
        # Convert expected values to tensors to compare with code results
        expected_attention_weights_tensor = tf.constant(
            attention_weights_expected, dtype=tf.float32
        )

        tf.random.set_seed(1)
        dropout_rate = 0.5
        attention_layer = Attention(dropout_rate=dropout_rate)
        if mask is not None:
            mask = tf.constant(mask, dtype=tf.float32)
        output, attention_weights = attention_layer(
            self.query, self.key, self.value, mask, training=True
        )
        # Check if the dropout layer runs without error
        assert output is not None
        assert attention_weights is not None
        # Check if attention weights contain zero values after applying dropout
        assert tf.reduce_any(attention_weights == 0)
        # Check if the non-zero attention weights are 2 times the expected values
        non_zero_mask = tf.cast(
            tf.math.not_equal(attention_weights, 0), dtype=tf.float32
        )
        expected_scaled_attention_weights = (
            1 / (1 - dropout_rate) * expected_attention_weights_tensor * non_zero_mask
        )
        assert np.allclose(
            attention_weights.numpy(),
            expected_scaled_attention_weights.numpy(),
            atol=1e-6,
        )
