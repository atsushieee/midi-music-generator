"""Tests for the Multi-head Attention layer used in the Transformer model."""
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

from generative_music.domain.model.transformer.layer.multi_head_attention.attention import \
    Attention
from generative_music.domain.model.transformer.layer.multi_head_attention.multi_head_attention import \
    MultiHeadedAttention


class TestMultiHeadAttention:
    """A test class for the Multi-head Attention layer.

    The class tests if the input tensor is correctly split into multiple heads,
    and the output is correctly concatenated.
    """

    @pytest.fixture(autouse=True)
    def init_module(self):
        """Initialize the multi-head attention layer tests.

        This fixture creates predefined number of heads and dimension of query, key and value.
        After that, MultiHeadAttention class is instantiated.
        """
        self.num_heads = 8
        self.dim_qkv = 512
        attention = Attention()
        self.multi_head_attention = MultiHeadedAttention(
            self.num_heads, self.dim_qkv, attention
        )
        self.batch_size = 2
        self.seq_length = 8

    def test_split_heads(self):
        """Test the split_heads function in the multi-head attention mechanism.

        This test checks if the input tensor is correctly split into multiple heads,
        by verifying if the returned output has the correct shape.
        """
        x = tf.random.normal((self.batch_size, self.seq_length, self.dim_qkv))
        split_result = self.multi_head_attention.split_heads(x, self.batch_size)
        assert split_result.shape == (
            self.batch_size,
            self.num_heads,
            self.seq_length,
            int(self.dim_qkv / self.num_heads),
        )

    def test_multi_head_attention(self):
        """Test the multi-head attention mechanism.

        This test verify if the returned output has the correct shape,
        which indicates that the multi-head attention output is correctly concatenated.
        """
        # Prepare test data
        query = np.ones(
            (self.batch_size, self.seq_length, self.dim_qkv), dtype=np.float32
        )
        key = np.ones(
            (self.batch_size, self.seq_length, self.dim_qkv), dtype=np.float32
        )
        value = np.ones(
            (self.batch_size, self.seq_length, self.dim_qkv), dtype=np.float32
        )
        # Set up the mock
        self.multi_head_attention.attention = MagicMock(
            return_value=(
                np.ones(
                    (
                        self.batch_size,
                        self.num_heads,
                        self.seq_length,
                        int(self.dim_qkv / self.num_heads),
                    )
                ),
                np.ones(
                    (self.batch_size, self.num_heads, self.seq_length, self.seq_length)
                ),
            )
        )
        # Call the function
        output = self.multi_head_attention(query, key, value)
        # Check if the output is correctly concatenated
        assert output.shape == (self.batch_size, self.seq_length, self.dim_qkv)
