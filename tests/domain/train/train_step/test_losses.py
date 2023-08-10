"""Tests for the Loss function."""
import pytest
import tensorflow as tf

from generative_music.domain.train.train_step.losses import \
    LabelSmoothedCategoricalCrossentropy

test_data = [
    (0.0, 0.28990924),
    (0.1, 0.4125352),
]


class TestLabelSmoothedCategoricalCrossentropy:
    """A test class for the LabelSmoothedCategoricalCrossentropy loss function.

    This class tests if the computed loss value matches the expected loss value
    for a given set of true labels, predicted probabilities and label smoothing parameter.
    It verifies the behavior of the label smoothing parameter
    and its effect on the loss value.
    """

    def setup_method(self):
        """Initialize the TestLabelSmoothedCategoricalCrossentropy tests.

        Set up the true labels and predicted probabilities for testing the loss function.
        This method is called before each test function is executed.
        """
        self.y_true = tf.constant([[1, 0]])
        self.y_pred = tf.constant([[[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]])

    @pytest.mark.parametrize("label_smoothing, expected_loss", test_data)
    def test_loss(self, label_smoothing: float, expected_loss: float):
        """Test the LabelSmoothedCategoricalCrossentropy loss function.

        This test checks if the computed loss value matches the expected loss value
        for a given set of true labels, predicted probabilities and label smoothing parameter.
        Args:
            label_smoothing (float):
                The label smoothing parameter to be applied to the true labels.
            expected_loss (float):
                The expected loss value for the given true labels,
                predicted probabilities and label smoothing.
        """
        loss = LabelSmoothedCategoricalCrossentropy(
            vocab_size=3, masked_id=2, label_smoothing=label_smoothing
        )
        loss_value = loss(self.y_true, self.y_pred)
        expected_value = tf.constant(expected_loss, dtype=tf.float32)
        tf.debugging.assert_near(loss_value, expected_value, rtol=1e-8, atol=1e-8)

    def test_loss_with_mask(self):
        """Test if the computed loss value with mask matches the expected loss value.

        This test checks if the loss function correctly applies the mask
        to ignore padding tokens in the true labels during loss computation.
        """
        loss = LabelSmoothedCategoricalCrossentropy(
            vocab_size=3, masked_id=0, label_smoothing=0.0
        )
        loss_value = loss(self.y_true, self.y_pred)
        expected_value = tf.constant(0.22314353, dtype=tf.float32)
        tf.debugging.assert_near(loss_value, expected_value, rtol=1e-8, atol=1e-8)
