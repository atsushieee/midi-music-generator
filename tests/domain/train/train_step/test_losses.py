"""Tests for the Loss function."""
import pytest
import tensorflow as tf

from generative_music.domain.train.train_step.losses import \
    LabelSmoothedCategoricalCrossentropy

test_data = [
    (0.0, 0.2899095),
    (0.1, 0.4125352),
]


class TestLabelSmoothedCategoricalCrossentropy:
    """A test class for the LabelSmoothedCategoricalCrossentropy loss function.

    This class tests if the computed loss value matches the expected loss value
    for a given set of true labels, predicted probabilities and label smoothing parameter.
    It verifies the behavior of the label smoothing parameter
    and its effect on the loss value.
    """

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
        y_true = tf.constant([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])
        y_pred = tf.constant([[[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]]])

        loss = LabelSmoothedCategoricalCrossentropy(label_smoothing=label_smoothing)
        loss_value = loss(y_true, y_pred)
        assert loss_value.numpy() == pytest.approx(expected_loss, abs=1e-6)
