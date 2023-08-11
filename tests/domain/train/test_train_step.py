"""Tests for the Train Step function."""
from unittest.mock import MagicMock, patch

import tensorflow as tf

from generative_music.domain.train.train_step import TrainStep


class TestTrainStep:
    """A test class for the train step function.

    This class tests if the train_step function correctly calls
    the model, loss, and optimizer,
    and produces the expected output after processing the input tensors.
    """

    def test_train_step(self):
        """Check the train_step function behavior.

        Tests if the train_step function correctly calls
        the model, loss and optimizer with the provided input tensors,
        and if it returns the expected loss value.
        """
        # Create mock objects
        mock_model = MagicMock(spec=tf.keras.Model)
        mock_loss = MagicMock(spec=tf.keras.losses.Loss)
        mock_optimizer = MagicMock(spec=tf.keras.optimizers.Optimizer)
        mock_x_batch = MagicMock(spec=tf.Tensor)
        mock_y_batch = MagicMock(spec=tf.Tensor)
        mock_y_pred = MagicMock(spec=tf.Tensor)
        mock_loss_value = MagicMock(spec=tf.Tensor)

        # Set the behavior of mock objects
        mock_model.return_value = mock_y_pred
        mock_loss.return_value = mock_loss_value
        mock_optimizer.return_value = None

        # Instantiate the TrainingStep class
        train_step = TrainStep(mock_model, mock_loss, mock_optimizer)

        # Mock the tensorflow.GradientTape context manager
        with patch("tensorflow.GradientTape") as mock_tape:
            mock_tape.__enter__.gradient.return_value = [
                MagicMock(spec=tf.Tensor)
                for _ in range(len(mock_model.trainable_variables))
            ]
            # Call result within the with statement
            # to ensure correct behavior of the mocked GradientTape.
            result = train_step(mock_x_batch, mock_y_batch)

        # Verify the calls
        mock_model.assert_called_once_with(mock_x_batch)
        mock_loss.assert_called_once_with(mock_y_batch, mock_y_pred)
        assert result == mock_loss_value

        # Verify the gradient calculation and optimization
        mock_optimizer.apply_gradients.assert_called_once()
