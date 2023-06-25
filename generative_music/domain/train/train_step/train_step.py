"""Custom training step implementation for Keras models.

The purpose of this file is to provide the TrainStep class,
which implements a custom training step for Keras models.
This allows for more control over the training process
and can be used in conjunction with custom loss functions and optimizers.
"""
import tensorflow as tf


class TrainStep:
    """This class implements a custom training step for a Keras model.

    It computes the loss for the given batch of data,
    and applies the gradients to update the model's weights.
    This allows for more control over the training process
    and can be used in conjunction with custom loss functions and optimizers.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        loss: tf.keras.losses.Loss,
        optimizer: tf.keras.optimizers.Optimizer,
    ):
        """Initialize the TrainStep class.

        Args:
            model (tf.keras.Model): The model to be trained.
            loss (tf.keras.losses.Loss): The loss function to compute the loss.
            optimizer (tf.keras.optimizers.Optimizer):
                The optimizer to apply gradients.
        """
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def __call__(self, x_batch: tf.Tensor, y_batch: tf.Tensor) -> tf.Tensor:
        """Compute the loss and apply gradients for the given batch.

        Args:
            x_batch (tf.Tensor):
                Input tensor with shape (batch_size, seq_length).
            y_batch (tf.Tensor):
                Ground truth tensor with shape (batch_size, seq_length, vocab_size).

        Returns:
            tf.Tensor: The computed loss for the given batch.
        """
        with tf.GradientTape() as tape:
            y_pred = self.model(x_batch)
            loss_value = self.loss(y_batch, y_pred)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value
