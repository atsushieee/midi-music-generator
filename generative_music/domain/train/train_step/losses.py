"""Loss functions for training music generation models.

The purpose of this file is to provide a collection of loss functions
that can be easily imported and used in different models and tasks.
"""
import tensorflow as tf


class LabelSmoothedCategoricalCrossentropy(tf.keras.losses.Loss):
    """This class is a custom Keras loss function that implements label-smoothed crossentropy.

    It computes the crossentropy loss between the true labels and predicted probabilities,
    with label smoothing applied to the true labels.
    Label smoothing is controlled by the `label_smoothing` argument, which is a float in [0, 1].
    When > 0, label values are smoothed, meaning the confidence on label values are relaxed.
    For example, if 0.1, use 0.1 / num_classes for non-target labels
    and 0.9 + 0.1 / num_classes for target labels.(quoted from the official documentation)
    This technique encourages the model to be less confident in its predictions,
    which can help prevent overfitting and improve generalization.
    """

    def __init__(self, label_smoothing: float = 0.1):
        """Initialize the LabelSmoothedCategoricalCrossentropy class.

        Args:
           label_smoothing (float, optional):
               The label smoothing factor to be applied to the true labels.
               Defaults to 0.1.
        """
        super().__init__()
        self.smoothed_loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the label-smoothed categorical crossentropy loss.

        Args:
            y_true (tf.Tensor):
                Ground truth tensor with shape (batch_size, seq_length, vocab_size).
            y_pred (tf.Tensor):
                Predicted probability tensor with shape (batch_size, seq_length, vocab_size).

        Returns:
            tf.Tensor: The label-smoothed categorical cross-entropy loss.
        """
        return self.smoothed_loss(y_true, y_pred)
