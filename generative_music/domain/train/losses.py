"""Loss functions for training music generation models.

The purpose of this file is to provide a collection of loss functions
that can be easily imported and used in different models and tasks.
"""
import tensorflow as tf


class LabelSmoothedCategoricalCrossentropy(tf.keras.losses.Loss):
    """This class is a custom Keras loss function that implements label-smoothed crossentropy.

    It computes the crossentropy loss between the true labels and predicted probabilities,
    with label smoothing applied to the true labels and ignoring masked positions.
    Label smoothing is controlled by the `label_smoothing` argument, which is a float in [0, 1].
    When > 0, label values are smoothed, meaning the confidence on label values are relaxed.
    For example, if 0.1, use 0.1 / num_classes for non-target labels
    and 0.9 + 0.1 / num_classes for target labels.(quoted from the official documentation)
    This technique encourages the model to be less confident in its predictions,
    which can help prevent overfitting and improve generalization.
    """

    def __init__(self, masked_id: int, vocab_size: int, label_smoothing: float = 0.1):
        """Initialize the LabelSmoothedCategoricalCrossentropy class.

        Args:
            masked_id (int):
                The value used for padding.
                Loss will be ignored for positions with this value.
            vocab_size (int):
                The size of the number of unique words or tokens in the dataset.
            label_smoothing (float, optional):
               The label smoothing factor to be applied to the true labels.
               Defaults to 0.1.
        """
        super().__init__()
        self.masked_id = masked_id
        self.vocab_size = vocab_size
        # `reduction=tf.keras.losses.Reduction.NONE`: avoid reducing the loss to a scalar sum.
        # Without this setting, masked parts would be included in the calculation,
        # preventing the desired masking operation in the call function.
        self.smoothed_loss = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing, reduction=tf.keras.losses.Reduction.NONE
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the label-smoothed categorical crossentropy loss.

        Args:
            y_true (tf.Tensor):
                Ground truth tensor with shape (batch_size, seq_length).
            y_pred (tf.Tensor):
                Predicted probability tensor with shape (batch_size, seq_length, vocab_size).

        Returns:
            tf.Tensor: The label-smoothed categorical cross-entropy loss.
        """
        y_true_one_hot = tf.one_hot(y_true, depth=self.vocab_size, dtype=y_pred.dtype)
        loss_values = self.smoothed_loss(y_true_one_hot, y_pred)
        mask = tf.math.equal(y_true, self.masked_id)
        mask = tf.cast(mask, dtype=loss_values.dtype)
        loss_values *= 1 - mask
        # The average loss by dividing the sum of loss_values
        # by the sum of unmasked elements (1 - mask).
        # This ensures that only the unmasked parts contribute to the final average loss.
        return tf.reduce_sum(loss_values) / tf.reduce_sum(1 - mask)
