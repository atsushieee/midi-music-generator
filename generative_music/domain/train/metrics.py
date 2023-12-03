"""Groupwise Loss Metrics for training models.

The purpose of this class is to provide a group-wise loss computation
which can be easily used in different models and tasks.
It calculates the group-wise loss for a given batch of data,
keeps track of the total loss for each group,
and updates it for each batch during training or evaluation.
"""
from typing import Dict, List, Tuple

import tensorflow as tf

from generative_music.domain.train.losses import LossCalculator


class GroupwiseLossMetrics(tf.keras.metrics.Metric):
    """This class calculates the group-wise loss for a given batch of data.

    IThe class keeps track of the total loss for each group
    and updates it for each batch during training or evaluation.
    """

    def __init__(
        self,
        masked_id: int,
        vocab_size: int,
        event2id_data: Dict[str, int],
        label_smoothing: float = 0.1,
        name="groupwise_loss",
    ):
        """Initialize the GroupwiseLossMetrics class.

        Args:
            masked_id (int):
                The value used for padding.
                Loss will be ignored for positions with this value.
            vocab_size (int):
                The size of the number of unique words or tokens in the dataset.
            event2id_data (Dict[str, int]):
                The dictionary with keys in the format {event_type}_{kind}
                and sequential integer values.
            label_smoothing (float, optional):
               The label smoothing factor to be applied to the true labels.
               Defaults to 0.1.
            name (str, optional): The name of the metric. Default is "groupwise_loss".
        """
        super().__init__(name=name)
        self.loss_calculator = LossCalculator(masked_id, vocab_size, label_smoothing)
        group_id_array, self.event_names = self._create_group_ids_and_names(
            event2id_data
        )
        self.group_ids = tf.constant(group_id_array, dtype=tf.int32)
        self.num_groups = max(group_id_array) + 1
        self.group_total_losses = self._init_group_total_losses()
        self.group_unmask_dict = {
            i: tf.math.equal(self.group_ids, i) for i in range(self.num_groups)
        }

    def update_state(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        """Update the state of the metric with new observations.

        This method is called for each batch of the dataset during training or evaluation.
        It calculates the loss for each group and updates the group-wise total losses.

        Args:
            y_true (tf.Tensor):
                Ground truth tensor with shape (batch_size, seq_length).
            y_pred (tf.Tensor):
                Predicted probability tensor with shape (batch_size, seq_length, vocab_size).
        """
        loss_values, mask = self.loss_calculator.calc_total_loss(y_true, y_pred)
        # Reshape the tensor to be suitable for the use in tf.scatter_nd.
        indices = tf.reshape(tf.range(self.num_groups), [-1, 1])
        updates_loss = []
        # Compute group-wise loss
        for i in range(self.num_groups):
            group_unmask = tf.gather(self.group_unmask_dict[i], y_true)
            group_unmask = tf.cast(group_unmask, dtype=loss_values.dtype)
            group_loss = loss_values * group_unmask
            group_loss_value = self.loss_calculator.average_loss(
                group_loss, 1 - group_unmask
            )
            updates_loss.append(group_loss_value)
        # The tf.stack operation is used to convert
        # the list of tensors `updates_loss` into a single tensor for tf.scatter_nd operation.
        updates_loss = tf.stack(updates_loss)
        self.group_total_losses = self.group_total_losses + tf.scatter_nd(
            indices, updates_loss, self.group_total_losses.shape
        )

    def result(self, total_steps: int) -> tf.Variable:
        """Calculate the average loss per step.

        Args:
            total_steps (int): The total number of steps taken during training.

        Returns:
            tf.Tensor:
                The average loss per step, calculated as the group total losses
                divided by the total number of steps.
        """
        return self.group_total_losses / total_steps

    def reset_states(self) -> tf.Variable:
        """Reset the state of the metric.

        This method is called at the end of an epoch/evaluation.

        Returns:
            tf.Variable:
                A variable initialized to zeros with a shape equal to the number of groups.
                This variable represents the reset total loss for each group.
        """
        self.group_total_losses = self._init_group_total_losses()

    def _create_group_ids_and_names(
        self, data_dict: Dict[str, int]
    ) -> Tuple[List[int], List[str]]:
        """Create a group id array and a list of event names.

        Given a dictionary with keys in the format {event_type}_{kind}
        and values as sequential integers, create an array and a list with rules:
        - The order of values indicates the kind
        - Event types are assigned ids in the order they appear, starting from 0
        - The array ends with an extra element representing the "padding" type
        - Replace spaces in event names with underscores and convert to lower case

        Args:
            data_dict (Dict[str, int]):
                The dictionary with keys in the format {event_type}_{kind}
                and sequential integer values.

        Returns:
            List[int]: The resulting array with event type ids.
            List[str]: The list of event name.
        """
        event_ids = {}
        event_names = []
        next_event_id = 0
        group_ids = []
        # Create a new dictionary sorted by value
        sorted_dict = dict(sorted(data_dict.items(), key=lambda item: item[1]))
        for key in sorted_dict:
            event_name = key.split("_")[0].replace(" ", "_").lower()
            # If the event has not yet appeared, assign a new id
            if event_name not in event_ids:
                event_names.append(event_name)
                event_ids[event_name] = next_event_id
                next_event_id += 1
            group_ids.append(event_ids[event_name])
        # Add the latest event id for padding data
        group_ids.append(next_event_id)
        return group_ids, event_names

    def _init_group_total_losses(self) -> tf.Variable:
        """Initialize the group total losses as a Keras weight.

        This method uses the `add_weight` function
        which is a common pattern when creating custom metrics in Keras.
        It ensures that the state of the metric is properly managed within the Keras training loops
        and gets reset between epochs.

        Returns:
            tf.Variable:
                A variable initialized to zeros with a shape equal to the number of groups.
                This variable represents the total loss for each group.
        """
        return self.add_weight(
            name="group_total_losses", initializer="zeros", shape=(self.num_groups,)
        )
