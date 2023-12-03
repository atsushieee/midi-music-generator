"""Testing Loss Metrics per Event."""
import tensorflow as tf

from generative_music.domain.train.metrics import GroupwiseLossMetrics


class TestGroupwiseLossMetrics:
    """A test class for the GroupwiseLossMetrics class.

    This class tests if the computed group-wise total losses match the expected losses
    for a given set of true labels and predicted probabilities.
    And it verifies the behavior of the update_state and reset_states methods
    and their effect on the group-wise total losses.
    """

    def setup_method(self):
        """Initialize the input tensors for the GroupwiseLossMetrics tests.

        This fixture creates predefined custom scheduler.
        """
        event2id_data = {
            "type1_name1": 0,
            "type2_name2": 1,
            "type2_name3": 2,
            "type1_name4": 3,
        }
        self.metrics = GroupwiseLossMetrics(
            masked_id=4, vocab_size=5, event2id_data=event2id_data, label_smoothing=0.0
        )
        self.y_true = tf.constant([[1, 0, 1, 2], [1, 1, 3, 4]])
        self.y_pred = tf.constant([[[0.2] * 5] * 4] * 2)

    def test_initialization(self):
        """Test if the GroupwiseLossMetrics class is correctly initialized.

        This test checks if the num_groups attribute of the GroupwiseLossMetrics instance
        is correctly set based on the input group_ids.
        """
        assert self.metrics.num_groups == 3

    def test_update_state(self):
        """Test if the update_state method correctly updates the group-wise total losses.

        This test checks if the update_state method correctly computes
        and accumulates the group-wise total losses
        based on the input true labels and predicted labels.
        """
        self.metrics.update_state(self.y_true, self.y_pred)
        expected_value = tf.constant([1.609438, 1.609438, 0.0])
        assert tf.reduce_all(tf.equal(expected_value, self.metrics.group_total_losses))

    def test_reset_states(self):
        """Test if the reset_states method correctly resets the group-wise total losses.

        This test checks if the reset_states method correctly resets
        the group-wise total losses to zero after the update_state method has been called.
        """
        self.metrics.update_state(self.y_true, self.y_pred)
        self.metrics.reset_states()
        assert tf.reduce_sum(self.metrics.group_total_losses) == 0.0
