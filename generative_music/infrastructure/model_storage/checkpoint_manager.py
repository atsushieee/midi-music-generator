"""CheckpointManager for managing training checkpoints.

The purpose of this file is to provide a CheckpointManager class
that can be imported and used to handle model checkpoints during training.
"""
import os

import tensorflow as tf


class CheckpointManager:
    """This class manages the checkpoints of the model during training.

    The CheckpointManager saves the state of the model and optimizer at each epoch.
    It also allows for the restoration of the model state in case of interruptions during training,
    or for the purpose of continuing training from a specific epoch.

    The current epoch number is also saved and restored along with the model and optimizer's state.
    This helps in keeping track of the training progress even in cases of interruptions
    or when training is spread across multiple sessions.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        checkpoint_dir: str,
    ):
        """Initialize the CheckpointManager class.

        Args:
            model (tf.keras.Model):
                The model to be trained and whose checkpoints are to be managed.
            optimizer (tf.keras.optimizers.Optimizer):
                The optimizer to be used for training the model.
            checkpoint_dir (str):
                The directory where the checkpoints will be saved.
        """
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        # Check if the directory exists, if not, create it
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint = tf.train.Checkpoint(
            epoch=tf.Variable(0), optimizer=self.optimizer, model=self.model
        )
        self._restore()

    def save(self, epoch: int):
        """Save the current state of the model, optimizer, and the current epoch number.

        Args:
            epoch (int):
                The current epoch number to be saved along with the model
                and optimizer's state.
        """
        self.checkpoint.epoch.assign(epoch)
        save_path = self.checkpoint.save(
            file_prefix=os.path.join(self.checkpoint_dir, "ckpt")
        )
        print(f"Saved checkpoint to {save_path}")

    def get_epoch(self):
        """Get the current epoch number.

        Returns:
            int: The current epoch number.
        """
        return self.checkpoint.epoch.numpy()

    def _restore(self):
        """Restore the latest checkpoint.

        If checkpoint exists, restore it. Otherwise, initialize the model and optimizer.
        """
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest_checkpoint:
            self.checkpoint.restore(latest_checkpoint)
            print(f"Restored from {latest_checkpoint}")
        else:
            print("Initializing from scratch.")
