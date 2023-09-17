"""Tests for the CheckpointManager used for saving and loading model weights and training epoch."""
import os
import shutil

import numpy as np
import tensorflow as tf

from generative_music.infrastructure.model_storage.checkpoint_manager import \
    CheckpointManager


class TestCheckpointManager:
    """A test class for the CheckpointManager."""

    def setup_method(self):
        """Initialize the TestCheckpointManager tests.

        This setup method initializes a simple Keras Sequential model
        with a single Dense layer.
        The model is built with a dummy input shape, assuming an input shape of 10.

        The method also prepares the CheckpointManager
        which handles the saving and loading of model weights.
        The model's initial weights are saved as a baseline
        for testing the functionality of the CheckpointManager.
        """
        self.model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
        # Add dummy data to build the model
        self.model.build((None, 10))  # Assuming input shape is 10 (Batch size is None)
        self.optimizer = tf.keras.optimizers.Adam()
        self.checkpoint_dir = "./test_checkpoints"

        self.manager = CheckpointManager(
            self.model, self.optimizer, self.checkpoint_dir
        )

    def teardown_method(self):
        """Clean up the test environment by removing the test data.

        This method is called after each test function is executed.
        """
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)

    def test_epoch_is_restored(self):
        """Test if the saved epoch is correctly restored.

        Check if the returned epoch matches the saved epoch.
        Also, verify if a new instance of CheckpointManager
        can restore the epoch correctly from the saved checkpoint.
        """
        self.manager.save(epoch=0)
        assert self.manager.get_epoch() == 0
        self.manager.save(epoch=10)
        assert self.manager.get_epoch() == 10

        # Create and restore a new manager
        new_manager = CheckpointManager(self.model, self.optimizer, self.checkpoint_dir)
        assert new_manager.get_epoch() == 10

    def test_model_weights_are_restored(self):
        """Test if the model weights are correctly restored.

        Change the model weights and save them using CheckpointManager.
        Verify if the weights are correctly restored
        when a new instance of CheckpointManager is used.
        The test is performed for each set of weights in the model.
        """
        # Change model weights and then save
        # Note: weights[0] represents the weight matrix
        # and weights[1] represents the bias vector for the Dense layer.
        # Here, we're adding random values to the weight matrix for testing.
        self.model.layers[0].weights[0].assign_add(
            tf.random.normal(shape=self.model.layers[0].weights[0].shape)
        )
        weights_before = self.model.get_weights()
        self.manager.save(epoch=0)

        # Create and restore a new manager
        new_manager = CheckpointManager(self.model, self.optimizer, self.checkpoint_dir)
        weights_after = new_manager.model.get_weights()

        # Verify weights are restored correctly
        # We loop through each set of weights
        # (i.e., weight matrix and bias vector) to ensure they're all close.
        for w_before, w_after in zip(weights_before, weights_after):
            np.testing.assert_allclose(w_before, w_after, rtol=1e-6, atol=1e-6)
