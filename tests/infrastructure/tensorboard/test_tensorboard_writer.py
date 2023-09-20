"""Tests for writing scalar values and hyperparameters to the TensorBoard log directory."""
import os
import shutil
import tempfile

from generative_music.infrastructure.tensorboard.tensorboard_writer import \
    TensorboardWriter


class TestTensorboardWriter:
    """A test class for the TensorboardWriter.

    The class tests if the scalar values and hyperparameters
    are correctly written to the TensorBoard log directory,
    and if the log directory and log files are correctly created.
    """

    def setup_method(self):
        """Initialize the TestTensorboardWriter tests.

        Set up the test environment by creating a temporary directory
        for logs and initializing the TensorboardWriter instance.
        This method is called before each test function is executed.
        """
        self.log_dir = tempfile.mkdtemp()
        self.hyperparameters = {"learning_rate": 0.01, "batch_size": 32}
        self.tensorboard_writer = TensorboardWriter(self.log_dir, self.hyperparameters)

    def teardown_method(self):
        """Clean up the test environment by removing the test data.

        This method is called after each test function is executed.
        """
        shutil.rmtree(self.log_dir)

    def test_write_scalar(self):
        """Check if the scalar values are correctly written to the TensorBoard log directory.

        This test checks the existence of the log directory and the creation of log files
        after the write_scalar function is called.
        """
        self.tensorboard_writer.write_scalar("test_metric", 0.5, 1)
        assert os.path.exists(self.log_dir)
        assert len(os.listdir(self.log_dir)) > 0

    def test_write_hyperparameters(self):
        """Check if the hyperparameters are correctly written to the TensorBoard log directory.

        This test checks the existence of the log directory and the creation of log files
        after the TensorboardWriter instance is initialized with hyperparameters.
        """
        assert os.path.exists(self.log_dir)
        assert len(os.listdir(self.log_dir)) > 0
