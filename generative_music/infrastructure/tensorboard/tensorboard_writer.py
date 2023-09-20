"""A simple writer for TensorBoard logs."""
from typing import Dict, Optional

import tensorflow as tf


class TensorboardWriter:
    """This class implements a writer for TensorBoard logs.

    It provides functionality to write scalar values and hyperparameters
    to a specified log directory.
    Scalar values can be any metric like loss or accuracy
    that can be tracked during training or evaluation.
    Hyperparameters are written as text and can be any model parameter or training setting.
    This allows for better monitoring and understanding of the training process.
    """

    def __init__(self, log_dir: str, hyperparameters: Optional[Dict[str, int]] = None):
        """Initialize the TensorboardWriter class.

        Args:
            log_dir (str): Path to the directory where TensorBoard logs will be saved.
            hyperparameters (dict, optional):
                Dictionary of hyperparameters. Default is None.
        """
        self.writer = tf.summary.create_file_writer(log_dir)
        self.hyperparameters = hyperparameters
        if self.hyperparameters:
            self._write_hyperparameters()

    def write_scalar(self, name: str, value: float, step: int):
        """Write a scalar value to TensorBoard.

        Args:
            name (str): Name of the data to be written.
            value (float): Value of the data to be written.
            step (int): Step number of the data (e.g., number of epochs or iterations).
        """
        with self.writer.as_default():
            tf.summary.scalar(name, value, step=step)

    def _write_hyperparameters(self):
        """Write hyperparameters to TensorBoard as text."""
        with self.writer.as_default():
            for name, value in self.hyperparameters.items():
                tf.summary.text(name, str(value), step=0)
