"""TensorBoard logging package for machine learning applications.

This package provides the TensorboardWriter class,
which is used for writing scalar values and hyperparameters to TensorBoard logs.
This class plays a crucial role in monitoring and understanding the training process,
and the impact of different hyperparameters on the model performance.
"""

from generative_music.infrastructure.tensorboard.tensorboard_writer import \
    TensorboardWriter

__all__ = ["TensorboardWriter"]
