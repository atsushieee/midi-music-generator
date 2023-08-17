"""Training Package.

This package contains modules and classes for training models,
including custom learning rate scheduler, loss function, and training step.
"""

from generative_music.domain.train.learning_rate_schedulers import \
    WarmupCosineDecayScheduler
from generative_music.domain.train.losses import \
    LabelSmoothedCategoricalCrossentropy
from generative_music.domain.train.train_data_loader import TrainDataLoader
from generative_music.domain.train.train_step import TrainStep

__all__ = [
    "WarmupCosineDecayScheduler",
    "LabelSmoothedCategoricalCrossentropy",
    "TrainStep",
    "TrainDataLoader",
]
