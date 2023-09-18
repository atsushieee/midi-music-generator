"""Training Package.

This package contains modules and classes for training models,
including custom learning rate scheduler, loss function, and training step,
and a class for calculating the number of steps per epoch based on MIDI files.
"""

from generative_music.domain.train.epoch_steps_calculator import \
    EpochStepsCalculator
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
    "EpochStepsCalculator",
]
