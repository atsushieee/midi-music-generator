"""Checkpoint and SavedModel management package for machine learning applications.

This package provides two main classes, CheckpointManager and SavedModelManager.
CheckpointManager handles model checkpoints during training,
while SavedModelManager is responsible for saving and loading models in the SavedModel format.
These tools are crucial for preserving training progress and deploying trained models.
"""

from generative_music.infrastructure.model_storage.checkpoint_manager import \
    CheckpointManager

__all__ = ["CheckpointManager"]
