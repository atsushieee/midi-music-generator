"""Batch Generation Package.

This package contains modules and classes for preparing and processing datasets
to be used in training music generation models,
including batch generation, subsequence extraction and mask generation.
"""

from generative_music.domain.dataset_preparation.batch_generation.batch_gererator import \
    BatchGenerator

__all__ = ["BatchGenerator"]
