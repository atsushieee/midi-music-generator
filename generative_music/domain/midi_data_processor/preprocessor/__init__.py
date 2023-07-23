"""Midi data pre-processing package.

This package contains various pre-processing modules and classes for MIDI data,
such as loading data, data augmentation, data splitting and chord extraction,
to prepare for machine learning tasks.
"""

from generative_music.domain.midi_data_processor.preprocessor.preprocessor import \
    Preprocessor

__all__ = ["Preprocessor"]
