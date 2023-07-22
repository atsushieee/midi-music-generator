"""Midi data tokenization (event2id and id2event) package.

This package contains various tokenization modules and classes for MIDI data,
including the creation of token mappings, tokenization and detokenization processes.
"""

from generative_music.domain.midi_data_processor.midi_tokenization.mapping_generator import \
    MappingGenerator

__all__ = ["MappingGenerator"]
