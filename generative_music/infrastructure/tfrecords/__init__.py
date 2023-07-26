"""MIDI data to TFRecord files conversion package and vice versa.

This package contains classes and functions
for converting tokenized MIDI data into TFRecord files,
and reading them back, which can be used for efficient storage
and input pipelines in machine learning applications.
"""

from generative_music.infrastructure.tfrecords.midi_tfrecords_writer import \
    MidiTFRecordsWriter

__all__ = ["MidiTFRecordsWriter"]
