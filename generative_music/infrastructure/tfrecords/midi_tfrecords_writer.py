"""A module for preparing datasets from MIDI files."""
from pathlib import Path
from typing import List

import tensorflow as tf


class MidiTFRecordsWriter:
    """A class for writing MIDI data to TFRecord files.

    This class provides methods for creating tf.train.Example objects
    from tokenized MIDI data and writing them to TFRecord files.
    """

    def _create_tf_example(self, tokenized_midi: List[int]) -> tf.train.Example:
        """Create a tf.train.Example object from the given tokenized MIDI data.

        Args:
            tokenized_midi (List[int]):
                A list of integers representing a tokenized MIDI file.

        Returns:
            tf.train.Example:
                A tf.train.Example object containing the tokenized MIDI data.
        """
        feature = {
            "tokenized_midi": tf.train.Feature(
                int64_list=tf.train.Int64List(value=tokenized_midi)
            )
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def write_tfrecords(self, tokenized_midis: List[List[int]], output_path: Path):
        """Write the given tokenized MIDI data to a TFRecord file.

        Args:
            tokenized_midis (List[List[int]]):
                A list of tokenized MIDI files,
                where each file is represented as a list of integers.
            output_path (Path):
                The output file path where the TFRecord file will be written.
        """
        with tf.io.TFRecordWriter(str(output_path)) as writer:
            for tokenized_midi in tokenized_midis:
                tf_example = self._create_tf_example(tokenized_midi)
                writer.write(tf_example.SerializeToString())
