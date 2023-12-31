"""A module for preparing datasets from MIDI files."""
from pathlib import Path
from typing import List

import tensorflow as tf


class MidiTFRecordsWriter:
    """A class for writing MIDI data to TFRecord files.

    This class provides methods for creating tf.train.Example objects
    from tokenized MIDI data and writing them to TFRecord files.
    """

    def write_tfrecords(self, tokenized_midis: List[List[int]], output_path: Path):
        """Write the given tokenized MIDI data to a TFRecord file.

        Args:
            tokenized_midis (List[List[int]]):
                A list of tokenized MIDI files,
                where each file is represented as a list of integers.
            output_path (Path):
                The output file path where the TFRecord file will be written.

        Notes:
            Serialization is the process of converting a data structure or object
            into a sequence of bytes that can be easily stored or transmitted.
            Serialization has several benefits:
                - It allows for efficient serialization and deserialization,
                　which can lead to faster processing times.
                - The resulting serialized data is generally smaller in size
                　compared to other formats like JSON or XML,
                　which can help reduce storage and bandwidth requirements.
        """
        self._ensure_output_directory_exists(output_path)

        with tf.io.TFRecordWriter(str(output_path)) as writer:
            for tokenized_midi in tokenized_midis:
                tf_example = self._create_tf_example(tokenized_midi)
                writer.write(tf_example.SerializeToString())

    def _ensure_output_directory_exists(self, output_path: Path):
        """Ensure the output directory exists, create it if not.

        Args:
            output_path (Path):
                The output file path where the TFRecord file will be written.
        """
        output_directory = output_path.parent
        output_directory.mkdir(parents=True, exist_ok=True)

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
