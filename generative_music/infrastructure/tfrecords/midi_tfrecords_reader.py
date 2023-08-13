"""A module for reading datasets from MIDI files."""
from pathlib import Path

import tensorflow as tf


class MidiTFRecordsReader:
    """A class for reading tokenized MIDI data from TFRecord files.

    This class provides methods for creating a dataset from TFRecord files
    containing tokenized MIDI data.
    """

    def create_dataset(self, input_path: Path) -> tf.data.Dataset:
        """Create a dataset from the given TFRecord file.

        Args:
            input_path (Path):
                The input file path where the TFRecord file is located.

        Returns:
            tf.data.Dataset:
                A tf.data.Dataset object containing the tokenized MIDI data.
                The actual returned object is a subclass of tf.data.Dataset
                (due to the use of the map method), but it can be treated
                as a tf.data.Dataset for most purposes.
                This is because the specific subclass may change
                depending on the operations applied to the data.
        """
        raw_dataset = self._read_tfrecords(input_path)
        return raw_dataset.map(self._parse_tf_example)

    def _read_tfrecords(self, input_path: Path) -> tf.data.TFRecordDataset:
        """Read the tokenized MIDI data from a TFRecord file.

        Args:
            input_path (Path):
                The input file path where the TFRecord file is located.

        Returns:
            tf.data.TFRecordDataset:
                A tf.data.TFRecordDataset object containing the tokenized MIDI data.
        """
        return tf.data.TFRecordDataset(str(input_path))

    def _parse_tf_example(self, example_proto: tf.Tensor) -> tf.Tensor:
        """Parse a tf.train.Example object from the given serialized example.

        This function is used for parsing each record in the dataset.

        Args:
            example_proto (tf.Tensor):
                A serialized tf.train.Example object.

        Returns:
            tf.Tensor:
                A tf.Tensor object containing the tokenized MIDI data.

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
        # tf.io.VarLenFeature is useful when the feature is a variable-length list.
        # In this case, the "tokenized_midi" feature is a list of integers
        # where the length can vary for each example.
        # Hence, tf.io.VarLenFeature is used to handle this variability.
        feature_description = {"tokenized_midi": tf.io.VarLenFeature(tf.int64)}
        # The function is used to parse the serialized tf.train.Example.
        # It converts the serialized data into a tf.Tensor or tf.SparseTensor.
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        # "tokenized_midi" is a tf.SparseTensor due to the use of tf.io.VarLenFeature.
        # However, tf.SparseTensor can be difficult to handle in subsequent processes.
        # Therefore, we convert it to a dense tf.Tensor using tf.sparse.to_dense.
        return tf.sparse.to_dense(parsed_features["tokenized_midi"])
