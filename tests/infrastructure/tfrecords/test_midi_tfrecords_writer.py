"""Tests for the TFRecordsWriter used for creating and writing TFRecord files."""
from pathlib import Path

import tensorflow as tf

from generative_music.infrastructure.tfrecords.midi_tfrecords_writer import \
    MidiTFRecordsWriter


class TestTFRecordsWriter:
    """A test class for the TFRecordsWriter."""

    def setup_method(self):
        """Initialize the MidiTFRecordsWriter tests.

        Set up the test environment by initializing the TFRecordsWriter instance.
        This method is called before each test function is executed.
        """
        self.tf_records_writer = MidiTFRecordsWriter()

    def test_create_tf_example(self):
        """Test if the tf.train.Example object is created correctly.

        Check if the returned tf.train.Example object is created correctly
        and if its features match the input tokenized_midi.
        """
        tokenized_midi = [42, 17, 58, 93, 71]
        tf_example = self.tf_records_writer._create_tf_example(tokenized_midi)
        assert isinstance(tf_example, tf.train.Example)
        assert (
            tf_example.features.feature["tokenized_midi"].int64_list.value
            == tokenized_midi
        )

    def test_write_tfrecords(self, tmp_path: Path):
        """Test if the TFRecord file is wrote correctly.

        Check if the TFRecord file is written correctly to the output path
        and if the data read from the file matches the original tokenized_midis.

        Args:
            tmp_path (Path): The temporary directory path provided by the pytest fixture.
        """
        tokenized_midis = [
            [42, 17, 58, 93, 71],
            [33, 56, 18, 19, 85],
            [64, 28, 37, 92, 11],
        ]
        output_path = tmp_path / "test.tfrecord"
        self.tf_records_writer.write_tfrecords(tokenized_midis, output_path)
        assert output_path.exists()
        # Read the records and compare them with the original data
        read_records = []
        for record in tf.data.TFRecordDataset(str(output_path)):
            example = tf.train.Example()
            example.ParseFromString(record.numpy())
            read_tokens = example.features.feature["tokenized_midi"].int64_list.value
            read_records.append(list(read_tokens))
        assert read_records == tokenized_midis
