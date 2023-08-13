"""Tests for the MidiTFRecordsReader used for reading and parsing TFRecord files."""
import pytest
import tensorflow as tf

from generative_music.infrastructure.tfrecords.midi_tfrecords_reader import \
    MidiTFRecordsReader
from generative_music.infrastructure.tfrecords.midi_tfrecords_writer import \
    MidiTFRecordsWriter


class TestMidiTFRecordsReader:
    """A test class for the MidiTFRecordsReader."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Initialize the MidiTFRecordsReader tests.

        Set up the test environment by writing a sample TFRecord file
        and initializing the MidiTFRecordsReader instance.
        This method is called before each test function is executed.
        """
        self.tokenized_midis = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15],
        ]
        self.tfrecord_file = tmp_path / "test.tfrecord"
        writer = MidiTFRecordsWriter()
        writer.write_tfrecords(self.tokenized_midis, self.tfrecord_file)
        self.reader = MidiTFRecordsReader()

    def test_read_tfrecords(self):
        """Test if the TFRecord file is correctly read.

        Check if the returned object is an instance of tf.data.TFRecordDataset.
        """
        dataset = self.reader._read_tfrecords(self.tfrecord_file)
        assert isinstance(dataset, tf.data.TFRecordDataset)

    def test_parse_tf_example(self):
        """Test if the tf.train.Example object is parsed correctly.

        Check if the returned object is an instance of tf.Tensor.
        """
        raw_dataset = tf.data.TFRecordDataset(str(self.tfrecord_file))
        example_proto = next(iter(raw_dataset))
        parsed_tensor = self.reader._parse_tf_example(example_proto)
        assert isinstance(parsed_tensor, tf.Tensor)

    def test_create_dataset(self):
        """Test if the dataset is correctly created from the TFRecord file.

        Check if the parsed MIDI data read from the dataset
        matches the original tokenized_midis.
        """
        dataset = self.reader.create_dataset(self.tfrecord_file)
        parsed_midis = []
        # Since TensorFlow's Eager Execution is enabled,
        # each element of the dataset is returned as an EagerTensor
        # when we iterate over the dataset.
        for tensor in dataset:
            parsed_midis.append(tensor.numpy().tolist())
        assert parsed_midis == self.tokenized_midis
