"""TrainService for training and validating models.

The purpose of this file is to provide a service for training and validating models.
This class manages the training and validation process,
utilizing training and validation data loaded from TensorFlow records using a DataLoader instance.
It is designed to be flexible and easy to use for a variety of different models.
"""
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

from generative_music.domain.dataset_preparation.batch_generation import \
    BatchGenerator
from generative_music.domain.model.transformer import Decoder
from generative_music.domain.train import (
    LabelSmoothedCategoricalCrossentropy, TrainStep,
    WarmupCosineDecayScheduler)
from generative_music.infrastructure.tfrecords import MidiTFRecordsReader


class DataLoader:
    """A utility for loading training and validation data from TensorFlow records.

    This class provides functionality to load training and validation data from TensorFlow records,
    and generate batches of sequences for model training.
    """

    def __init__(
        self,
        tfrecords_dir: Path,
        batch_size: int,
        seq_length: int,
        padding_id: int,
        bar_start_token_id: int,
        buffer_size: int,
    ):
        """Initialize the DataLoader instance.

        Args:
            tfrecords_dir (Path): The directory containing the TensorFlow records.
            batch_size (int): The number of sequences in a batch.
            seq_length (int): The length of each sequence in a batch.
            padding_id (int): The token ID used for padding.
            bar_start_token_id (int):
                The token ID used to indicate
                the start of a new bar (musical measure) in the sequence.
            buffer_size (int): The size of the buffer used for shuffling the dataset.
        """
        self.tfrecords_dir = tfrecords_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.padding_id = padding_id
        self.bar_start_token_id = bar_start_token_id
        self.buffer_size = buffer_size

    def load_train_data(self) -> tf.data.Dataset:
        """Load the training data from the TensorFlow records and generate batches.

        Returns:
            A tf.data.Dataset object representing the generated batches.
        """
        return self._load_data("train.tfrecords")

    def load_val_data(self) -> tf.data.Dataset:
        """Load the validation data from the TensorFlow records and generate batches.

        Returns:
            A tf.data.Dataset object representing the generated batches.
        """
        return self._load_data("val.tfrecords")

    def _load_data(self, filename) -> tf.data.Dataset:
        """Load the data from the TensorFlow records and generate batches.

        Args:
            filename (str): The name of the TensorFlow records file.

        Returns:
            A tf.data.Dataset object representing the generated batches.
        """
        # Instantiate the MidiTFRecordsReader
        tfrecords_reader = MidiTFRecordsReader()
        # Load the data from the TensorFlow records
        dataset = tfrecords_reader.create_dataset(self.tfrecords_dir / filename)
        # Instantiate the BatchGenerator for the data
        batch_generator = BatchGenerator(
            dataset,
            self.batch_size,
            self.seq_length,
            self.padding_id,
            self.bar_start_token_id,
            self.buffer_size,
        )
        # Generate the datasets
        data = batch_generator.generate_batches()
        return data


class TrainService:
    """A service for training and validating a model.

    This class manages the training and validation process of a model.
    It uses training and validation data loaded from TensorFlow records
    using a DataLoader instance.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        loss: tf.keras.losses.Loss,
        optimizer: tf.keras.optimizers.Optimizer,
        data_loader: DataLoader,
        epochs: int,
    ):
        """Initialize the TrainService instance.

        Args:
            model (tf.keras.Model): The model to be trained.
            loss (tf.keras.losses.Loss): The loss function for training the model.
            optimizer (tf.keras.optimizers.Optimizer):
                The optimizer for training the model.
            data_loader (DataLoader):
                The DataLoader instance for loading training and validation data.
            epochs (int): The number of epochs for training the model.
        """
        self.train_step = TrainStep(model, loss, optimizer)
        self.model = model
        self.loss = loss
        self.train_data = data_loader.load_train_data()
        self.val_data = data_loader.load_val_data()
        self.epochs = epochs

    def train(self):
        """Train and validate the model for a certain number of epochs.

        For each epoch, the model is trained using the training data,
        and then validated using the validation data.
        The average loss value for each epoch is printed.
        """
        for epoch in range(self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            train_loss = self._run_epoch(self.train_data, is_training=True)
            print(f"  - train loss: {train_loss:.4f}")
            val_loss = self._run_epoch(self.val_data)
            print(f"  - valid loss: {val_loss:.4f}")

    def _run_epoch(self, data, is_training=False) -> float:
        """Run one epoch of training or validation.

        Args:
            data (tf.data.Dataset): The training or validation data.
            is_training (bool):
                Whether to run training or validation. Default is False.

        Returns:
            float: The average loss value of the epoch.
        """
        total_loss = tf.constant(0.0)
        total_steps = 0
        progress_bar = tqdm(
            data, desc="Training" if is_training else "Validation", dynamic_ncols=True
        )
        for x_batch, y_batch, mask in progress_bar:
            if is_training:
                loss_value = self.train_step(x_batch, y_batch, mask)
            else:
                y_pred = self.model(x_batch)
                loss_value = self.loss(y_batch, y_pred)
            total_loss += loss_value
            total_steps += 1
            progress_bar.set_postfix(
                {"loss": total_loss.numpy() / total_steps}, refresh=True
            )
        progress_bar.close()
        return total_loss / total_steps


if __name__ == "__main__":
    # Set the hyper parameter
    # TODO Fetch via Args or implement config
    # num_layers = 12
    # d_model = 768
    # num_heads = 12
    # ff_dim = 1024
    # seq_len = 2048
    num_layers = 4
    d_model = 128
    num_heads = 8
    ff_dim = 512
    seq_len = 512
    batch_size = 4
    epochs = 100
    # The number of score files for the train.
    buffer_size = 542
    data_dir = Path("generative_music/data/tfrecords")
    # TODO Automatically retrieve from json file
    vocab_size = 335
    bar_start_token_id = 0
    padding_id = 334
    # Instantiate the model
    transformer_decoder = Decoder(
        num_layers, d_model, num_heads, ff_dim, vocab_size, seq_len
    )
    # Instantiate the loss
    loss = LabelSmoothedCategoricalCrossentropy(
        masked_id=vocab_size, vocab_size=vocab_size
    )
    # Instantiate the optimizer with a custom learning rate scheduler
    lr_scheduler = WarmupCosineDecayScheduler()
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    data_loader = DataLoader(
        data_dir, batch_size, seq_len, padding_id, bar_start_token_id, buffer_size
    )
    train_service = TrainService(
        transformer_decoder, loss, optimizer, data_loader, epochs
    )
    train_service.train()
