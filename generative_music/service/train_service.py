"""TrainService for training and validating models.

The purpose of this file is to provide a service for training and validating models.
This class manages the training and validation process,
utilizing training and validation data loaded from TensorFlow records using a DataLoader instance.
It is designed to be flexible and easy to use for a variety of different models.
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf
import yaml
from tqdm import tqdm

from generative_music.domain.model.transformer import Decoder
from generative_music.domain.train import (
    EpochStepsCalculator, LabelSmoothedCategoricalCrossentropy,
    TrainDataLoader, TrainStep, WarmupCosineDecayScheduler)
from generative_music.infrastructure.model_storage import CheckpointManager


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
        data_loader: TrainDataLoader,
        epochs: int,
        checkpoint_dir: str,
        epoch_steps_calculator: EpochStepsCalculator,
    ):
        """Initialize the TrainService instance.

        Args:
            model (tf.keras.Model): The model to be trained.
            loss (tf.keras.losses.Loss): The loss function for training the model.
            optimizer (tf.keras.optimizers.Optimizer):
                The optimizer for training the model.
            data_loader (TrainDataLoader):
                The DataLoader instance for loading training and validation data.
            epochs (int): The number of epochs for training the model.
            checkpoint_dir (str):
                The directory where the checkpoints will be saved.
            epoch_steps_calculator (EpochStepsCalculator):
                The instance used to calculate the total steps
                needed for each epoch of training and validation.
        """
        self.train_step = TrainStep(model, loss, optimizer)
        self.model = model
        self.loss = loss
        self.train_data = data_loader.load_train_data()
        self.val_data = data_loader.load_val_data()
        self.epochs = epochs
        self.checkpoint_manager = CheckpointManager(model, optimizer, checkpoint_dir)
        self.start_epoch = self.checkpoint_manager.get_epoch()
        self.train_total_steps = epoch_steps_calculator.train_total_steps
        self.val_total_steps = epoch_steps_calculator.val_total_steps

    def train(self):
        """Train and validate the model for a certain number of epochs.

        For each epoch, the model is trained using the training data,
        and then validated using the validation data.
        The average loss value for each epoch is printed.
        After each epoch, the model's state is saved as a checkpoint.
        """
        for epoch in range(self.start_epoch, self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            train_loss = self._run_epoch(self.train_data, is_training=True)
            print(f"  - train loss: {train_loss:.4f}")
            val_loss = self._run_epoch(self.val_data)
            print(f"  - valid loss: {val_loss:.4f}")
            self.checkpoint_manager.save(epoch + 1)

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
            data,
            total=self.train_total_steps if is_training else self.val_total_steps,
            desc="Training" if is_training else "Validation",
            dynamic_ncols=True,
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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a configuration file in YAML format.

    Args:
    config_path (str): The path to the configuration file.

    Returns:
    Dict: The loaded configuration.
    """
    with open(config_path, "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def load_json(json_path: str) -> Dict[str, int]:
    """Load a JSON file.

    Args:
    json_path (str): The path to the JSON file.

    Returns:
    Dict: The loaded JSON data.
    """
    with open(json_path, "r") as jsonfile:
        data = json.load(jsonfile)
    return data


if __name__ == "__main__":
    # Set the hyper parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_env", type=str, default="test")
    args = parser.parse_args()

    # Load the model config file
    cfg_model = load_config("generative_music/config/model.yml")[args.model_env]
    num_layers = cfg_model["num_layers"]
    d_model = cfg_model["d_model"]
    num_heads = cfg_model["num_heads"]
    ff_dim = cfg_model["ff_dim"]
    seq_len = cfg_model["seq_len"]
    batch_size = cfg_model["batch_size"]
    epochs = cfg_model["epochs"]
    # The number of score files for the train.
    buffer_size = cfg_model["buffer_size"]

    # Load the dataset config file
    cfg_dataset = load_config("generative_music/config/dataset.yml")
    tfrecords_dir = Path(cfg_dataset["paths"]["tfrecords_dir"])
    train_basename = cfg_dataset["dataset_basenames"]["train"]
    val_basename = cfg_dataset["dataset_basenames"]["val"]
    ckpt_dir = cfg_dataset["paths"]["ckpt_dir"]
    midi_data_dir = Path(cfg_dataset["paths"]["midi_data_dir"])
    train_ratio = cfg_dataset["ratios"]["train_ratio"]
    val_ratio = cfg_dataset["ratios"]["val_ratio"]
    epoch_steps_calculator = EpochStepsCalculator(
        midi_data_dir, train_ratio, val_ratio, batch_size
    )

    # Load the JSON file
    json_data = load_json("generative_music/data/event2id.json")
    vocab_size = len(json_data) + 1
    bar_start_token_id = json_data["Bar_None"]
    padding_id = max(json_data.values()) + 1
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
    data_loader = TrainDataLoader(
        batch_size,
        seq_len,
        padding_id,
        bar_start_token_id,
        buffer_size,
        tfrecords_dir,
        train_basename,
        val_basename,
    )
    train_service = TrainService(
        transformer_decoder,
        loss,
        optimizer,
        data_loader,
        epochs,
        ckpt_dir,
        epoch_steps_calculator,
    )
    train_service.train()
