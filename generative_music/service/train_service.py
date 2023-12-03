"""TrainService for training and validating models.

The purpose of this file is to provide a service for training and validating models.
This class manages the training and validation process,
utilizing training and validation data loaded from TensorFlow records using a DataLoader instance.
It is designed to be flexible and easy to use for a variety of different models.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
import yaml
from tqdm import tqdm

from generative_music.domain.model.transformer import Decoder
from generative_music.domain.train import (
    EpochStepsCalculator, GroupwiseLossMetrics,
    LabelSmoothedCategoricalCrossentropy, TrainDataLoader, TrainStep,
    WarmupCosineDecayScheduler)
from generative_music.infrastructure.model_storage import CheckpointManager
from generative_music.infrastructure.tensorboard import TensorboardWriter


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
        tensorboard_dir: str,
        checkpoint_dir: str,
        epoch_steps_calculator: EpochStepsCalculator,
        events_loss: GroupwiseLossMetrics,
        cfg_model: Dict[str, Any],
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
            tensorboard_dir (str):
                The directory where the tensorboard will be saved.
            checkpoint_dir (str):
                The directory where the checkpoints will be saved.
            epoch_steps_calculator (EpochStepsCalculator):
                The instance used to calculate the total steps
                needed for each epoch of training and validation.
            events_loss (GroupwiseLossMetrics): The total loss for each group.
            cfg_model (Dict[str, Any]):
                A dictionary containing the configuration parameters for the model.
        """
        self.train_step = TrainStep(model, loss, optimizer)
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_data = data_loader.load_train_data()
        self.val_data = data_loader.load_val_data()
        self.epochs = epochs
        self.checkpoint_manager = CheckpointManager(model, optimizer, checkpoint_dir)
        self.start_epoch = self.checkpoint_manager.get_epoch()
        self.train_total_steps = epoch_steps_calculator.train_total_steps
        self.val_total_steps = epoch_steps_calculator.val_total_steps
        self.events_loss = events_loss
        self.tensorboard_dir = tensorboard_dir
        self.cfg_model = cfg_model
        (
            self.tensorboard_train_writer,
            self.tensorboard_val_writer,
            self.tensorboard_events_writers,
        ) = self._create_tensorboards()

    def train(self):
        """Train and validate the model for a certain number of epochs.

        For each epoch, the model is trained using the training data,
        and then validated using the validation data.
        The average loss value for each epoch is printed.
        After each epoch, the model's state is saved as a checkpoint.
        """
        for epoch in range(self.start_epoch, self.epochs):
            print(f"Epoch {epoch + 1}/{self.epochs}")
            train_loss = self._run_epoch(self.train_data, epoch + 1, is_training=True)
            print(f"  - train loss: {train_loss:.4f}")
            val_loss = self._run_epoch(self.val_data, epoch + 1)
            print(f"  - valid loss: {val_loss:.4f}")
            self.checkpoint_manager.save(epoch + 1)

    def _create_tensorboards(
        self,
    ) -> Tuple[TensorboardWriter, TensorboardWriter, List[TensorboardWriter]]:
        """Create tensorboards for the training and validation phases.

        This method creates separate TensorboardWriter instances for the training phase,
        validation phase, and for each event in the events_loss attribute.

        Returns:
            TensorboardWriter:
                The TensorboardWriter instance for writing scalar values
                and hyperparameters to TensorBoard logs for training.
            TensorboardWriter:
                The TensorboardWriter instance for writing scalar values
                to TensorBoard logs for validation.
            List[TensorboardWriter]:
                A list of TensorboardWriter instances for writing scalar values
                to TensorBoard logs for each event in the events_loss attribute.
        """
        tensorboard_train_writer = self._create_tensorboard_writer(
            "train", self.cfg_model
        )
        tensorboard_val_writer = self._create_tensorboard_writer("val")
        tensorboard_events_writers = [
            self._create_tensorboard_writer(event_name)
            for event_name in self.events_loss.event_names
        ]
        return (
            tensorboard_train_writer,
            tensorboard_val_writer,
            tensorboard_events_writers,
        )

    def _create_tensorboard_writer(
        self, sub_dir: str, hyperparameters: Optional[Dict[str, int]] = None
    ) -> TensorboardWriter:
        """Create a TensorboardWriter instance.

        Args:
            sub_dir (str): Subdirectory for the Tensorboard logs.
            hyperparameters (Dict[str, Any], optional):
                A dictionary containing the configuration parameters for the model.
                Deault is None.

        Returns:
            TensorboardWriter: The TensorboardWriter instance.
        """
        return TensorboardWriter(
            f"{self.tensorboard_dir}/{sub_dir}", hyperparameters=hyperparameters
        )

    def _run_epoch(self, data, epoch: int, is_training=False) -> float:
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
            y_pred, loss_value = self._compute_loss(x_batch, y_batch, mask, is_training)
            self.events_loss.update_state(y_batch, y_pred)
            total_loss += loss_value
            total_steps += 1
            progress_bar.set_postfix(
                {"loss": total_loss.numpy() / total_steps}, refresh=True
            )
        progress_bar.close()
        average_loss = total_loss / total_steps
        self._write_tensorboard(average_loss, epoch, is_training)

        event_loss_name = "loss/events_train" if is_training else "loss/events_val"
        total_group_losses = self.events_loss.result(total_steps)
        for i in range(self.events_loss.num_groups - 1):
            self.tensorboard_events_writers[i].write_scalar(
                event_loss_name, total_group_losses[i], epoch
            )
        self.events_loss.reset_states()
        return total_loss / total_steps

    def _compute_loss(
        self, x_batch: tf.Tensor, y_batch: tf.Tensor, mask: tf.Tensor, is_training: bool
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute the loss for the given batch.

        Args:
            x_batch (tf.Tensor): The input batch.
            y_batch (tf.Tensor): The target batch.
            mask (tf.Tensor): The mask for the batch.
            is_training (bool): Whether the model is in training mode.

        Returns:
            Tuple[tf.Tensor, tf.Tensor]: The predictions and the computed loss.
        """
        if is_training:
            y_pred, loss_value = self.train_step(x_batch, y_batch, mask, is_training)
            step_num = self.optimizer.iterations.numpy()
            current_lr = self.optimizer.learning_rate(step_num)
            self.tensorboard_train_writer.write_scalar(
                "learning_rate", current_lr, step_num
            )
        else:
            y_pred = self.model(x_batch)
            loss_value = self.loss(y_batch, y_pred)
        return y_pred, loss_value

    def _write_tensorboard(self, average_loss: float, epoch: int, is_training: bool):
        """Write the average loss to the appropriate TensorBoard.

        Args:
            average_loss (float): The average loss to write.
            epoch (int): The current epoch.
            is_training (bool): Whether the model is in training mode.
        """
        if is_training:
            self.tensorboard_train_writer.write_scalar("loss", average_loss, epoch)
            return
        self.tensorboard_val_writer.write_scalar("loss", average_loss, epoch)


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
    parser.add_argument("--resumed_dir", type=str)
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
    ckpt_base_dir = cfg_dataset["paths"]["ckpt"]["base_dir"]
    tensorboard_base_dir = cfg_dataset["paths"]["tensorboard_dir"]
    midi_data_dir = Path(cfg_dataset["paths"]["midi_data_dir"])
    train_ratio = cfg_dataset["ratios"]["train_ratio"]
    val_ratio = cfg_dataset["ratios"]["val_ratio"]
    data_transpose_amounts = cfg_dataset["data_augmentation"]["transpose_amounts"]
    data_stretch_factors = cfg_dataset["data_augmentation"]["stretch_factors"]

    epoch_steps_calculator = EpochStepsCalculator(
        midi_data_dir,
        train_ratio,
        val_ratio,
        batch_size,
        data_transpose_amounts,
        data_stretch_factors,
    )

    # Tensorboard initialization
    # Setting for new training
    current_time = datetime.now().strftime("%y%m%d%H%M")
    tensorboard_dir = f"{tensorboard_base_dir}/{current_time}_{args.model_env}"
    ckpt_dir = f"{ckpt_base_dir}/{current_time}_{args.model_env}"
    # If resuming training from a previous state, update tensorboard and ckpt directory.
    if args.resumed_dir:
        tensorboard_dir = f"{tensorboard_base_dir}/{args.resumed_dir}"
        ckpt_dir = f"{ckpt_base_dir}/{args.resumed_dir}"

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
        masked_id=vocab_size - 1, vocab_size=vocab_size
    )
    events_loss = GroupwiseLossMetrics(vocab_size - 1, vocab_size, json_data)
    # Instantiate the optimizer with a custom learning rate scheduler
    # Here, the masked_id is set to vocab_size - 1, because the IDs are assigned
    # sequentially from 0 and the last ID is vocab_size - 1.
    train_total_steps = epoch_steps_calculator.train_total_steps * epochs
    lr_scheduler = WarmupCosineDecayScheduler(
        warmup_steps=int(train_total_steps * 0.2), total_steps=train_total_steps
    )
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
        tensorboard_dir,
        ckpt_dir,
        epoch_steps_calculator,
        events_loss,
        cfg_model,
    )
    train_service.train()
