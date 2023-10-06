"""Inference service for generating music events and writing them into MIDI files.

This module provides a service for generating music events from a trained model,
converting these events into their tokenized form,
and writing the tokenized data as a MIDI file.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf
import yaml

from generative_music.domain.inference import (GenerationParameters,
                                               Id2EventConverter, Sampler)
from generative_music.domain.midi_data_processor.midi_representation import \
    Config
from generative_music.domain.midi_data_processor.midi_tokenization import \
    Tokenizer
from generative_music.domain.midi_data_processor.postprocessor import \
    DataWriter
from generative_music.domain.model.transformer import Decoder
from generative_music.infrastructure.model_storage import CheckpointManager


class InferenceService:
    """A service for generating and converting music events using a trained model.

    This class manages the process of generating music events from a trained model
    and converting these events into a MIDI file.
    It uses a trained model loaded from a checkpoint and generates events.
    The generated events are then wrote into a MIDI file using a DataWriter instance.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        checkpoint_dir: str,
        checkpoint_number: int,
        params: GenerationParameters,
    ):
        """Initialize the InferenceService.

        Args:
            model (tf.keras.Model): The trained model to use for generation.
            checkpoint_dir (str): Directory where model checkpoints are stored.
            checkpoint_number (int): Specific checkpoint number to load.
            params (GenerationParameters): Parameters for the generation process.
        """
        self.model = model
        self.config = Config()
        self.midi_writer = DataWriter(self.config)
        self.tokenizer = Tokenizer(self.config)
        self.sampler = Sampler(self.tokenizer)
        self.id2event_converter = Id2EventConverter(self.tokenizer, self.config)
        optimizer = tf.keras.optimizers.Adam()
        self.checkpoint_manager = CheckpointManager(
            self.model, optimizer, checkpoint_dir
        )
        self.checkpoint_manager.restore_from_checkpoint(checkpoint_number)
        self.params = params

    def generate(self, file_path: Path):
        """Generate music events and write them to a MIDI file.

        The generation process uses the model to generate a sequence of events.
        The process ends when the target number of bars is reached
        or when the maximum length of the sequence is exceeded.
        At last, the process writes a sequence of events to a MIDI file.

        Args:
            file_path (Path): Path of the file to write the generated MIDI data.
        """
        event_ids = self.sampler.generate_initial_event_ids()
        events = []
        for event_id in event_ids:
            event = self.tokenizer.detokenize(event_id, 0)
            events.append(event)

        current_generated_bar = 0

        while current_generated_bar < self.params.n_target_bar:
            logits = self.model(tf.constant([event_ids]))
            logit = logits[-1, -1]
            event_id = self.sampler.sample_with_temperature(
                logits=logit, temperature=self.params.temperature, topk=self.params.topk
            )
            event_ids.append(event_id)
            # If it's a padding id, don't convert to an event
            if event_id == self.params.padding_id:
                continue
            event = self.id2event_converter.convert_id_to_events(
                event_id, current_generated_bar
            )
            events.append(event)
            # if bar event (only work for batch_size=1)
            if event_id == self.tokenizer.event2id["Bar_None"]:
                current_generated_bar += 1
            # Exit the while loop if the maximum length is exceeded
            if len(events) == self.params.max_length:
                print("The length of events has exceeded max length.")
                break
        self.midi_writer.write_midi_file(events, file_path)


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
    parser.add_argument("--n_target_bar", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    # Load the model config file
    cfg_model = load_config("generative_music/config/model.yml")[args.model_env]
    num_layers = cfg_model["num_layers"]
    d_model = cfg_model["d_model"]
    num_heads = cfg_model["num_heads"]
    ff_dim = cfg_model["ff_dim"]
    seq_len = cfg_model["seq_len"]

    # Load the dataset config file
    cfg_dataset = load_config("generative_music/config/dataset.yml")
    ckpt = cfg_dataset["paths"]["ckpt"]
    ckpt_base_dir = ckpt["base_dir"]
    ckpt_dir_name = ckpt["dir_name"]
    ckpt_path = f"{ckpt_base_dir}/{ckpt_dir_name}"
    ckpt_number = ckpt["number"]
    results_dir = Path(cfg_dataset["paths"]["result_data_dir"])
    if not results_dir.exists():
        results_dir.mkdir()

    # Load the JSON file
    event2id_data = load_json("generative_music/data/event2id.json")
    vocab_size = len(event2id_data) + 1
    padding_id = max(event2id_data.values()) + 1

    # Instantiate the model
    transformer_decoder = Decoder(
        num_layers, d_model, num_heads, ff_dim, vocab_size, seq_len
    )

    generation_parameters = GenerationParameters(
        args.n_target_bar,
        args.temperature,
        args.topk,
        vocab_size,
        padding_id,
        args.max_length,
    )
    inference_service = InferenceService(
        transformer_decoder, ckpt_path, ckpt_number, generation_parameters
    )

    present_time = datetime.now().strftime("%Y%m%d%H%M%S")
    result_path = results_dir / Path(f"{present_time}.mid")
    inference_service.generate(result_path)
