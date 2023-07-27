"""A module for preparing datasets from MIDI files."""
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from generative_music.domain.dataset_preparation.dataset_splitter import \
    DatasetSplitter
from generative_music.domain.midi_data_processor.midi_representation import \
    Config
from generative_music.domain.midi_data_processor.midi_tokenization import \
    Tokenizer
from generative_music.domain.midi_data_processor.preprocessor import \
    Preprocessor


class DatasetPreparer:
    """A class for preparing datasets from MIDI files.

    This class preprocesses and tokenizes MIDI files
    and splits them into train, validation, and test datasets.
    """

    def __init__(
        self,
        data_dir: Path,
        midi_config: Config,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ):
        """Initialize the DatasetPreparer instance.

        Args:
            data_dir (Path): The directory containing the MIDI files.
            midi_config (Config): The configuration object for MIDI processing.
            train_ratio (float): The ratio of the dataset to be used for training.
            val_ratio (float): The ratio of the dataset to be used for validation.
            test_ratio (float): The ratio of the dataset to be used for testing.
        """
        self.splitter = DatasetSplitter(data_dir, train_ratio, val_ratio, test_ratio)
        self.midi_config = midi_config

    def prepare(
        self, csv_filepath: Path
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        """Split the MIDI files into train, validation and test datasets, and tokenize.

        Args:
            csv_filepath (Path):
                The path where the CSV file containing the split information will be saved.

        Returns:
            Tuple[List[List[int]], List[List[int]], List[List[int]]]:
                The train, validation, and test datasets as lists of tokenized events.
        """
        split_data = self.splitter.split_data()
        # Write the filepaths and their corresponding splits to a CSV file
        self.splitter.create_split_csv(split_data, csv_filepath)

        # Tokenize the data
        tokenized_data = {}
        for split, filepaths in split_data.items():
            tokenized_data[split] = self._process_files(filepaths, split)

        return (
            tokenized_data["train"],
            tokenized_data["validation"],
            tokenized_data["test"],
        )

    def _process_files(self, filepaths: List[str], split: str) -> List[List[int]]:
        """Preprocess and tokenize the MIDI files.

        Args:
            filepaths (List[str]):
                A list of filepaths for the MIDI files to be processed.
            split (str):
                The dataset split currently being processed (train/validation/test).

        Returns:
            List[List[int]]: A list of tokenized events for each MIDI file.
        """
        tokenized_data = []
        tokenizer = Tokenizer(self.midi_config)
        for filepath in tqdm(filepaths, desc=f"Processing {split} MIDI files"):
            preprocessor = Preprocessor(Path(filepath), self.midi_config)
            events = preprocessor.process()
            tokenized_events = [tokenizer.tokenize(event) for event in events]
            tokenized_data.append(tokenized_events)
        return tokenized_data
