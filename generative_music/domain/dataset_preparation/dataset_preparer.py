"""A module for preparing datasets from MIDI files."""
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

from generative_music.domain.dataset_preparation.data_augmented_param_generator import \
    DataAugmentedParamGenerator
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
        csv_filepath: Path,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        train_basename: str = "train",
        val_basename: str = "validation",
        test_basename: str = "test",
        data_transpose_amounts: List[int] = [0],
        data_stretch_factors: List[float] = [1.0],
    ):
        """Initialize the DatasetPreparer instance.

        Args:
            data_dir (Path): The directory containing the MIDI files.
            midi_config (Config): The configuration object for MIDI processing.
            csv_filepath (Path):
                The path where the CSV file containing the split information will be saved.
            train_ratio (float): The ratio of the dataset to be used for training.
            val_ratio (float): The ratio of the dataset to be used for validation.
            test_ratio (float): The ratio of the dataset to be used for testing.
            train_basename (str, optional):
                The base name of the training file (without extension).
                Default is "train".
            val_basename (str, optional):
                The base name of the validation file (without extension).
                Default is "validation".
            test_basename (str, optional):
                The base name of the test file (without extension).
                Default is "test".
            data_transpose_amounts (List[int], optional):
                A list of integer values to shift the pitch
                of the MIDI files for data augmentation.
                Each integer represents the number of semitones to shift.
                Default is [0], meaning no shift.
            data_stretch_factors (List[float], optional):
                A list of float values to stretch or shrink the tempo
                of the MIDI files for data augmentation.
                Each float represents the factor by which to stretch the tempo.
                Default is [1.0], meaning no change in tempo.
        """
        self.splitter = DatasetSplitter(
            data_dir,
            csv_filepath,
            train_ratio,
            val_ratio,
            test_ratio,
            train_basename,
            val_basename,
            test_basename,
        )
        self.midi_config = midi_config
        self.train_basename = train_basename
        self.val_basename = val_basename
        self.test_basename = test_basename
        # Initialize the tokenizer
        self.tokenizer = Tokenizer(self.midi_config)
        # Initialize the data augmentor
        self.data_augmented_param_generator = DataAugmentedParamGenerator(
            data_transpose_amounts, data_stretch_factors
        )

    def prepare(self) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        """Split the MIDI files into train, validation and test datasets, and tokenize.

        Returns:
            Tuple[List[List[int]], List[List[int]], List[List[int]]]:
                The train, validation, and test datasets as lists of tokenized events.
        """
        split_data = self.splitter.split_data()
        # Write the filepaths and their corresponding splits to a CSV file
        self.splitter.create_split_csv(split_data)

        # Tokenize the data
        tokenized_data = {}
        for split, filepaths in split_data.items():
            tokenized_data[split] = self._process_files(filepaths, split)

        return (
            tokenized_data[self.train_basename],
            tokenized_data[self.val_basename],
            tokenized_data[self.test_basename],
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
        for filepath in tqdm(filepaths, desc=f"Processing {split} MIDI files"):
            preprocessor = Preprocessor(Path(filepath), self.midi_config)
            # data augmentation only in the case of training
            if split == self.train_basename:
                for (
                    shift,
                    stretch,
                ) in self.data_augmented_param_generator.generate():
                    events = preprocessor.generate_events(
                        is_augmented=True, shift=shift, stretch=stretch
                    )
                    tokenized_events = [
                        self.tokenizer.tokenize(event) for event in events
                    ]
                    tokenized_data.append(tokenized_events)
            else:
                events = preprocessor.generate_events()
                tokenized_events = [self.tokenizer.tokenize(event) for event in events]
                tokenized_data.append(tokenized_events)

        return tokenized_data
