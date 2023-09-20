# MIDI Music Generation:
1. Using the predetermined MIDI representation settings, map each representation to an ID for training purposes.
2. Split the data into train/validation/test sets and convert each into tfrecords.
3. Train to be able to generate MIDI data from scratch.

---

- [MIDI Music Generation](#midi-note-generation)
  - [Getting started](#getting-started)
    - [Prerequisites](#prerequisites)
      - [Preparation for poetry env](#preparation-for-poetry-env)
      - [Token Mapping](#token-mapping)
      - [Dataset Creation](#sataset-creation)
    - [Train](#train)
  - [Description of each directory](#description-of-each-directory)
  - [Code Quality Management](#code-quality-management)
    - [Linter and Formatter](#linter-and-formatter)
    - [Test](#test)

## Getting Started

* Environment Requirements: **python 3.11.1, poetry 1.2.2**

### Prerequisites
#### Preparation for poetry env
``` bash
# install dependencies
$ make init
```
#### Token Mapping
1. Modify the values assigned to variables in [config file](generative_music/domain/midi_data_processor/midi_representation/config.py) as needed.
   > Note: be aware that this may cause test code to fail, so adjust accordingly
2. Execute the following command.
``` bash
# create id2event and event2id json files
$ make token-mapping
```
#### Dataset Creation
1. Download the MIDI data from [this link](https://github.com/YatingMusic/remi/tree/master#obtain-the-midi-data).
2. Move the train directory located inside the downloaded data directory to `generative_music/data`.
3. Rename the folder from `train` to `midis`.
4. Execute the following command.
``` bash
# # Create the CSV file the TensorFlow records for dataset split (train/val/test)
$ make dataset
```

### Train
1. Modify the values assigned to variables in [config files](generative_music/config) as needed.
2. Execute the following command.
``` bash
# start training based on config setting
$ make train
```
#### Resume Training
If you want to resume training from a certain checkpoint, you can specify the resumed_dir as follows:
```  bash
# resume training from a specific directory
$ make train resumed_dir=<path_to_your_directory>
```
Please replace <path_to_your_directory> with the actual path to the directory
where your training checkpoint is located.

## Description of each directory
```
repository TOP
│
├ generative_music .. Package containing the main features of this module
│  ├ config .. Configuration yaml files used during training
│  │
│  ├ data .. Storage for pre-acquired MIDI data and data written out for training and test
│  │
│  ├ domain .. Package for granular parts that do not depend on other features
│  │  ├ dataset_preparation .. Preparing and splitting the dataset
│  │  │ └ batch_generation .. Preparing and processing datasets to be used in training
│  │  │
│  │  ├ midi_data_processor .. Functions and settings for handling MIDI data
│  │  │ ├ midi_representation .. Settings for handling MIDI for training purposes
│  │  │ ├ midi_tokenization .. Mapping of MIDI event information to token IDs
│  │  │ ├ postprocessor .. Writing event information to MIDI
│  │  │ └ preprocessor .. Converting MIDI files to token IDs
│  │  │
│  │  ├ model .. Model architecture
│  │  │ ├ transformer .. Transformer model
│  │  │ └ utils .. Neural network model utilities
│  │  │
│  │  └ train .. learning rate scheduler, loss function, training step and loading train data
│  │
│  ├ service .. Combining granular parts and bundling them for specific purposes
│  │
│  └ infrastructure .. Handling external resources and data storage
│     └ tfrecords .. Writing and reading TensorFlow record files
│
└ tests .. Test package. The hierarchical structure below follows the main module
    ├ domain
    └ infrastructure
```

### Code Quality Management
#### Linter and Formatter
``` bash
$ make lint
$ make format
```
#### Test
1. Implement [this](#token-mapping) only for the first time.
2. Execute the following command.
``` bash
$ make test
```
