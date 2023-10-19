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

* Environment Requirements: **python 3.9, poetry 1.2.2**

### Prerequisites
#### Preparation for poetry env
```bash
# install dependencies
$ make init
```
#### Token Mapping
1. Modify the values assigned to variables in [config file](generative_music/domain/midi_data_processor/midi_representation/config.py) as needed.
   > Note: be aware that this may cause test code to fail, so adjust accordingly
2. Execute the following command.
```bash
# create id2event and event2id json files
$ make token-mapping
```
#### Dataset Creation
1. Download the MIDI data from [this link](https://github.com/YatingMusic/remi/tree/master#obtain-the-midi-data).
2. Move the train directory located inside the downloaded data directory to `generative_music/data`.
3. Rename the folder from `train` to `midis`.
4. Before proceeding, adjust certain parameters in the [dataset.yml](generative_music/config/dataset.yml) file as needed:
   - `ratios`:
     - These values determine the split ratio of the dataset into training, validation, and testing sets.
     - For example, a `train_ratio` of 0.8 means that 80% of the data will be used for training.
   - `data_augmentation`:
     - This section contains parameters for data augmentation.
     - `transpose_amounts` are the semitone shifts to be applied for transposition.
     - `stretch_factors` are the factors by which the tempo of the music will be stretched.
     > Note that the dataset will be augmented by multiplying the number of original MIDI files by the product of the lengths of `transpose_amounts` and `stretch_factors`.
5. Execute the following command.
```bash
# create the CSV file, the tfrecords, and split the dataset into train/val/test
$ make dataset
```

### Train
1. Modify the values assigned to variables in [config files](generative_music/config) as needed.
2. Execute the following command.
```bash
# start training based on config setting
# If you want to specify the model environment, pass it as an argument like `model_env=gpt-2`.
# If not specified, `test` will be used by default.
$ make train model_env=[your_model_environment]
```
#### Resume Training
If you want to resume training from a certain checkpoint,
you can specify the directory containing the checkpoint files as follows:
```bash
# resume training from a specific directory
$ make train resumed_dir=<directory_containing_your_checkpoint_files>
```
Please replace `<directory_containing_your_checkpoint_files>`
with the actual directory containing the checkpoint files.
The default path for the checkpoint directory is specified
in the [dataset.yml](generative_music/config/dataset.yml) file under the ckpt_dir key.

#### Monitor Training Progress with TensorBoard
You can monitor the progress of your training using TensorBoard.

This allows you to track changes in loss, learning rate, and various hyperparameters.
To start TensorBoard, execute the following command:

```bash
# Start TensorBoard
$ make tensorboard log_dir_name=<directory_containing_your_tensorboard_files>
```
Please replace `<directory_containing_your_tensorboard_files>`
with the actual tensorboard directory where your training logs are located.
[This](generative_music/data/tensorboard) is the initial setting path for the tensorboard directory.
If you have changed the save location for TensorBoard during training,
please specify the new directory using the `log_path` argument.

With TensorBoard, you can visualize different aspects of your model, such as:
- Training / validation losses
- Learning rate per step
- Hyperparameters such as:
  - Transformer decoder settings
  - Max sequence length
  - The number of epochs

This can be useful for tuning your model and improving its performance.

### Inference
1. Modify the values assigned to variables in [config files](generative_music/config) as needed.
2. Execute the following command.
```bash
# Run inference with default parameters
$ make inference
```
This command will work with the default parameter settings. The default values are as follows:
- `model_env`: The environment of the model. The default is `test`.
- `n_target_bar`: The number of bars to target. The default is `2`.
- `max_length`: The maximum length. The default is `512`.
- `temperature`: The temperature parameter. The default is `1.2`.
- `topk`: The value of k in top-k sampling. The default is `5`.

Each parameter can be changed as needed.
For example, if you want to set `n_target_bar` to `4` and `max_length` to `1024`, you can run the command as follows:
```bash
# Run inference with custom parameters (n_target_bar=4, max_length=1024)
$ make inference n_target_bar=4 max_length=1024
```
This allows you to freely adjust the output results.

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
│     ├ model_storage .. Storing model checkpoints and SavedModels
│     ├ tensorboard .. Storing data necessary for visualization with TensorBoard
│     └ tfrecords .. Writing and reading TensorFlow record files
│
└ tests .. Test package. The hierarchical structure below follows the main module
    ├ domain
    └ infrastructure
```

### Code Quality Management
#### Linter and Formatter
```bash
$ make lint
$ make format
```
#### Test
1. Implement [this](#token-mapping) only for the first time.
2. Execute the following command.
```bash
$ make test
```
