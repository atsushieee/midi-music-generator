.PHONY: init
init:
	poetry install

.PHONY: update-check
update-check:
	poetry show --outdated

.PHONY: lint
lint:
	poetry run pflake8 generative_music tests
	poetry run mypy generative_music tests
	poetry run pydocstyle generative_music tests

.PHONY: format
format:
	poetry run black generative_music tests
	poetry run isort generative_music tests

.PHONY: test
test:
	poetry run pytest -vs tests

.PHONY: token-mapping
token-mapping:
	poetry run python generative_music/service/midi_mapping_service.py

.PHONY: dataset
dataset:
	poetry run python generative_music/service/dataset_service.py

.PHONY: train
model_env ?= test
train:
ifeq ($(resumed_dir),)
	poetry run python generative_music/service/train_service.py --model_env $(model_env)
else
	poetry run python generative_music/service/train_service.py --model_env $(model_env) --resumed_dir $(resumed_dir)
endif

.PHONY: tensorboard
log_path ?= generative_music/data/tensorboard
tensorboard:
	poetry run tensorboard --logdir $(log_path)/$(log_dir_name)
