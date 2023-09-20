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
train:
ifeq ($(resumed_dir),)
	poetry run python generative_music/service/train_service.py --model_env test
else
	poetry run python generative_music/service/train_service.py --model_env test --resumed_dir $(resumed_dir)
endif
