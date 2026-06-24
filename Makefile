.PHONY: install test lint prepare-data train-sft train-grpo eval aggregate plots

install:
	python -m pip install -e ".[train,chem,dev]"

test:
	python -m pytest

lint:
	python -m ruff check src tests

prepare-data:
	bash scripts/01_download_data.sh
	bash scripts/02_prepare_sft.sh

train-sft:
	bash scripts/03_train_sft.sh

train-grpo:
	bash scripts/04_train_grpo.sh

eval:
	bash scripts/06_eval_model.sh

aggregate:
	bash scripts/07_collect_results.sh

plots:
	bash scripts/08_make_plots.sh

