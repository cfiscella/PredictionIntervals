ROOT_DIR := ./
SRC_DIR := ./src

.PHONY: all setup data features model visualizations

all: data features model visualizations

setup:
	pip install PredictionIntervals

data: setup
	python $(SRC_DIR)/data/make_dataset.py data/raw/raw_full_dataset.csv

features: data
	python $(SRC_DIR)/features/build_features.py

model: features
	python $(SRC_DIR)/models/validate_model.py reports/report_local.csv

visualizations: model
	python $(SRC_DIR)/visualization/visualize.py reports/figures/figures_local
