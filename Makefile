ROOT_DIR := ./
SRC_DIR := ./src
VENV = venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: all setup data features model visualizations clean

all: data features model visualizations clean

$(VENV)/bin/activate:
	python -m venv venv
	venv/bin/pip install $(ROOT_DIR)

data: $(VENV)/bin/activate
	venv/bin/python src/data/make_dataset.py data/raw/raw_full_dataset.csv

features: data
	venv/bin/python $(SRC_DIR)/features/build_features.py

model: features
	venv/bin/python $(SRC_DIR)/models/validate_model.py reports/report_local.csv

visualizations: model
	PYTHON $(SRC_DIR)/visualization/visualize.py reports/figures/figures_local

clean:
	rm -rf __pycache__
	rm -rf $(VENV)
