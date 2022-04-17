ROOT_DIR:=./
SRC_DIR:=./src

.PHONY: all setup data features model visualizations

all: data features model visualizations

setup: requirements.txt
  pip install -r $(ROOT_DIR)/requirements.txt

data: make_dataset.py
  python $(SRC_DIR)/make_dataset.py

features: data build_features.py
  python $(SRC_DIR)/build_features.py

model: features validate_model.py
  python $(SRC_DIR)/validate_model.py

visualizations: model visualize.py
  python $(SRC_DIR)/visualize.py
