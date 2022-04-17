.PHONY: all setup data features model visualizations

all: data features model visualizations

setup: requirements.txt
  pip install -r requirements.txt

data: make_dataset.py
  python make_dataset.py

features: data build_features.py
  python build_features.py

model: features validate_model.py
  python validate_model.py

visualizations: model visualize.py
  python visualize.py
