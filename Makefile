all: data features model visualizations

data: make_dataset.py
  python make_dataset.py

features: data build_features.py
  python build_features.py

model: 

visualizations: visualizations.py
  python visualizations
