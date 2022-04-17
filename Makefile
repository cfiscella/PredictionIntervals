all: data features model visualizations

data: make_dataset.py
  python make_dataset.py

features: data 

model: 

visualizations: visualizations.py
  python visualizations
