# PredictionIntervals

Stock Price Prediction Intervals w/ LSTM and Fuzzy Clustering

## Installation

```bash
$ pip install PredictionIntervals
```

## Usage

### Fuzzy Interval
`PredictionIntervals` can be used to predict upper and lower bounds
for time series as follows:

```python
from PredictionIntervals.models import fuzzy_interval
import pandas as pd
import keras


X_train = pd.read_csv("x_train.csv")  # data importing
y_train = pd.read_csv("y_train.csv")
X_test= pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv")

#sub-models
regressor_model = [keras.layers.Bidirectional(keras.layers.LSTM(units=256, 
return_sequences=False,dropout = .2)),keras.layers.Dense(units=1)]

interval_model = [keras.layers.Bidirectional(keras.layers.LSTM(units=256, 
return_sequences=False,dropout = .2)),keras.layers.Dense(units=2)]

#compile dictionaries are arguments fed to compile method for keras.models
regress_compile_dict ={"optimizer":'adam', "loss":'mean_squared_error',
"metrics":["mean_absolute_percentage_error"]}

interval_compile_dict={"loss":"mean_squared_error","optimizer":"adam"}

#model dictionaries are arguments fed to fit method for keras.models
regress_model_dict={"epochs":50}

interval_model_dict = {"epochs":50,"batch_size":32}


fuzzy_test = fuzzy_interval(regressor_model,regressor_window=5,regress_compile_dict,
                regress_model_dict,clusters=3, interval_model,interval_window = 5,
                interval_compile_dict,interval_model_dict,cluster_alpha=.05)
                
fuzzy_test.fit(X_train,y_train)
predictions = fuzzy_test.predict(X_test)

#measures % of values covered by intervals
coverage = fuzzy_test.evaluate(y_test,predictions,method = 'coverage')
#measures adjusted average width of intervals
pinaw = fuzzy_test.evaluate(y_test,predictions,method = 'interval_width_average) 
```
### WindowGenerator
We also offer a WindowGenerator object that assists in the creation of time windowing
of multiple timeseries for feature construction for LSTM models.

```python
from src.models.features.timeseriesprocessing import WindowGenerator

data = pd.read_csv("x_train.csv")  # data importing

look_back = 10 #number of previous timesteps to use as features

shift = 1 #number of timesteps ahead to forecast

target_length = 1 #number of timesteps being forecasted

target_columns = [x for x in data.columns if 'target' in x]

input_columns = [x for x in data.columns if x not in target_columns]

window_model = WindowGenerator(data,look_back,shift,target_length,input_columns,target_columns) 
```

For more information on `WindowGenerator` object and methods please see docs.

### Rolling_Window

## Data
