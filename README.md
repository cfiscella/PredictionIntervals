# PredictionIntervals

Stock Price Prediction Intervals w/ LSTM and Fuzzy Clustering

## Installation

The simplest way to recreate the project is to clone the repository, set it as your current working directory and run the make command:

```bash
$ git clone https://github.com/cfiscella/PredictionIntervals/
$ cd ./PredictionIntervals
$ make
```
Local recreations of model results will be saved in your current working directory.

The project also supports '''bash $ tox''' which will report successful if median perofrmance accross all ETFs across all rolling windows is greater than median performance of baseline model.

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

We finally provide `Rolling_Validation` object to streamline model validation for time series.
Time series provide particular challenges to the validation process, including suceptibility to
lookahead bias and information leakage.  One way of addressing this is through a rolling window 
validation framework implemented here. Details about this process can be found [here](https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4)


```python
from src.features.timeseriesprocessing import Rolling_Validation
from src.models.clusterintervals import fuzzy_cluster

inputs = pd.read_csv("inputs.csv")  # data importing
targets = pd.read_csv('targets.csv')

window_length = (1/8)*len(inputs) #length of data in given window

shift = (1/12)*len(inputs) #length of 'roll' forward for each window

model = fuzzy_cluster #class that can be instantiated with a parameter dictionary

fixed_parameter_dict = { #parameters to instantiate model class that will stay fixed during rolling validation process
"regressor_model":[Bidirectional(LSTM(units=256, return_sequences=False,dropout = .2)),Dense(units=1)],                      
"regressor_window":10, "regress_compile_dict": {"optimizer":'adam', "loss":'mean_squared_error',
"metrics":["mean_absolute_percentage_error"]},
"regress_model_dict":{"verbose":True, "epochs":50},"clusters":3, "cluster_alpha":.05,
"interval_model":[Bidirectional(LSTM(units=256 return_sequences=False,dropout = .2)),Dense(units=2)],
"interval_window":40,"interval_model_dict":{"epochs":100,"batch_size":32,"verbose":True},
"interval_compile_dict":{"loss":"mean_squared_error","optimizer":"adam"},
}

e_model = Rolling_Validation(window_model.inputs,window_model.targets,window_length,
shift,model,fixed_parameter_dict)
```

For more information on the `Rolling_Validation` object please see docs.
## Data

The dataset provided is combpiled from the Yahoo Finance API, gold.org, U.S Energy and Information Administration, and fred.stlouisfed.org. Yahoo Finance data includes daily High, Low, Volume and Close.  The processed dataset includes dates from 12/30/2003 to 6/30/2021. Targets are daily returns for sector ETFs including: XLY, XLP, XLE, XLF, XLV, XLI, XLK, XLB, XLU. Features can be subgrouped into US Fixed Income, Exchange Rates, Global Indecies, US Sector ETFs, and Comodities. A summary of features and targets can be found below. Processed features were outer joined and forward filled.
