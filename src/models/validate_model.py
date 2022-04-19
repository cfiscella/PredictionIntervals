import tensorflow as tf

import pandas as pd
import numpy as np

# Keras
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, RepeatVector, Input, BatchNormalization, Bidirectional,multiply, concatenate, Flatten, Activation, dot,Layer

from src.features.ts_process.validation import RollingValidation
from src.features.ts_process.standardize import WindowMinMaxScaler

from src.models.model.baseline import interval_baseline
from src.models.model.fuzzy_interval import fuzzy_interval

data = pd.read_csv('/content/drive/Shareddrives/Prediction Intervals/Data/CompleteDataSet/processed_dataset.csv',parse_dates = True,index_col = "DATE")

###defining sub-models
def seq2seq(time_steps,n_hidden,n_features):
  input = Input(shape = (time_steps,n_features))

  encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(
    n_hidden, activation='elu',
    return_state=True, return_sequences=True,dropout = .2, recurrent_dropout = .2)(input)
  ###does more wild stuff(cross etc. with lower momentum)
  #encoder_last_h = BatchNormalization(momentum=0.2)(encoder_last_h)
  #encoder_last_c = BatchNormalization(momentum=0.2)(encoder_last_c)
  decoder_input = RepeatVector(1)(encoder_last_h)
  
  decoder_stack_h = LSTM(n_hidden, activation='elu',
  return_state=False, return_sequences=True,dropout = .2,recurrent_dropout = .2)(
  decoder_input, initial_state=[encoder_last_h, encoder_last_c])

  attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
  attention = Activation('softmax')(attention)

  
  context = dot([attention, encoder_stack_h], axes=[2,1])
  #context = BatchNormalization(momentum=0.2)(context)

  decoder_combined_context = concatenate([context, decoder_stack_h])

  decoder_combined_context = Flatten()(decoder_combined_context)

  #decoder_combined_context = BatchNormalization(momentum = .2)(decoder_combined_context)

  #dense_1 = Dense(64)(decoder_combined_context)
  out = Dense(2)(decoder_combined_context)
  model = Model(inputs = input,outputs = out)


  return model

def bidirectional_regression(time_steps=0,n_hidden=0,n_features=0):
  input = Input(shape = (time_steps,n_features))
  lstm1 = Bidirectional(LSTM(units = n_hidden,return_sequences = False,dropout = .2))(input)
  dense1 = Dense(128)(lstm1)
  out = Dense(1)(dense1)
  model = Model(inputs = input,outputs = out)
  return model

###sub-model objects
class EarlyStopByAbsVal(keras.callbacks.Callback):
    def __init__(self, min_delta = 0, verbose = 0):
        super(keras.callbacks.Callback, self).__init__()
        self.min_delta = min_delta
        self.verbose = verbose
        self.last_loss = 0


    def on_epoch_end(self, epoch, logs={}):

        epoch_loss = logs['loss']
        if abs(epoch_loss-self.last_loss) > self.min_delta:
            if self.verbose >0:
                print("Epoch %05d: early stopping Threshold" % epoch)
                self.model.stop_training = True
        else:
            self.last_loss = epoch_loss

interval_opt = tf.keras.optimizers.Adam(lr = .001,clipnorm = 1)
interval_callback = EarlyStopByAbsVal(min_delta = 10**(-7))

###Define Global Parameters

start_date = "2000-01-01"
end_date = "2021-06-30"



#get a list of sector tickers
'''
communication services: XLC
consumer discretionary: XLY
consumer staples: XLP
Energy: XLE
Financials: XLF
Healthcare: XLV
Industrials: XLI
Information Technology: XLK
Materials: XLB
Real Estate: IYR
Utilities: XLU
'''



#interval_mod = bidirectional(40,256,120)


fixed_parameter_dict_model = {"regressor_model":bidirectional_regression,
                        


"regressor_window":10,

"regress_model_dict" : {"time_steps":10,"n_hidden":512,"n_features":119},

"regress_compile_dict": {"optimizer":'adam', "loss":'mean_squared_error',"metrics":["mean_absolute_percentage_error"]},



"regress_fit_dict":{"verbose":True, "epochs":125},

"clusters":3,
"cluster_alpha":.045,
"vol_scale":.8,

"interval_model":seq2seq,

"interval_model_dict": {"time_steps":40,"n_hidden":256,"n_features":120},

"interval_window":40,

"interval_fit_dict":{"epochs":150,"batch_size":32,"verbose":True,'callbacks':[interval_callback]},

"interval_compile_dict":{"loss":"mean_squared_error","optimizer":interval_opt},


}

fixed_parameter_dict_baseline = {"alpha":.05}

standardizer = WindowMinMaxScaler()

###experimental global params
length = len(data)
window_length = (1/8)*length
shift = (1/12)*length
train_test_val_split = (1/6,2/3,1/6)

###main
model_result_dict = {}
baseline_result_dict = {}
truth_dict = {}
input_columns = [x for x in data if 'target' not in x]
target_etfs = ['XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLB', 'XLU']

for etf in target_etfs:
  window_model = WindowGenerator(data,10,1,1,input_columns,[etf+"_target"])
  window_baseline = WindowGenerator(data,1,1,1,input_columns,[etf + "_target"])
  e_model = Rolling_Validation(window_model.inputs,window_model.targets,window_length,shift,fuzzy_interval,fixed_parameter_dict_model,standardization =standardizer)
  e_baseline =Rolling_Validation(window_baseline.inputs,window_baseline.targets,window_length,shift,interval_baseline,fixed_parameter_dict_baseline)
  model_result_dict[etf]=e_model.experiment_dict 
  baseline_result_dict[etf] = e_baseline.experiment_dict
  truth_dict[etf] = e_baseline.true_dict
  
###need to break here to save stuff that can then be used to visualize

test_index = pd.MultiIndex.from_product([[],[],[]], names=["etf","experiment", "bound"])
truth_index = pd.MultiIndex.from_product([[],[]], names=["etf","experiment"])

model_result = pd.DataFrame(columns = test_index)
baseline_result = pd.DataFrame(columns = test_index)
truth_result = pd.DataFrame(columns = truth_index)

for etf in model_result_dict.keys():
  for experiment in model_result_dict[etf].keys():
    sub_index = pd.MultiIndex.from_product([[etf],[experiment],['lower','upper']],names = ['etf','experiment','bound'])
    truth_index = pd.MultiIndex.from_product([[etf],[experiment]],names = ['etf','experiment'])
    
    model_lower = pd.Series(model_result_dict[etf][experiment][:,0],name = 'lower')
    model_upper = pd.Series(model_result_dict[etf][experiment][:,1],name = 'upper')
    model_sub_result = pd.concat([model_lower,model_upper],axis =1)
    model_sub_result.columns = sub_index
    model_result = pd.concat([model_result,model_sub_result],axis =1)

    baseline_lower = pd.Series(baseline_result_dict[etf][experiment][:,0],name = 'lower')
    baseline_upper = pd.Series(baseline_result_dict[etf][experiment][:,1],name = 'upper')
    baseline_sub_result = pd.concat([baseline_lower,baseline_upper],axis =1)
    baseline_sub_result.columns = sub_index
    baseline_result = pd.concat([baseline_result,baseline_sub_result],axis =1)

    true = pd.DataFrame(truth_dict[etf][experiment].flatten(),columns = truth_index)
    truth_result = pd.concat([truth_result,true],axis =1)
    
    
    
model_true = truth_result[-len(model_result):].reset_index(drop = True)


etfs = list(set(model_true.columns.get_level_values(0)))
experiments = list(set(model_true.columns.get_level_values(1)))
mods = ['lstm','baseline']

result_index = pd.MultiIndex.from_product([etfs,experiments,mods], names=["etf","experiment", "model"])
result = pd.DataFrame(index = result_index,columns = ["Coverage","PINAW"])
for etf in etfs:
  for experiment in experiments:

    model_cov = coverage(model_true[etf][experiment].to_numpy(),model_result[etf][experiment].to_numpy())
    baseline_cov = coverage(truth_result[etf][experiment].to_numpy(),baseline_result[etf][experiment].to_numpy())
    result.loc[(etf,experiment,'lstm'),"Coverage"] = model_cov
    result.loc[(etf,experiment,'baseline'),"Coverage"] = baseline_cov
    

    model_width = interval_width_average(model_true[etf][experiment].to_numpy(),model_result[etf][experiment].to_numpy())
    baseline_width = interval_width_average(truth_result[etf][experiment].to_numpy(),baseline_result[etf][experiment].to_numpy())
    result.loc[(etf,experiment,'lstm'),"PINAW"] = model_width
    result.loc[(etf,experiment,'baseline'),"PINAW"] = baseline_width
    
result["Efficiency"] = result["Coverage"]/result["PINAW"]

result_to_save = result["Efficiency"]

###need to save

result_to_save.to_csv(***path to save results)
