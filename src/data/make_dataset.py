import numpy as np
import pandas as pd

raw_data = pd.read_csv('/PredictionIntervals/data/raw/raw_full_dataset.csv',parse_dates = True,index_col = "DATE",)['2003-12-30':'2021-06-30'].drop("1Y_yield",axis=1)
data = raw_data.replace(".",np.nan).fillna(method = 'ffill')

drop_XLC = [x for x in data.columns if "XLC" in x]

data.drop(drop_XLC,axis =1, inplace = True)

data = data.astype("float64")

pd.to_csv('/PredictionIntervals/data/interim/interim_dataset.csv')
