###need to address how raw_data is imported and where interim data is saved
import numpy as np
import pandas as pd

import sys
#import os

path = sys.argv[1]

raw_data = pd.read_csv(path,parse_dates = True,index_col = "DATE",)['2003-12-30':'2021-06-30'].drop("1Y_yield",axis=1)
data = raw_data.replace(".",np.nan).fillna(method = 'ffill')

drop_XLC = [x for x in data.columns if "XLC" in x]

data.drop(drop_XLC,axis =1, inplace = True)

data = data.astype("float64")

#cwd = os.getcwd()

#saved_path = 'interim_dataset1.csv'

data.to_csv('interim_dataset1.csv')
