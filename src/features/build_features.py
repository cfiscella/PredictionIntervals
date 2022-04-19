#figure out where to import interim and where to save preprocessed
import numpy as np
import pandas as pd

interim = pd.read_csv('/data/interim/interim_dataset.csv',parse_dates = True,index_col = "DATE")

etfs = ['XLY',  'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLK', 'XLB', 'IYR', 'XLU']

for etf in etfs:
  interim[etf+"_target"] = np.log(interim["('Close', '"+etf+"')"]).shift(-1)-np.log(interim["('Close', '"+etf+"')"])


preprocessed =interim.iloc[:-1,:]

preprocessed.to_csv('data/processed/processed_dataset.csv')
