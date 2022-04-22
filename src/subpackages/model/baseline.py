import numpy as np

class interval_baseline:
  def __init__(self,alpha):
    self.alpha = alpha
  def fit(self,X_train,y_train):
    upper = np.quantile(y_train.flatten(),(1-self.alpha/2))
    lower = np.quantile(y_train.flatten(),self.alpha/2)
    self.upper_prediction = upper
    self.lower_prediction = lower
  def predict(self,X_test):
    result = np.empty((len(X_test),2))
    for i in range(len(X_test)):
      result[i] = np.array([self.lower_prediction,self.upper_prediction])
    return result
