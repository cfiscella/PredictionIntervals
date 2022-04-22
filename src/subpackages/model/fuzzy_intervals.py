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
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, Bidirectional,multiply, concatenate, Flatten, Activation, dot,Layer

from scipy.stats import zscore

from fcmeans import FCM
from arch import arch_model

from src.features.ts_process.shape import shape_data

###helper functions for computing prediction intervals for each time step:
###flow is dict ->cluster_PI (we find each interval for each cluster)->
#instance_interval (we use cluster_dict to then find weighted intervals for each time step(this is the bulk of the work))->
#prediction_interval (reformats instance_interval)
###ideally cluster_dic should be in form of dictionary of dictionaries {cluster#:{residualindex: (residual,membershipgrade)}} (we don't even need the actual residual for right now)

###dictionary constructor for prediction interval computation
def dict_construct(df):
  return {cluster: {ind:(df["Residual"][ind],df[str(cluster)][ind]) for ind in range(len(df))}  for cluster in range(len(df.columns)-2) }

###transforms clusterdictionary specified above into new format where new dictionary has 
###instance index as keys, and list of tuples as values. list of tuples is of form [(cluster,membershipgrade)]
def dic_transform(cluster_dic):
  ###take cluster list and get all the unique indicies of the various residuals
  list_of_indicies = list(set([item for sublist in [x.keys() for x in cluster_dic.values()] for item in sublist]))
  ###instantiate the new dictionary with keys = indicies we got above and values are going to be a list
  new_dic = {ind:[] for ind in list_of_indicies}
  ###for eac cluster in the clusterdic we go through all the different residuals in each cluster, and store their weights in the new dictionary that'll keep track of every
  ###cluster the residual belongs to, along with the associated weight for that cluter
  for cluster in cluster_dic.keys():
    for residindex in cluster_dic[cluster].keys():
      new_dic[residindex].append((cluster,cluster_dic[cluster][residindex][1]))

  return new_dic

###returns dictionary where keys are cluster numbers and values are a tuple with (lowerinterval, upperinterval)
def cluster_PI(cluster_dic,alpha):
  result_dic = {}
  for cluster in cluster_dic.keys():
    ###we start by sorting the clusters by ascending residuals
    sorted_instances =  sorted(cluster_dic[cluster].items(),key = lambda x: x[1][0])
    ###we create a list of the membership weights where the jth element of the membership list corresponds to the jth element of the sorted residual list
    membershiplist = [x[1][1]for x in sorted_instances]
    ###we use this sum of weights to find the alpha = .05 interval
    low_quantile = (alpha/2)*sum(membershiplist)
    high_quantile = (1-alpha/2)*sum(membershiplist)
    
    memsum = 0
    upperindex = 0
    lowerindex = 0
    ###procedure for upper/lower prediciton intervals for each cluster is as follows:
    ###1.loop through classweights sorted according to their associated residuals where residuals are increasing
    ###for each classweight we check to see if it's less than (alpha/2)*sum of all class weights
    ###if yes, we add that classweight to a cumulative sum and keep going
    ###if no (running sum of classweights is bigger than the lowquantile) break loop and select associated residual with the previous classweight as our lower bound
    for instance in sorted_instances:
      memsum+=instance[1][1]
      if memsum<low_quantile:
        lowerindex+=1
      if memsum<high_quantile:
        upperindex +=1

    result_dic[cluster] = (sorted_instances[lowerindex][1][0],sorted_instances[upperindex][1][0])

  return result_dic

###once we have the upper and lower bounds for each cluster, we go through each of the individual residuals, look at the clusters they belong to,
###and compute weighted (by classweight) sum for upper and lower bounds
###returns dictionary where indexed residuals are keys and values are tuples(lower interval, upper interval)
def instance_interval(cluster_dic,alpha):
  result = {}
  residualindexdic = dic_transform(cluster_dic)
  cluster_intervals = cluster_PI(cluster_dic,alpha)

  for residual in residualindexdic.keys():
    reslower =0
    resupper =0
    for cluster in residualindexdic[residual]:
      reslower +=cluster[1]*cluster_intervals[cluster[0]][0]
      resupper +=cluster[1]*cluster_intervals[cluster[0]][1]
    result[residual] = (reslower,resupper)
  return result

###after this, we have a list of indicies and associated intervals, so it'd be nice to have a series of residuals where index is residual index and value is the actual residual
###then we take x = pd.Series(instance_interval), y = residual_series and our target is something like: target = 


def prediction_interval(instance_interval_dict,ret):

  x = pd.Series(instance_interval_dict)
  upper = x.apply(lambda x:x[1])
  lower = x.apply(lambda x:x[0])
  length = upper-lower
  y = np.reshape(ret,len(ret))

  return np.array([list((y[i]+(lower[i]),(upper[i])+y[i])) for i in range(len(y))])


def melt(a,b,melt_factor):
  #melt a into b
  spread = b-a
  melted_spread = melt_factor*spread
  return a+melted_spread

def z_score_inverse(zscores,original_mean,original_std):
  return original_std*zscores+original_mean


###model metrics
def coverage(y_true,y_pred):
  ###percentage of prices that fall within predicted interval

  ##boolean list of whether prices in test set fall within their predicted intervals

  in_interval_list = [y_true[i] >= y_pred[i,0] and y_true[i] <= y_pred[i,1] for i in range(len(y_true))]
  ###returns proportion of samples that fell within their predicted intervals
  return np.sum(in_interval_list)/len(in_interval_list)



###adjusts mean_prediction_interval as a precentage of true values so different price series can be compared
def interval_width(y_true,y_pred):
  return y_pred[:,1]-y_pred[:,0]

def interval_width_average(y_true,y_pred):
  return (1/np.ptp(y_true))*np.mean(interval_width(y_true,y_pred))

def interval_bias(y_true,y_pred):
  ###function to know whether true is closer to upper or lower
  midpoint = (y_pred[:,0]+y_pred[:,1])/2

  return y_true-midpoint

###main model objects
class fuzzy_interval:
    """
    A class to implement fuzzy interval method of generating prediction intervals.

    ...

    Attributes
    ----------
    regressor : keras.model
        model used to generate point predictions on training set
    regressor_window : int
        lookback period used in regressor model
    regress_compile_dict : dict
        dictionary of arguments used as input for regressor.compile()
    regress_model_dict : dict
        dictionary of arguments used as input for regressor.fit()
    clusters : int
        number of clusters used for fuzzy-c clustering of residuals
    vol_scale : float
        degree of shifting interval_lengths to match training data volatility
    residuals : ndarray
        stores residuals of point prediction model on training set
    mse : float
        mean squared error of point prediction model on training set
    cluster_df : DataFrame
        DataFrame storing cluster membership weights of each time step within each cluster
    alpha : float
        hyperparameter affecting width of intervals for each cluster
    intervals : ndarray
        ndarray of computed intervals for each time step
    interval_model : keras model
        model fit on intervals
    interval_window : int
        lookback period used in interval_model
    interval_model_dict : dict
        dictionary of arguments used as input in interval_model
    interval_compile_dict : dict
        dictionary of arguments used as input in interval_model.compile()
    interval_fit_dict : dict
        dictionary of arguments used as input in interval_model.fit()

    Methods
    -------
    fit(X_train,y_train):
        Fits complete model to time series.
    predict(X_test):
        Returns out of sample predictions for given time series
    evaluate(y_true,y_predict,method = 'coverage'):
        Evaluates model perofrmance given sequence of true values and prediction intervals and evaluation metric.
    """
    def __init__(self,regressor_model,regressor_window,regress_model_dict,regress_compile_dict,regress_fit_dict,clusters,
                   interval_model,interval_window,interval_model_dict,interval_compile_dict,interval_fit_dict,cluster_alpha=.05,vol_scale = .5):
          """
          Constructs the minimum necessary attributes for the RollingWindow object.

          Parameters
          ----------
              regressor_model : keras.model
                    model used to generate point predictions on training set
              regressor_window : int
                    lookback period used in regressor model
              regress_compile_dict : dict
                    dictionary of arguments used as input for regressor.compile()
              regress_model_dict : dict
                    dictionary of arguments used as input for regressor.fit()
              clusters : int
                    number of clusters used for fuzzy-c clustering of residuals
              interval_model : keras model
                    model fit on intervals
              interval_window : int
                    lookback period used in interval_model
              interval_compile_dict : dict
                    dictionary of arguments used as input in interval_model.compile()
              interval_fit_dict : dict
                    dictionary of arguments used as input in interval_model.fit()
              cluster_alpha : float
                    hyperparameter affecting width of intervals for each cluster
              vol_scale : float
                    hyperparameter tuning similarity in distribution of interval lengths to volatility of training sample
      """
          self.regress_model_dict = regress_model_dict
          self.regressor = regressor_model(**self.regress_model_dict)
          self.regressor_window = regressor_window
          self.regress_compile_dict = regress_compile_dict
          self.regress_fit_dict = regress_fit_dict

          self.clusters = clusters
          self.residuals = None
          self.mse = None
          self.cluster_df = None
          self.alpha = cluster_alpha
          self.intervals = None
          self.interval_model_dict = interval_model_dict         
          self.interval_model = interval_model(**self.interval_model_dict)
          self.interval_fit_dict = interval_fit_dict
          self.interval_window = interval_window
          self.interval_compile_dict = interval_compile_dict
          self.vol_scale = vol_scale

####regression methods
    def regression_fit(self,X_train,y_train):

        y_train = y_train.reshape(-1,1)

        self.regressor.compile(**self.regress_compile_dict)
        self.regressor.fit(X_train,y_train,**self.regress_fit_dict)
        return None

    def regression_predict(self,X_test):
        return self.regressor.predict(X_test)
      
    def regression_evaluate(self,X_test,y_test):
        mse = self.regressor.evaluate(X_test,y_test)
        self.mse = mse
        return self.mse
      
    def regression_residuals(self,y_true,y_predict,save_residuals = False):
        residuals = y_true.reshape(-1,1).reshape(len(y_true))-np.reshape(y_predict,len(y_predict)).reshape(len(y_true))

        if save_residuals == True:
          self.residuals = pd.Series(residuals)
        return residuals
####cluster methods   
    def cluster_fit(self):
        fcm = FCM(n_clusters = self.clusters)
        fcm.fit(pd.DataFrame(self.residuals).values)
        result_df = pd.DataFrame()
        result_df["Residual"] = self.residuals
        for i in range(self.clusters):
          result_df[str(i)] = fcm.u[:,i]
        fcm_labels = fcm.predict(pd.DataFrame(self.residuals).values)

        result_df["clusters"] = fcm_labels
        self.cluster_df = result_df
        return None
###interval methods
    def interval_generate(self,X_train):
        cluster_dict = dict_construct(self.cluster_df)
        self.raw_intervals = instance_interval(cluster_dict,self.alpha)
        raw_ints = pd.Series(self.raw_intervals)
        raw_upper = raw_ints.apply(lambda x:x[1])
        raw_lower = raw_ints.apply(lambda x:x[0])
        width = raw_upper-raw_lower
        width_mean = np.mean(width)
        width_std = np.std(width)
        width_z = zscore(width)

        adjusted_width = z_score_inverse(melt(width_z,zscore(self.vol),self.vol_scale),width_mean,width_std)
        width_difference = (adjusted_width-width)/2
        adjusted_upper = raw_upper+width_difference
        adjusted_lower = raw_lower-width_difference
        self.adjusted_intervals = {i:(adjusted_lower[i],adjusted_upper[i]) for i in range(len(adjusted_upper))}
        self.intervals = prediction_interval(self.adjusted_intervals,X_train) 
        return None

    def interval_fit(self,X_train):
        original_observations = self.y_data


        regression_predictions = self.regression_predict(X_train).reshape(-1,1)


        raw_X_final_train = pd.concat([pd.DataFrame(dissolve(X_train,self.regressor_window)[self.regressor_window-1:]),
                                     pd.DataFrame(regression_predictions)],axis = 1)
        X_train_final = shape_data(raw_X_final_train,self.interval_window)

        ###remember X_train is already shaped for LsTM i.e. made in 3 diensions w lookback window
        #########################shape data in here, might result in dependency issue


        interval_y = self.intervals[len(self.intervals)-len(X_train_final):,:]
        self.interval_model.compile(**self.interval_compile_dict)

        self.interval_model.fit(X_train_final,interval_y,**self.interval_fit_dict)
        predictions = self.interval_model.predict(X_train_final)

        return None

    def interval_predict(self,X_test):


        return self.interval_model.predict(X_test)
      
    def lower_interval(self):
        return self.test_class.intervals[:,0]      
        
    def upper_interval(self):
        return self.test_class.intervals[:,1]
      
###comprehensive methods
    def fit(self,X_train,y_train):
          """
          Fits fuzzy_interval model on training data.

          Parameters
          ----------
          X_train : DataFrame
              Training inputs
          y_train : series
              True training targets

          Returns
          -------
          None
          """
          self.y_data = y_train
          garch = arch_model(self.y_data,vol = 'GARCH',p=1,q=1)
          garch_fit = garch.fit()
          vol_est = pd.DataFrame(garch_fit.conditional_volatility).values.flatten()
          self.vol = vol_est
          print("Fitting Regression")
          self.regression_fit(X_train, y_train)
          predictions = self.regression_predict(X_train)
          print("Regression Fit Completed")

        ###need to add predictions as feature for part 3 could do it here or could do it in interval_generate
          self.regression_residuals(y_train,self.regression_predict(X_train).reshape(len(X_train)),save_residuals=True)
          self.cluster_fit()
          self.interval_generate(predictions)

          print("Fitting Intervals")
          self.interval_fit(X_train)
          print("Interval Fit Completed")
          return None
        ###need to adjust here,
    def predict(self,X_test):
          """
          Returns a series of next day prediction intervals for a given input time series.

          Parameters
          ----------
          X_test : DataFrame
              Input DataFrame

          Returns
          -------
          prediction_intervals : ndarray
              Array of form [[lower,upper]] where lower is lower bound prediction ad upper is upper bound prediction
      """
        ###step 1 is to generate regression predictions as a feature, x_test already formatted for regression prediction
          regression_predictions = self.regression_predict(X_test).reshape(-1,1)
        

          raw_X_final_test = pd.concat([pd.DataFrame(dissolve(X_test,self.regressor_window)[self.regressor_window-1:]),
                                     pd.DataFrame(regression_predictions)],axis = 1)

          X_test_final = shape_data(raw_X_final_test,self.interval_window)
        ###going to write a cute little function to combine regression predictions with reshaped X_test and concatenate

          return self.interval_predict(X_test_final)
    def evaluate(self,y_true,y_predict, method = "coverage"):
      """
      Evaluates prediction interval model performance.

      If method = 'coverage', the percent of true values covered will be returned.

      If method = 'interval_width_average', the adjusted average interval width will be returned.

      If method = 'interval_bias', the bias (off-centeredness) of the intervals will be returned

      Parameters
      ----------
      y_true : ndarray
          Array of true values used to evaluate prediction intervals
      y_predict : ndarray
          Array of prediction intervals to be evaluated
      method : string
          Indicates evaluation metric to be returned
      Returns
      -------
      metric : float
          Indicated evaluation metric
      """
      if method == "coverage":
          return coverage(y_true,y_predict)
      if method == "interval_width_average":
          return interval_width_average(y_true,y_predict)
      if method == "interval_bias":
          return interval_bias(y_true,y_predict)
