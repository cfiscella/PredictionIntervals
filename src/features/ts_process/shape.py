import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler

def shape_data(df,window):
    '''
    Returns Numpy array of dimension (len(df),window,len(df.columns)). Creates features from lags of time series to be used for modeling.

            Parameters:
                    df (DataFrame): A Pandas DataFrame
                    window (int): Length of lookback window

            Returns:
                    ds (ndarray): Numpy array of dimension (len(df),window,len(df.columns)).
    '''

    win_t = window
    ds = []

    for ind in range(win_t, len(df)+1):

      ds.append(df.iloc[ind-win_t:ind,:].to_numpy())
    result = np.array(ds)

    return result

def dissolve(shaped_data,window):
    '''
    Inverse of shape_data. Returns pre-shaped, unwindowed data as numpy array.
            Parameters:
                    shaped_data (list like): Shaped data of dimension (len(data),window,features)
                    window (int): Length of lookback window

            Returns:
                    ds (nd.array): Numpy array of dimension (len(df),features).
    '''

    ds = []
    for i in range(window):
      ds.append(shaped_data[0][i])
    for j in range(1,len(shaped_data)):
      ds.append(shaped_data[j][-1])
    result = np.array(ds)

    return result

class WindowGenerator:
    """
    A class to format time series for modeling.

    ...

    Attributes
    ----------
    data : str
        first name of the person
    input_list : str
        family name of the person
    target_list : int
        age of the person
    target_columns_indices : list
        list of indicies of target columns to predict
    column_indices : list
        list of indicies of features
    input_length : int
        length of look-back window
    target_length : int
        length of consecutive target days predicted
    shift : int
        number of samples between when look-back period ends and target begins
    total_window_size : int
        input_length+shift
    input_slice : slice
        slice object to index input data
    input_indices : ndarray
        array of input indices
    target_start : int
        index of where target data begins
    targets_slice : slice
        slice object to index target data
    target_indices : ndarray
        array of target indices
    inputs : ndarray
        split and shaped inputs from data
    targets : ndarray
        split and shaped targets from data
    train : (ndarray,ndarray)
        (input,target) tuple for model training
    val : (ndarray,ndarray)
        (input,target) tuple for model validation
    test : (ndarray,ndarray)
        (input,target) tuple for model testing

    Methods
    -------
    data_split(self, split = [.6,.2,.2],val_behind = True,standardization = "None"):
        Splits inputs and targets into train, test and val sets.
    """
    def __init__(self,data,input_length,shift,target_length,input_list,target_list):
    # Store the raw data.
        """
        Constructs the minimum necessary attributes for the WindoGenerator object.

        Parameters
        ----------
            data : DataFrame
                raw time series DataFrame including inputs and targets used to construct inputs and targets for modeling
            input_length : int
                length of look-back window
            shift : int
                number of time steps between last input value and corresponding first target
            target_length : int
                length of target array per sample
            input_list : list
                list of columns in data to be used as inputs
            target_list : list
                list of columns in data to be used as targets
      """
        self.data = data
        self._train = None
        self._val = None
        self._test = None
        self.input_list = input_list
    # Work out the label column indices.
        self.target_list = target_list
        if target_list is not None:
          self.target_columns_indices = {name: i for i, name in
                                    enumerate(target_list)}
        self.column_indices = {name: i for i, name in
                           enumerate(data.columns)}

    # Work out the window parameters.
        self.input_length = input_length
        self.target_length = target_length
        self.shift = shift

        self.total_window_size = input_length + shift

        self.input_slice = slice(0, input_length)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.target_start = self.total_window_size - self.target_length
        self.targets_slice = slice(self.target_start, None)
        self.target_indices = np.arange(self.total_window_size)[self.targets_slice]

    ###initialize targets and inputs which will be arrays of arrays
        self._targets =None
        self._inputs = None
  
    def split_window(self, features):
    ###private method used to generate input and target arrays
      inputs = shape_data(features[self.input_list],self.input_length)
      labels = shape_data(features[self.target_list][self.input_length+self.shift:],self.target_length)

      ###trim, make lengths of each the same s
      if len(inputs)>len(labels):
        inputs = inputs[:len(labels)]
      else:
        labels = labels[:len(inputs)]

      self._inputs = inputs
      self._targets = labels


      return inputs, labels

      WindowGenerator.split_window = split_window

    @property
    def inputs(self):
      if type(self._inputs) == type(None):
        return self.split_window(self.data)[0]
      else:
        return self._inputs
    @inputs.setter
    def inputs(self,data):
      self._inputs = self.split_window(data)[0]

    @property
    def targets(self):
      if type(self._targets) == type(None):
        return self.split_window(self.data)[1]
      else:
        return self._targets

        WindowGenerator.inputs = inputs
        WindowGenerator.targets = targets

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, indecies):

      self._train = self.inputs[indecies[0]:indecies[1],:,:],self.targets[indecies[0]:indecies[1],:,:]
    @property
    def val(self):
      return self._val

    @val.setter
    def val(self, indecies):

      self._val = self.inputs[indecies[0]:indecies[1],:,:],self.targets[indecies[0]:indecies[1],:,:]
    @property
    def test(self):
      return self._test
      WindowGenerator.train = train
      WindowGenerator.val = val
      WindowGenerator.test = test
    @test.setter
    def test(self, indecies):

      self._test = self.inputs[indecies[0]:indecies[1],:,:],self.targets[indecies[0]:indecies[1],:,:]
  


    def data_split(self, split = [.6,.2,.2],val_behind = True,standardization = "None"):
        """
        Splits data into train, test and (optionally) validation sets and assigns them as attributes.

        If the argument 'val_behind' is True, then validation attribute is assigned.

        If the argument 'standardization' is specified, data will standardized by the specified method and transofrmed.

        Parameters
        ----------
        split : list
            Split between training set, testing set and validation set
        val_behind : bool
            Indicates whether validation set will be behind or infront of training set
        standardization : string
            Indicates standardization method used

        Returns
        -------
        None 
        -------
        """
        val_length = int(len(self.data)*split[2])
        train_length = int(len(self.data)*split[0])
        test_length = int(len(self.data)*split[1])
        if val_behind:
          val_slice = [0,val_length]
          train_slice = [val_length,val_length+train_length]
          test_slice = [val_length+train_length,val_length+train_length+test_length]
        else:
          train_slice = [0,train_length]
          val_slice = [train_length,val_length+train_length]
          test_slice = [val_length+train_length,val_length+train_length+test_length]

        if standardization == "None":

          self.val = val_slice
          self.train = train_slice
          self.test = test_slice

        elif standardization == "MinMax":
          minmax = MinMaxScaler()
          minmax.fit(self.data[self.input_list].iloc[train_slice[0]:train_slice[1]])
          target = self.data["target"].reset_index()

          self.scaler = minmax         
          scaled_data = pd.concat([pd.DataFrame(self.scaler.transform(self.data[self.input_list]),
                                            columns = self.input_list),target],
                              axis = 1,).iloc[:-1,:]

          self.inputs = scaled_data
          self.val = val_slice
          self.train = train_slice
          self.test = test_slice 
        
        elif standardization == "Standard":
          standard = StandardScaler()
          standard.fit(self.data[self.input_list].iloc[train_slice[0]:train_slice[1]])
          target = self.data["target"].reset_index()

          self.scaler = standard         
          scaled_data = pd.concat([pd.DataFrame(self.scaler.transform(self.data[self.input_list]),
                                            columns = self.input_list),target],
                              axis = 1,).iloc[:-1,:]

          self.inputs = scaled_data
          self.val = val_slice
          self.train = train_slice
          self.test = test_slice
        return None
        WindowGenerator.data_split = data_split 
