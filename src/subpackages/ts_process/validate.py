###helper function for combining validation hyperparameters and fixed parameters
import numpy as np

def combine_dicts(dic1,dic2):
  ###start by combining dict elements of each
  res_repeat = {} 
  repeated_keys = list(set(dic1.keys()).intersection(set(dic2.keys())))
  non_repeat_1 = {key:dic1[key] for key in dic1.keys() if key not in repeated_keys}
  non_repeat_2 = {key:dic2[key] for key in dic2.keys() if key not in repeated_keys}
  non_repeated = {**non_repeat_1,**non_repeat_2}
  for key in repeated_keys:
    merged_dict = {**dic1[key],**dic2[key]}
    res_repeat[key] = merged_dict
  res_final = {**res_repeat,**non_repeated}
  return res_final

class Rolling_Validation:
    """
    A class to implement rolling window cross validation for time series.

    ...

    Attributes
    ----------
    input : ndarray
        ndarray to be split into windows and used as inputs to model
    target : ndarray
        ndarray to be split into windows and used as targets
    window_length : int
        length of each rolling window
    shift : int
        number of time steps rolling window is shifted up before retraining/retesting
    model : class
        model class that has train and fit methods that will be used for each window
    fixed_parameter_dict : dict
        dictionary of model parameters that will not be tuned on a validation set
    train_test_split : float
        float between 0 and 1 determining split of each window between training and tesitng data
    hyperparameter_tuning : bool
        determines whether or not to tune hyperparameters on validation set before out of sample testing
    variable_model_parameter_grid : dict
        dictionary of model_parameter_name:tuning_list key:value pairs to tune during hyperparameter tuning
    train_test_val_split : tuple
        if 'hyperparameter_tuning' = True, rolling_window will be split according to (train,test,val) fractions
    standardization : string
        string indicating standardization method when fitting and testingmodels
    verbose : bool
        determines whether to print model summaries after each window
    experiment_dict : dict
        dictionary storing out of sample predictions for each rolling window
    true_dict : dict
        dictionary storing true out of sample values for each rolling window

    Methods
    -------
    conduct_experiment():
        Returns a dictionary of window_number:out_of_sample_prediciton_array key:value pairs.
    """
    def __init__(self,input,target,window_length,shift,
                 model,fixed_model_parameters,train_test_split = .8,hyperparameter_tuning = False,variable_model_parameter_grid=None,
                 train_test_val_split = (.6,.2,.2),standardization = None,verbose = True):
        """
        Constructs the minimum necessary attributes for the RollingWindow object.

        Parameters
        ----------
            input : ndarray
                ndarray to be split into windows and used as inputs to model
            target : ndarray
                ndarray to be split into windows and used as targets
            window_length : int
                length of each rolling window
            shift : int
                number of time steps rolling window is shifted up before retraining/retesting
            model : class
                model class that has train and fit methods that will be used for each window
            fixed_parameter_dict : dict
                dictionary of model parameters that will not be tuned on a validation set
            train_test_split : float
                float between 0 and 1 determining split of each window between training and tesitng data
            hyperparameter_tuning : bool
                determines whether or not to tune hyperparameters on validation set before out of sample testing
            variable_model_parameter_grid : dict
                dictionary of model_parameter_name:tuning_list key:value pairs to tune during hyperparameter tuning
            train_test_val_split : tuple
                if 'hyperparameter_tuning' = True, rolling_window will be split according to (train,test,val) fractions
            standardization : string
                string indicating standardization method when fitting and testingmodels
            verbose : bool
                determines whether to print model summaries after each window
      """
        self.variable_parameter_grid =variable_model_parameter_grid
        self.target = target
        self.input = input
        self.fixed_model_parameters = fixed_model_parameters
        self.model = model
        self.data_length = len(input)
        self.window_length = window_length
        self.shift = shift
        self.train_test_split = train_test_split
        self.train_test_val_split = train_test_val_split
        self.verbose = verbose
        self.experiment_results = None
        self.hyperparameter_tuning = hyperparameter_tuning
        self.standardization = standardization
        self._experiment_indecies = None
        self._experiment_dict = None
    def get_indecies(self):
  ###return dictionary {train:[list of tuples where tuple[0] is start and tuple[1] is finish]}
      if self.hyperparameter_tuning == True:

        val = [(int(i),int(i+self.window_length*self.train_test_val_split[0])) for i in range(0,int(self.data_length-self.window_length+1),int(self.shift))]
        train = [(int(i+self.window_length*self.train_test_val_split[0]),
            int(i+self.window_length*self.train_test_val_split[0]+self.window_length*self.train_test_val_split[1])) for i in range(0,int(self.data_length-self.window_length+1),int(self.shift))]
        test = [(int(i+self.window_length*self.train_test_val_split[0]+self.window_length*self.train_test_val_split[1]),
           int(i+self.window_length*self.train_test_val_split[0]+self.window_length*self.train_test_val_split[1]+self.window_length*self.train_test_val_split[2]))
        for i in range(0,int(self.data_length-self.window_length+1),int(self.shift))]

        result_dict = {}
        result_dict["val"] = val
        result_dict["train"] = train
        result_dict["test"] = test

      else:
        train = [(int(i),int(i+self.window_length*self.train_test_split)) for i in range(0,int(self.data_length-self.window_length+1),int(self.shift))]
        test = [(int(i+self.window_length*self.train_test_split),
            int(i+self.window_length*self.train_test_split+self.window_length*(1-self.train_test_split))) 
            for i in range(0,int(self.data_length-self.window_length+1),int(self.shift))]

        result_dict = {}
        result_dict["train"] = train
        result_dict["test"] = test

        return result_dict

        Rolling_Validation.get_indecies =get_indecies

    @property
    def experiment_indecies(self):
      if type(self._experiment_indecies) == type(None):
        return self.get_indecies()
      else:
        return self._experiment_indecies


    def conduct_experiment(self):
        """
        Splits data into windows, trains, tests and (optionally) tunes model and returns out of sample predicitons for each window.

        Parameters
        ----------

        Returns
        -------
        result_dict : dict
            Dictionary of experiment_number:out_of_sample_prediciton key:value pairs that performance metrics can be computed with
        -------
        """
        windownumber = 0
        result_dict = {}
        true_dict = {}
        index_dict = self.experiment_indecies
  ###outer loop
        for window in range(len(index_dict["train"])):

          if self.hyperparameter_tuning== False:

            if self.standardization == None:
              X_train = self.input[index_dict["train"][window][0]:index_dict["train"][window][1]]
              y_train = self.target[index_dict["train"][window][0]:index_dict["train"][window][1]]

              X_test = self.input[index_dict["test"][window][0]:index_dict["test"][window][1]]
              y_test = self.target[index_dict["test"][window][0]:index_dict["test"][window][1]]
            else:
              X_train = self.standardization.fit_transform(self.input[index_dict["train"][window][0]:index_dict["train"][window][1]])

              y_train = self.target[index_dict["train"][window][0]:index_dict["train"][window][1]]

              X_test = self.standardization.transform(self.input[index_dict["test"][window][0]:index_dict["test"][window][1]])
              y_test = self.target[index_dict["test"][window][0]:index_dict["test"][window][1]]


          if self.verbose == True:
            print('''''''''''''Window Number: {}'''''''''''''''.format(windownumber))
            print("Training Initialized")

          total_parameters = self.fixed_model_parameters

          model = self.model(**total_parameters)
          model.fit(X_train,y_train)
          if self.verbose == True:
            print("Training Complete")

            print("Predicting")
          predictions = model.predict(X_test)



          result_dict[windownumber] = predictions
          true_dict[windownumber] = y_test

      

      
          windownumber+=1
        self.true_dict = true_dict
        self._experiment_dict = result_dict
        return result_dict
        Rolling_Experiment.conduct_experiment = conduct_experiment
    @property
    def experiment_dict(self):
      if type(self._experiment_dict) == type(None):
        return self.conduct_experiment()
      else:
        return self._experiment_dict
