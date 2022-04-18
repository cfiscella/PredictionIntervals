from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from src.features.ts_process. import shape_data, dissolve

class WindowStandardScaler(StandardScaler):

  def fit(self,X):
    X_dissolved = dissolve(X,X.shape[1])
    StandardScaler.fit(self,X_dissolved)

  def transform(self,X):

    X_dissolved = dissolve(X,X.shape[1])
    transformed = StandardScaler.transform(self,X_dissolved)

    return shape_data(pd.DataFrame(transformed),X.shape[1])

  def fit_transform(self,X):
    self.fit(X)
    return self.transform(X)

  def inverse_transform(self,X):
    X_dissolved = dissolve(X,X.shape[1])
    inverse_transformed = StandardScaler.inverse_transform(self,X_dissolved)
    return shape_data(pd.DataFrame(inverse_transformed),X.shape[1])
  
class WindowMinMaxScaler(MinMaxScaler):
  def fit(self,X):
    X_dissolved = dissolve(X,X.shape[1])
    MinMaxScaler.fit(self,X_dissolved)

  def transform(self,X):

    X_dissolved = dissolve(X,X.shape[1])
    transformed = MinMaxScaler.transform(self,X_dissolved)

    return shape_data(pd.DataFrame(transformed),X.shape[1])

  def fit_transform(self,X):
    self.fit(X)
    return self.transform(X)

  def inverse_transform(self,X):
    X_dissolved = dissolve(X,X.shape[1])
    inverse_transformed = MinMaxScaler.inverse_transform(self,X_dissolved)
    return shape_data(pd.DataFrame(inverse_transformed),X.shape[1])
