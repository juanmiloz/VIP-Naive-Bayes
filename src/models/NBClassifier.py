import base_models
import numpy as pd
import pandas as pd


class NBClassifier(base_models):

    def __init__(self):
        pass

    def fit(self, X, y):
   
    for i in range(X):
      p = np.bincount(X[:,i])
      ii = np.nonzero(y)[0]
      probabilities[i] = y[ii]/(X.shape[0]) #Calcula las probabilidades de cada uno de los feature de la matriz 
      
    
    pass

    def predict(self, X):
        pass
