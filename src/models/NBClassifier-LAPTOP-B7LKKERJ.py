import base_models
import numpy as pd
import pandas as pd


class NBClassifier(base_models):

    def __init__(self):
        pass

    def dataToNumbers(self):
        pass

    def fit(self, X, y):
        for i in range(len(y)):
            print("x")

    def predict(self, X):
        pass

    ##P(y|X) = [P(y)*P(X|y)]/P(X)
    ##probabilidad que pase y dato que X(alguna columna de X)