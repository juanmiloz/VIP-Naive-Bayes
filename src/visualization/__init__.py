import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

from src.models.NBClassifier import NBClassifier


# Test to correct funtioning

def accuracy(y_true, y_pred):
    accuracy_num = np.sum(y_true == y_pred) / len(y_true)
    return accuracy_num


X, y = datasets.make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=123)

nb = NBClassifier()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

print("Naive Bayes classification accuracy", accuracy(y_test, predictions))

"""

# Test with data base with medicine

df = pd.read_csv("test.csv" )

X_train = df.values[:, :df.shape[1]-1]
y_train = df.values[:, df.shape[1]-1]

X_test = [[23, 1, 2, 1, 25.355], [47, 2, 1, 2, 10.321]]
nb = NBClassifier()
nb.fit(X_train, y_train.astype(int))
prediction = nb.predict(X_test)

print("The answers to your entered characteristics is:", prediction)
"""
