# import base_models
import numpy as np


class NBClassifier():

    def __init__(self):
        pass

    def fit(self, X, y):
        num_samples, num_features = X.shape  # assigment to num_samples and numFeature the value of shape
        self._classes = np.unique(y) # obtain the unique value for each classes
        num_classes = len(self._classes) # assigment the length to classes

        # np.float64 same float but an accurate representation of the values
        # np.zeros creates an array of (n,m) size, and typenp.float64 and fills it with zeros
        self._mean = np.zeros((num_classes, num_features), dtype=np.float64)
        self._var = np.zeros((num_classes, num_features), dtype=np.float64)
        self._priors = np.zeros(num_classes, dtype=np.float64)

        for i in self._classes:
            X_i = X[i == y]
            self._mean[i, :] = X_i.mean(axis=0)
            self._var[i, :] = X_i.var(axis=0)
            self._priors[i] = X_i.shape[0] / float(num_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return y_pred

    def _predict(self, x):
        posteriors = []

        for idx, i in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x-mean)**2 / (2*var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator/denominator
