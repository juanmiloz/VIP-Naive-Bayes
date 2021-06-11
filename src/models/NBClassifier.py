# import base_models
import numpy as np


class NBClassifier():

    def __init__(self):
        pass

    """
    Trains the NBClassifier by calculating the priors and the mean & variance for each class.
    
    Arguments:
        X: numpy.ndarray, an (m,n) matrix representing the features of
        the data with which the model will be trained. training data.
        y: numpy.array, an (m) array contianing the corresponding
        classes of each feature or datapoint in X. training labels.
    """
    def fit(self, X, y):
        num_samples, num_features = X.shape  # assigment to num_samples and numFeature the value of the shape(rows x colums) of X.
        self._classes = np.unique(y) # obtain the unique elements of the Y array to get an array that lists all classes in integer values. ej: y[1,0,1,2,2] _clases[0,1,2]
        print(self._classes)
        num_classes = len(self._classes) # the number of classes is the length of the list of classes.

        #init mean, var, priors
        # np.float64 same float but an accurate representation of the values(allows more digits)
        # np.zeros creates an array of (n,m) size, and typenp.float64 and fills it with zeros.
        self._mean = np.zeros((num_classes,num_features), dtype=np.float64)
        self._var = np.zeros((num_classes, num_features), dtype=np.float64)
        #one prior per class.
        self._priors = np.zeros(num_classes,dtype=np.float64)

        #iterates trough the classes to asign the mean,var,and prior for each class
        for cl in self._classes:
            #geting the samples that contains the class cl.
                #to get the samples, as Y has the same length as the X_features, when cl == y indicates the row in X_features (and in y)  that contains the class cl.
            X_cl = X[cl==y]


            #for the row cl in mean and var, .mean & .var calculates the mean and var for each column in the row of the class cl.
            self._mean[cl,:] = X_cl.mean(axis=0)
            self._var[cl,:] = X_cl.var(axis=0)

            #the prior probability is the frecuency of the current class in the training samples.
                # number of samples that contains the class cl / number of total samples.
            self._priors[cl] = X_cl.shape[0] / float(num_samples)


    """
        predicts for X database with multiple samples, by predicting each sample and adding the predictions to a y_pred array.

        Arguments:
            self.
            X:numpy.ndarray, an (m,n) matrix representing the features of the data.
    """
    def predict(self, X):
        #iterates trough all samples of X.
        y_pred = [self._predict(x) for x in X]
        #returns the predictions.
        return y_pred

    """
        predicts for a single samples by calculating the posterior probabilities, the class conditional,and the pior for each class
        and choose the class with the highest probability.

        Arguments:
            self.

    """
    def _predict(self, x):

        posteriors = []

        #this for iterates trough the classes returning each class and the index in self._classes by using enumerate method.
        for i, cl in enumerate(self._classes):
            #gets the prior for current class and aplies the log function for other calculations.
            prior = np.log(self._priors[i])

            class_conditional = np.sum(np.log(self._pdf(i, x)))
            posterior = prior + class_conditional
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x-mean)**2 / (2*var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator/denominator
