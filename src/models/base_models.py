from abc import ABC, abstractmethod


class SupervisedModel(ABC):
    """
    This is an abstract class for implementing classification models following a sklearn-like implementation
    and with step functions that will be useful to plot progressively the functionality of the model.
    """

    def __init__(self):
        """
        Constructor

        Here you define the hyper-parameters and configurations of your model as attributes
        """
        super(SupervisedModel, self).__init__()

    @abstractmethod
    def step_fit(self, x, y):
        """
        
        """
        pass

    @abstractmethod
    def step_predict(self, x):
        """

        """
        pass

    @abstractmethod
    def fit(self, X, y):
        """
        Trains the classifier.

        Arguments:
          X: numpy.ndarray, an (m,n) matrix representing the features of
             the data with which the model will be trained
          y: numpy.array, an (m) array contianing the corresponding
             classes of each feature or datapoint in X
        """
        # This method should call step_fit
        pass

    @abstractmethod
    def predict(self, X):
        """
        Perform inference on the trained model.

        pre: the classifier is already trained.

        Arguments:
        X: numpy.ndarray, an (m,n) matrix representing the features of
            the data with which the model will be trained

        Returns:
        y: numpy.array, an (m) array containing the class predicted for each datapoint in X
        """
        # This method should call step_predict
        pass
