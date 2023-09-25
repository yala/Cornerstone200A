import numpy as np
import tqdm

class LogisticRegression():
    """
        A logistic regression model trained with stochastic gradient descent.
    """

    def __init__(self, num_epochs=100, learning_rate=1e-4, batch_size=16, regularization_lambda=0,  verbose=False):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.verbose = verbose
        self.regularization_lambda = regularization_lambda

    def fit(self, X, Y):
        """
            Train the logistic regression model using stochastic gradient descent.
        """
        raise NotImplementedError("Not implemented yet")

    def gradient(self, X, Y):
        """
            Compute the gradient of the loss with respect to theta and bias with L2 Regularization.
            Hint: Pay special attention to the numerical stability of your implementation.
        """
        raise NotImplementedError("Not implemented yet")eta, grad_bias

    def predict_proba(self, X):
        """
            Predict the probability of lung cancer for each sample in X.
        """
        raise NotImplementedError("Not implemented yet")

    def predict(self, X, threshold=0.5):
        """
            Predict the if patient will develop lung cancer for each sample in X.
        """
        raise NotImplementedError("Not implemented yet")