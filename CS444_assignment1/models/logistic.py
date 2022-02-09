"""Logistic regression model."""

from calendar import EPOCH
from os import XATTR_CREATE
import numpy as np
import pandas as pd


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """

        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.threshold = threshold

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        return 1 / (1 + -1*np.exp(np.inner(self.w, z)))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        self.w = np.random.rand(X_train.shape[1])

        for _ in range(self.epochs):
            for dp in range(X_train[0]):
                self.w = self.w + (self.lr * self.sigmoid(-y_train[dp]*np.inner(self.w, X_train[dp]))) 
        
        # TODO: implement me
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """

        predictions = []
        for dp in range(X_test.shape[0]):
            predictions.append(np.round(self.sigmoid(dp)))

        # TODO: implement me
        return predictions
