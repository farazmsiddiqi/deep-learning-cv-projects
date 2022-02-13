"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int, threshold: float):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """
        # TODO: implemented
        return 1 / (1 + np.exp(-1*np.inner(self.w, z)))

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        self.w = np.random.rand(X_train.shape[1])
        # can potentially scale the update rule to inc/dec confidence
        for epoch in range(self.epochs):
            decay = self.lr * (1 - epoch / self.epochs)
            for dp in range(X_train.shape[0]):
                if y_train[dp] == 1: #is a mushroom
                    self.w = self.w + (decay * self.sigmoid(-X_train[dp])) * X_train[dp]
                else:
                    self.w = self.w - (decay * self.sigmoid(X_train[dp])) * X_train[dp]

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
            predictions.append(np.round(self.sigmoid(X_test[dp])))

        return predictions
