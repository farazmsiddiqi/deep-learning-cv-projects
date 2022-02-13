"""Perceptron model."""

from typing import final
import numpy as np
from sklearn.feature_selection import SelectFdr


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None  # TODO: change this (weights)
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.exp_param = .5

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape[0], X_train.shape[1]

        self.w = np.random.rand(self.n_class, D)

        for epoch in range(self.epochs):
            for dp in range(N):
                # take the weight matrix (not vector, bc multiclass) and multiply it by the feature vector of one datapoint
                predictions = self.w @ X_train[dp]

                decay = self.lr * (1 - (N * epoch + dp) / (N * self.epochs)) ** self.exp_param
                update_vec = decay * X_train[dp]

                for class_label in range(self.n_class):
                    if class_label == y_train[dp]:
                        self.w[class_label] = self.w[class_label] + update_vec
                    elif predictions[class_label] > predictions[y_train[dp]]:
                        self.w[class_label] = self.w[class_label] - update_vec

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

        final_predictions = []
        for dp in range(X_test.shape[0]):
            final_predictions.append(np.argmax(self.w @ X_test[dp]))

        return final_predictions

