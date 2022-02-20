"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.W = None  # TODO: updated in train()
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in the Lecture.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implemented
        # Initializing the weights and bias randomly
        self.W = np.random.rand(self.n_class, X_train.shape[1]+1)

        # Cycling through training data
        for epoch in range(self.epochs):
            for obs in range(X_train.shape[0]):
                # Calculating predictions for all classes per observation
                X_t_bias = np.append(X_train[obs], 1.0)
                predictions = self.W @ X_t_bias
                # Linearly decaying learning rate as number of epochs increases
                decayed_lr = self.lr * (1 - epoch/self.epochs)
                update_vector = decayed_lr * X_t_bias  # Saving computation time
                # Update rule for each class
                for c in range(self.n_class):
                    if predictions[c] > predictions[y_train[obs]]:
                        self.W[y_train[obs]] += update_vector
                        self.W[c] -= update_vector

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
        # TODO: implemented
        # Get all class predictions for each test observation and select class with highest score
        return np.array([np.argmax(self.W @ np.append(X_test[i], 1)) for i in range(X_test.shape[0])])
