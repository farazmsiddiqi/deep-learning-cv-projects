# """Support Vector Machine (SVM) model."""

# import numpy as np


# class SVM:
#     def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
#         """Initialize a new classifier.

#         Parameters:
#             n_class: the number of classes
#             lr: the learning rate
#             epochs: the number of epochs to train for
#             reg_const: the regularization constant
#         """
#         self.w = None
#         self.lr = lr
#         self.epochs = epochs
#         self.reg_const = reg_const
#         self.n_class = n_class

#     def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
#         """Calculate gradient of the svm hinge loss.

#         Inputs have dimension D, there are C classes, and we operate on
#         mini-batches of N examples.

#         Parameters:
#             X_train: a numpy array of shape (N, D) containing a mini-batch
#                 of data
#             y_train: a numpy array of shape (N,) containing training labels;
#                 y[i] = c means that X[i] has label c, where 0 <= c < C

#         Returns:
#             the gradient with respect to weights w; an array of the same shape
#                 as w
#         """
#         # TODO: implement me
#         return

#     def train(self, X_train: np.ndarray, y_train: np.ndarray):
#         """Train the classifier.

#         Hint: operate on mini-batches of data for SGD.

#         Parameters:
#             X_train: a numpy array of shape (N, D) containing training data;
#                 N examples with D dimensions
#             y_train: a numpy array of shape (N,) containing training labels
#         """
#         return

#     def predict(self, X_test: np.ndarray) -> np.ndarray:
#         """Use the trained weights to predict labels for test data points.

#         Parameters:
#             X_test: a numpy array of shape (N, D) containing testing data;
#                 N examples with D dimensions

#         Returns:
#             predicted labels for the data in X_test; a 1-dimensional array of
#                 length N, where each element is an integer giving the predicted
#                 class.
#         """
#         # TODO: implement me
#         return
"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.W = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        # TODO: implemented
        N = X_train.shape[0]
        grad_w = np.zeros((self.n_class, X_train.shape[1]))

        for obs in range(N):
            # Scores for all classes
            scores = self.W @ X_train[obs]
            # Score for correct class
            correct_score = scores[y_train[obs]]
            hinge_losses = 1 - correct_score + scores  # calculate hinge losses (ignored in for loop if non-positive)
            update_vector = X_train[obs] * self.lr  # Saving computation time
            # compute gradient for each observation and store values according to update rule from slides
            for c in range(self.n_class):
                if hinge_losses[c] > 0 and c != y_train[obs]:
                    grad_w[c] += update_vector  # add (not subtract) because we do -grad in train()
                    grad_w[y_train[obs]] -= update_vector  # subtract (not add) because we do -grad in train()
        return grad_w + self.W * self.reg_const / N


    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        # TODO: implemented
        N = X_train.shape[0]

        self.W = np.random.rand(self.n_class, X_train.shape[1])  # Initializing the weights randomly
        mini_batch_size = min((250, N))  # in case we have fewer than 250 training points

        for _ in range(self.epochs):
            batch_idx = np.random.choice(N, mini_batch_size)  # choose <batch_size> i.i.d. samples
            # Update weights using gradient (don't multiply by lr because already multiplied in calc_gradient)
            self.W -= self.calc_gradient(X_train[batch_idx], y_train[batch_idx])

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
        final_predictions = []
        for dp in range(X_test.shape[0]):
            final_predictions.append(np.argmax(self.W @ X_test[dp]))

        return final_predictions


