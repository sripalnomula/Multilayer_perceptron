# utils.py: Utility file for implementing helpful utility functions used by the ML algorithms.
#
# Submitted by: Sripal Reddy Nomula -- [srnomula]
#
# Based on skeleton code by CSCI-B 551 Fall 2022 Course Staff
import sys

import numpy as np
import scipy


def euclidean_distance(x1, x2):
    """
    Computes and returns the Euclidean distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """

    return np.sum(np.square(x1 - x2))


def manhattan_distance(x1, x2):
    """
    Computes and returns the Manhattan distance between two vectors.

    Args:
        x1: A numpy array of shape (n_features,).
        x2: A numpy array of shape (n_features,).
    """

    return np.sum(np.abs(x1 - x2))


def identity(x, derivative=False):
    """
    Computes and returns the identity activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    if derivative:
        return 1
    else:
        return x


def sigmoid(x, derivative=False):
    """
    Computes and returns the sigmoid (logistic) activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    x = np.clip(x, -1e30, 1e30)

    if derivative:
        sig_out = 1 / (1 + np.exp(-x))
        return sig_out * (1 - sig_out)
    else:
        return 1 / (1 + np.exp(-x))


def tanh(x, derivative=False):
    """
    Computes and returns the hyperbolic tangent activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """
    x = np.clip(x, -1e100, 1e100)
    # tan_h = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    if derivative:
        return 1 - np.square(x)
    else:
        return np.tanh(x)


def relu(x, derivative=False):
    """
    Computes and returns the rectified linear unit activation function of the given input data x. If derivative = True,
    the derivative of the activation function is returned instead.

    Args:
        x: A numpy array of shape (n_samples, n_hidden).
        derivative: A boolean representing whether or not the derivative of the function should be returned instead.
    """

    if derivative:
        return np.where(x > 0.0, 1, 0)
    return np.where(x > 0.0, x, 0)


def softmax(x, derivative=False):
    x = np.clip(x, 1e-100, 1e100)
    if not derivative:
        c = np.max(x, axis=1, keepdims=True)
        return np.exp(x - c - np.log(np.sum(np.exp(x - c), axis=1, keepdims=True)))

    else:
        return softmax(x) * (1 - softmax(x))


def cross_entropy(y, p):
    """
    Computes and returns the cross-entropy loss, defined as the negative log-likelihood of a logistic model that returns
    p probabilities for its true class labels y.

    Args:
        y:
            A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the
            input data used when fitting the model.
        p:
            A numpy array of shape (n_samples, n_outputs) representing the predicted probabilities from the softmax
            output activation function.
    """

    loss = 0
    # p = np.clip(p, 1e-30, 1e30)
    for x_y, x_p in zip(y, p):
        loss -= (x_y * np.log(x_p + pow(10, -10)) + ((1 - x_y) * np.log(1 - x_p + pow(10, -10))))

    return loss/len(y)



def one_hot_encoding(y):
    """
    Converts a vector y of categorical target class values into a one-hot numeric array using one-hot encoding: one-hot
    encoding creates new binary-valued columns, each of which indicate the presence of each possible value from the
    original data.

    Args:
        y: A numpy array of shape (n_samples,) representing the target class values for each sample in the input data.

    Returns:
        A numpy array of shape (n_samples, n_outputs) representing the one-hot encoded target class values for the input
        data. n_outputs is equal to the number of unique categorical class values in the numpy array y.
    """

    n_outputs = len(set(y))
    output = np.zeros((len(y), n_outputs))
    unique_labels = list(set(y))  # to maintain order and convert back to list
    for ind, sample in enumerate(y):
        output[ind, unique_labels.index(sample)] = 1
    return output

    # raise NotImplementedError('This function must be implemented by the student.')
