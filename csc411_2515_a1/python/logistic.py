""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid
from math import *

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """
    # In case of MNIST classification of 4 and 9, output will be integer values
    # TODO: Finish this function
    N, M = data.shape
    y = [0]*N
    for i in range(0, N):
        z = 0
        for j in range(0, M):
            z = z + weights[j] * data[i,j] + weights[-1]
        y[i] = sigmoid(z)
        #iprint 'z y[i]', z, y[i]
    augdata = np.ones((N, M+1))
    augdata[:, :-1] = data
    z = np.dot(augdata, weights) # z is N x 1
    y = sigmoid(z)
    return y

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of binary targets. Values should be either 0 or 1
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy.  CE(p, q) = E_p[-log q].  Here
                       we want to compute CE(targets, y).
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    # TODO: Finish this function
    c = 0
    ce = 0.0
    for i in range(0, len(y)):
        if (y[i][0] > 0.5 and targets[i][0] == 1):
            c = c+1
        if (y[i][0] < 0.5 and targets[i][0] == 0):
            c = c+1
        ce = ce + float(np.power(y[i], targets[i])[0] * np.power(1-y[i], 1-targets[i])[0])
    frac_correct = 1.000*c/len(targets)
    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """
    # TODO: Finish this function
    N, M = data.shape
    z = [0.0]*N
    df = np.zeros((M+1, 1))
    f = 0.0
    augdata = np.ones((N, M+1))
    augdata[:, :-1] = data
    z = np.dot(augdata, weights) # z is N x 1
    f = float(np.dot(np.transpose(1-targets), z)[0] + sum(np.log(np.exp(-z) + 1))) # f is scalar
    df = (np.dot(np.transpose(augdata), np.subtract(sigmoid(z).reshape((N, 1)), targets.reshape((N, 1)))))
    y = np.array(logistic_predict(weights, data))
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """

    # TODO: Finish this function
    N, M = data.shape
    lam = hyperparameters['weight_regularization']
    z = [0.0]*N
    df = np.zeros((M+1, 1))
    f = 0.0
    augdata = np.ones((N, M+1))
    augdata[:, :-1] = data
    penweights = weights
    penweights[-1] = 0
    z = np.dot(augdata, weights) # z is N x 1
    f = float(np.dot(np.transpose(1-targets), z)[0] + sum(np.log(np.exp(-z) + 1))) + lam*sum(np.power(weights[0:-1], 2)) # f is scalar
    df = (np.dot(np.transpose(augdata), np.subtract(sigmoid(z).reshape((N, 1)), targets.reshape((N, 1))))) + lam * penweights.reshape((M+1, 1))
    y = np.array(logistic_predict(weights, data))
    return f, df, y
