"""
Miscellaneous utilities for the automatic statistician

Created August 2014

@authors: James Robert Lloyd (james.robert.lloyd@gmail.com)
"""

from numpy import log
import numpy as np

import sklearn

import re


def make_string_list_unique(string_list):
    """Convert a list of strings into a list of unique strings, in-place. Also replaces blanks with 'blank'"""
    seenset = set()

    for ind, x in enumerate(string_list):
        i = 1
        newx = ' '.join(x.split())  # converts all whitespace to space
        newx = re.sub("[^0-9a-zA-Z_\-()+*/?&#@'{}%!$;. ]", "", newx)

        if newx == '':
            newx = 'blank'

        while newx in seenset:
            newx = '{} ({})'.format(x, i)
            i += 1
        seenset.add(newx)
        string_list[ind] = newx


def BIC(dist, data, n_params):
    MSE = sklearn.metrics.mean_squared_error(data.y, dist.conditional_mean(data))
    n = data.X.shape[0]
    return n * log(MSE) + n_params * log(n)


# def RBF_kernel_loop(X, Y, lengthscales):
#     K = np.zeros((X.shape[0], Y.shape[0]))
#     for i in range(K.shape[0]):
#         for j in range(K.shape[1]):
#             for d in range(lengthscales.size):
#                 K[i, j] += ((X[i, d] - Y[j, d]) ** 2) / (lengthscales[d] ** 2)
#     return np.exp(-K/2)


def RBF_kernel(X, Y, lengthscales):
    X = X / np.tile(lengthscales, (X.shape[0], 1))
    Y = Y / np.tile(lengthscales, (Y.shape[0], 1))

    G = np.sum(X * X, 1)
    H = np.sum(Y * Y, 1)

    Q = np.tile(G, (Y.shape[0], 1)).T
    R = np.tile(H, (X.shape[0], 1))

    K = Q + R - 2*np.dot(X, Y.T)
    K = np.exp(-K/2)

    return K


def MMD(X, Y, lengthscales):
    K_XX = RBF_kernel(X, X, lengthscales)
    K_YY = RBF_kernel(Y, Y, lengthscales)
    K_XY = RBF_kernel(X, Y, lengthscales)

    m = X.shape[0]
    n = Y.shape[0]

    return (np.sum(K_XX) / (m ** 2)) - 2 * (np.sum(K_XY) / (m * n)) + (np.sum(K_YY) / (n ** 2))