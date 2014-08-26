"""
Miscellaneous utilities for the automatic statistician

Created August 2014

@authors: James Robert Lloyd (james.robert.lloyd@gmail.com)
"""

from numpy import log

import sklearn

import re


def make_string_list_unique(string_list):
    """Convert a list of strings into a list of unique strings (after removing spaces)"""
    unique_list = []
    while len(string_list) > 0:
        next_string = string_list[0]
        if re.sub(' ', '', next_string) in [re.sub(' ', '', el) for el in unique_list]:
            string_list[0] += ' (again)'
        else:
            unique_list.append(next_string)
            string_list = string_list[1:]
    return unique_list


def BIC(dist, data, n_params):
    MSE = sklearn.metrics.mean_squared_error(data.y, dist.conditional_mean(data))
    n = data.X.shape[0]
    return n * log(MSE) + n_params * log(n)