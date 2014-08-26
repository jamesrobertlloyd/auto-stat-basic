"""
Data type objects

Created August 2014

@authors: James Robert Lloyd (james.robert.lloyd@gmail.com)
"""

import numpy as np
import re
import os.path

import util

#### TODO
#### - Set up a propert object hierarchy


class XYDataSet():
    """Data set wrapper - in the future it will do clever memory things etc."""

    def __init__(self):
        self.X_labels = None
        self.y_label = None
        self.X = None
        self.y = None
        self.cv_indices = None
        self.name = ''
        self.path = ''

    def set_cv_indices(self, cv_indices):
        """Cross validation scheme"""
        self.cv_indices = cv_indices

    def load_from_file(self, fname):
        # Tidy up new lines if necessary
        with open(fname, 'rU') as data_file: # Opening up in universal new line mode
            contents = data_file.read()
        with open(fname, 'w') as data_file:
            data_file.write(contents)
        # Load numeric data - assume first row is header, comma delimited
        data = np.loadtxt(fname, delimiter=',', skiprows=1, ndmin=2)
        self.X = data[:, :-1]
        self.y = data[:, -1].flatten()
        # Load labels
        with open(fname, 'r') as data_file:
            labels = [re.sub('[.]', '', label) for label in data_file.readline().strip().split(',')]
            labels = [label if (not label == '') else '(blank)' for label in labels]
            if not len(labels) > 100:
                #### FIXME - this is a quick hack to guard against a newline bug
                labels = util.make_string_list_unique(labels)
            self.X_labels = labels[:-1]
            self.y_label = labels[-1]
        self.name = os.path.splitext(os.path.split(fname)[-1])[0]
        self.path = os.path.split(fname)[0]

    def copy(self):
        """Copy of self"""
        copy = XYDataSet()
        copy.X_labels = self.X_labels
        copy.y_label = self.y_label
        copy.X = self.X
        copy.y = self.y
        copy.cv_indices = self.cv_indices
        copy.name = self.name
        copy.path = self.path
        return copy

    def subsets(self, indices_list):
        """Given list of indices it returns lists of subset data sets"""
        data_sets = []
        for indices in indices_list:
            new_data_set = self.copy()
            new_data_set.X_labels = self.X_labels
            new_data_set.y_label = self.y_label
            new_data_set.X = self.X[indices, :]
            new_data_set.y = self.y[indices]
            new_data_set.cv_indices = None
            data_sets.append(new_data_set)
        return data_sets

    def input_subset(self, subset):
        """Subsets the input variables and returns the data set"""
        new_data_set = self.copy()
        new_data_set.X_labels = [self.X_labels[index] for index in subset]
        new_data_set.y_label = self.y_label
        new_data_set.X = self.X[:, subset]
        new_data_set.y = self.y
        return new_data_set

    @property
    def cv_subsets(self):
        subsets = []
        if not self.cv_indices is None:
            for train, test in self.cv_indices:
                subsets.append(tuple(self.subsets([train, test])))
            return subsets
        else:
            raise RuntimeError('Cross validation not specified')


class XSeqDataSet():
    """Data set wrapper - in the future it might do clever memory / disk things etc."""

    def __init__(self):
        self.labels = None
        self.X = None
        self.cv_indices = None
        self.name = ''
        self.path = ''

    def set_cv_indices(self, cv_indices):
        """Cross validation scheme"""
        self.cv_indices = cv_indices

    def load_from_file(self, fname):
        # Tidy up new lines if necessary
        with open(fname, 'rU') as data_file: # Opening up in universal new line mode
            contents = data_file.read()
        with open(fname, 'w') as data_file:
            data_file.write(contents)
        # Load numeric data - assume first row is header, comma delimited
        data = np.loadtxt(fname, delimiter=',', skiprows=1, ndmin=2)
        self.X = data
        # Load labels
        with open(fname, 'r') as data_file:
            labels = [re.sub('[.]', '', label) for label in data_file.readline().strip().split(',')]
            labels = [label if (not label == '') else '(blank)' for label in labels]
            if not len(labels) > 100:
                #### FIXME - this is a quick hack to guard against a newline bug
                labels = util.make_string_list_unique(labels)
            self.labels = labels
        self.name = os.path.splitext(os.path.split(fname)[-1])[0]
        self.path = os.path.split(fname)[0]

    def copy(self):
        """Copy of self"""
        copy = XSeqDataSet()
        copy.labels = self.labels
        copy.X = self.X
        copy.cv_indices = self.cv_indices
        copy.name = self.name
        copy.path = self.path
        return copy

    def subsets(self, indices_list):
        """Given list of indices it returns lists of subset data sets"""
        data_sets = []
        for indices in indices_list:
            new_data_set = self.copy()
            new_data_set.X = self.X[indices, :]
            new_data_set.cv_indices = None
            data_sets.append(new_data_set)
        return data_sets

    def variable_subsets(self, subset):
        """Subsets the variables and returns the data set"""
        new_data_set = self.copy()
        new_data_set.labels = [self.labels[index] for index in subset]
        new_data_set.X = self.X[:, subset]
        return new_data_set

    def convert_to_XY(self):
        """Turns into a regression data set by making the asumption that the final column in the target"""
        XY = XYDataSet()
        XY.X_labels = self.labels[:-1]
        XY.y_label = self.labels[-1]
        XY.X = self.X[:, :-1]
        XY.y = self.X[:, -1]
        XY.cv_indices = self.cv_indices
        XY.name = self.name
        XY.path = self.path
        return XY

    @property
    def cv_subsets(self):
        subsets = []
        if not self.cv_indices is None:
            for train, test in self.cv_indices:
                subsets.append(tuple(self.subsets([train, test])))
            return subsets
        else:
            raise RuntimeError('Cross validation not specified')