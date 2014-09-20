"""
Data type objects

Created August 2014

@authors: James Robert Lloyd (james.robert.lloyd@gmail.com)
"""

import numpy as np
import re
import os.path
import copy

import util

#### TODO
#### - Extend to include info about missing data
#### - Extend to include info about manipulated variables
#### - What is the correct final architecture for this - ultimately needs to handle any database


class DataSet(object):
    """
    Data set wrapper - in the future it will do clever memory things etc.
    Also, this currently maintains a list of arrays and a list of labels for those arrays
    Ultimately we would want to store generic relational databases
    """

    def __init__(self):
        self.name = ''
        self.path = ''  # Location of data file - useful to keep around
        self.arrays = {}
        self.labels = {}
        self.cv_indices = None

    def set_cv_indices(self, cv_indices):
        """Cross validation scheme"""
        self.cv_indices = cv_indices

    def load_from_file(self, fname):
        raise RuntimeError('load_from_file not implemented')

    def copy(self):
        """
        :rtype: DataSet
        """
        return copy.deepcopy(self)  # This should be sufficient for most data sets

    def subsets(self, indices_list):
        """Given list of indices it returns lists of subset data sets"""
        data_sets = []
        for indices in indices_list:
            new_data_set = self.copy()
            for (key, array) in self.arrays.iteritems():
                # Subset all data sets
                # TODO - should this be a deep copy?
                new_data_set.arrays[key] = array[indices]
            data_sets.append(new_data_set)
        return data_sets

    @property
    def cv_subsets(self):
        subsets = []
        if not self.cv_indices is None:
            for train, test in self.cv_indices:
                subsets.append(tuple(self.subsets([train, test])))
            return subsets
        else:
            raise RuntimeError('Cross validation not specified')


class XSeqDataSet(DataSet):
    def __init__(self):
        super(XSeqDataSet, self).__init__()
        self.labels['X'] = None
        self.arrays['X'] = None

    def load_from_file(self, fname):
        # Tidy up new lines if necessary
        with open(fname, 'rU') as data_file:  # Opening up in universal new line mode
            contents = data_file.read()
        with open(fname, 'w') as data_file:
            data_file.write(contents)
        # Load numeric data - assume first row is header, comma delimited
        data = np.loadtxt(fname, delimiter=',', skiprows=1, ndmin=2)
        self.arrays['X'] = data
        # Load labels
        with open(fname, 'r') as data_file:
            labels = [re.sub('[.]', '', label) for label in data_file.readline().strip().split(',')]
            labels = [label if (not label == '') else '(blank)' for label in labels]
            if not len(labels) > 100:
                #### FIXME - this is a quick hack to guard against a newline bug
                labels = util.make_string_list_unique(labels)
            self.labels['X'] = labels
        self.name = os.path.splitext(os.path.split(fname)[-1])[0]
        self.path = os.path.split(fname)[0]

    def variable_subsets(self, subset):
        """Subsets the variables and returns the data set"""
        new_data_set = self.copy()
        new_data_set.labels['X'] = [self.labels['X'][index] for index in subset]
        new_data_set.arrays['X'] = self.arrays['X'][:, subset]
        return new_data_set

    def convert_to_XY(self):
        """Turns into a regression data set by making the asumption that the final column in the target"""
        # TODO - take a position of output variable as a target
        XY = XYDataSet()
        XY.labels['X'] = self.labels['X'][:-1]
        XY.labels['X'] = self.labels['X'][-1]
        XY.arrays['X'] = self.arrays['X'][:, :-1]
        XY.arrays['y'] = self.arrays['X'][:, -1]
        XY.cv_indices = self.cv_indices
        XY.name = self.name
        XY.path = self.path
        return XY


class XYDataSet(DataSet):
    """Data set wrapper - in the future it will do clever memory things etc."""

    def __init__(self):
        super(XYDataSet, self).__init__()
        self.X_labels = None
        self.y_label = None
        self.X = None
        self.y = None
        self.cv_indices = None
        self.name = ''
        self.path = ''

    def load_from_file(self, fname):
        # Tidy up new lines if necessary
        with open(fname, 'rU') as data_file:  # Opening up in universal new line mode
            contents = data_file.read()
        with open(fname, 'w') as data_file:
            data_file.write(contents)
        # Load numeric data - assume first row is header, comma delimited
        data = np.loadtxt(fname, delimiter=',', skiprows=1, ndmin=2)
        self.arrays['X'] = data[:, :-1]
        self.arrays['y'] = data[:, -1].flatten()
        # Load labels
        with open(fname, 'r') as data_file:
            labels = [re.sub('[.]', '', label) for label in data_file.readline().strip().split(',')]
            labels = [label if (not label == '') else '(blank)' for label in labels]
            if not len(labels) > 100:
                #### FIXME - this is a quick hack to guard against a newline bug
                labels = util.make_string_list_unique(labels)
            self.labels['X'] = labels[:-1]
            self.labels['y'] = labels[-1]
        self.name = os.path.splitext(os.path.split(fname)[-1])[0]
        self.path = os.path.split(fname)[0]

    def input_subset(self, subset):
        """Subsets the input variables and returns the data set"""
        new_data_set = self.copy()
        new_data_set.labels['X'] = [self.labels['X'][index] for index in subset]
        new_data_set.labels['y'] = self.labels['y']
        new_data_set.arrays['X'] = self.arrays['X'][:, subset]
        new_data_set.arrays['y'] = self.arrays['y']
        return new_data_set