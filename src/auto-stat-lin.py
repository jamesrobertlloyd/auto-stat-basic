"""
Basic version of automatic statistician implementing linear models and a couple of model building strategies
Design of objects and methods is reflecting eventual plans to have a message passing system but currently we are just
passing things around in memory for simplicity

Created June 2014

@authors: James Robert Lloyd (james.robert.lloyd@gmail.com)
          Christian Steinruecken (christian.steinruecken@gmail.com)
"""

from __future__ import division

import sklearn
import sklearn.linear_model
from sklearn.cross_validation import KFold

import numpy as np

##############################################
#                                            #
#                   Data                     #
#                                            #
##############################################


class DataSet():
    """Data set wrapper object thingy"""

    def __init__(self):
        self.X_labels = None
        self.y_labels = None
        self.X = None
        self.y = None

    def load_from_file(self, fname):
        # Load numeric data
        data = np.loadtxt(fname, delimiter=',', skiprows=1, ndmin=2)
        self.X = data[:, :-1]
        self.y = data[:, -1].flatten()
        # Load labels
        with open(fname, 'r') as data_file:
            labels = data_file.readline().strip().split(',')
            self.X_labels = labels[:-1]
            self.y_labels = labels[-1]

    def partition(self, indices_list):
        data_sets = []
        for indices in indices_list:
            new_data_set = DataSet()
            new_data_set.X_labels = self.X_labels
            new_data_set.y_labels = self.y_labels
            new_data_set.X = self.X[indices, :]
            new_data_set.y = self.y[indices]
            data_sets.append(new_data_set)
        return data_sets

##############################################
#                                            #
#                  Models                    #
#                                            #
##############################################


class LinearModel(sklearn.linear_model.LinearRegression):
    """Simple linear regression model"""

    def __init__(self):
        super(LinearModel, self).__init__(fit_intercept=True, normalize=False, copy_X=True)

    def train(self, data):
        self.fit(data.X, data.y)
        #### Remember the data - this should really be implemented with a load mechanism
        self.data = data

    def predict(self, data):
        return super(LinearModel, self).predict(data.X)

    def models(self):
        return [self]

    def describe(self):
        description = 'I am a linear model trained by least squares.\n'
        description += 'The output %s:' % self.data.y_labels
        # Sort predictors by size of coefficient
        coef_names_and_values = zip(self.data.X_labels, self.coef_)
        sorted_coef = sorted(coef_names_and_values, key=lambda a_pair: np.abs(a_pair[1]), reverse=True)
        n_predictors = 0
        for name_and_value in sorted_coef:
            name = name_and_value[0]
            value = name_and_value[1]
            if value > 0:
                n_predictors += 1
                description += '\n - increases linearly with input %s' % name
            elif value < 0:
                n_predictors += 1
                description += '\n - decreases linearly with input %s' % name
        if n_predictors == 0:
            description += '\n - does not vary with the inputs'
        return description


class LassoLinReg(sklearn.linear_model.LassoLarsCV):
    """Lasso trained linear regression model"""

    def __init__(self):
        super(LassoLinReg, self).__init__()

    def train(self, data):
        self.fit(data.X, data.y)
        ##### Remember the data - this should really be implemented with a load mechanism
        self.data = data

    def predict(self, data):
        return super(LassoLinReg, self).predict(data.X)

    def models(self):
        return [self]

    def describe(self):
        description = 'I am a linear model trained by cross validated LASSO.\n'
        description += 'The output %s:' % self.data.y_labels
        # Sort predictors by size of coefficient
        coef_names_and_values = zip(self.data.X_labels, self.coef_)
        sorted_coef = sorted(coef_names_and_values, key=lambda a_pair: np.abs(a_pair[1]), reverse=True)
        n_predictors = 0
        for name_and_value in sorted_coef:
            name = name_and_value[0]
            value = name_and_value[1]
            if value > 0:
                n_predictors += 1
                description += '\n - increases linearly with input %s' % name
            elif value < 0:
                n_predictors += 1
                description += '\n - decreases linearly with input %s' % name
        if n_predictors == 0:
            description += '\n - does not vary with the inputs'
        return description

##############################################
#                                            #
#             Cross validation               #
#                                            #
##############################################


class CrossValidationExpert():
    def __init__(self, model_class, data, fold_indices):
        self.model_class = model_class
        self.data = data
        self.fold_indices = fold_indices
        self.knowledge_base = []

    def run(self):
        # Calculate cross validated RMSE
        RMSE_sum = 0
        fold_count = 0

        for train, test in self.fold_indices:
            data_sets = self.data.partition([train, test])
            train_data = data_sets[0]
            test_data = data_sets[1]
            temp_model = self.model_class()
            temp_model.train(train_data)

            fold_count += 1
            RMSE_sum += np.sqrt(sklearn.metrics.mean_squared_error(test_data.y, temp_model.predict(test_data)))

        cv_RMSE = RMSE_sum / fold_count
        # Train on full data
        self.model = self.model_class()
        self.model.train(self.data)
        # Save fact to knowledge base
        self.knowledge_base.append({'label': 'CV-RMSE', 'model': self.model, 'score': cv_RMSE})
        #### This fact should also contain folds and whatnot....

    def models(self):
        return [self.model]

    @property
    def knowledge(self):
        return self.knowledge_base

##############################################
#                                            #
#                Diagnostics                 #
#                                            #
##############################################


class RegressionDiagnosticsExpert():
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.knowledge_base = []

    def run(self):
        # Calculate RMSE
        RMSE = np.sqrt(sklearn.metrics.mean_squared_error(self.data.y, self.model.predict(self.data)))
        # Save this to the knowledge base
        self.knowledge_base.append({'label': 'Test-set-RMSE', 'model': self.model, 'score': RMSE})
        #### This fact should reference the data somehow....
        # Perform some sort of hypothesis test
        pass

    @property
    def knowledge(self):
        return self.knowledge_base

##############################################
#                                            #
#                  Manager                   #
#                                            #
##############################################


class Manager():
    def __init__(self, data):
        """

        :type data: DataSet
        """
        self.data = data

    def run(self):
        # Partition data in train / test
        train_indices = range(0, int(np.floor(len(self.data.y) / 2)))
        test_indices  = range(int(np.floor(len(self.data.y) / 2)), int(len(self.data.y)))
        data_sets = self.data.partition([train_indices, test_indices])
        self.train_data = data_sets[0]
        self.test_data = data_sets[1]
        # Create folds of training data
        self.train_folds = KFold(len(self.train_data.y), n_folds=5, indices=False)
        # Initialise list of models
        self.experts = [CrossValidationExpert(LinearModel, self.train_data, self.train_folds),
                        CrossValidationExpert(LassoLinReg, self.train_data, self.train_folds)]
        # Train the models
        print('Experts running')
        for expert in self.experts:
            expert.run()
        # Get them to report back
        print('Experts reporting facts')
        self.knowledge_base = []
        for expert in self.experts:
            self.knowledge_base += expert.knowledge
        #### Ideally duplicates would be removed - this would be fastest if knowledge was hashable - currently not
        # Select / order
        self.cv_models = []
        for fact in self.knowledge_base:
            if fact['label'] == 'CV-RMSE':
                self.cv_models.append((fact['model'], fact['score']))
        self.cv_models = sorted(self.cv_models, key=lambda a_fact : a_fact[1])
        # Run model diagnostics
        print('Running diagnostics')
        for fact in self.cv_models:
            checking_expert = RegressionDiagnosticsExpert(fact[0], self.test_data)
            checking_expert.run()
            self.knowledge_base += checking_expert.knowledge
        # Report
        print('Creating report')

        print('\nList of models in order of cross validated error (best first)')
        for (i, fact) in enumerate(self.cv_models):
            print('\n%d\n' % (i+1))
            print(fact[0].describe())

        print('\nThose cross validated errors in full\n')
        for fact in self.cv_models:
            print(fact[1])

        print('\nTest set errors\n')
        for cv_fact in self.cv_models:
            model = cv_fact[0]
            for fact in self.knowledge_base:
                # Look for test set RMSE
                if (fact['label'] == 'Test-set-RMSE') and (fact['model'] == model):
                    print(fact['score'])

##############################################
#                                            #
#                   Main                     #
#                                            #
##############################################


def main():
    data = DataSet()
    data.load_from_file('../data/test-lin/simple-01.csv')
    manager = Manager(data)
    manager.run()

if __name__ == "__main__":
    main()