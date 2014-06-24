"""
Basic version of automatic statistician implementing linear models and a couple of model building strategies
This version is designed to run as a single threaded batch job - this is to produce a demo quickly and test some
model building, description and criticism ideas in a more familiar environment

Created June 2014

@authors: James Robert Lloyd (james.robert.lloyd@gmail.com)
          Christian Steinruecken (christian.steinruecken@gmail.com)
"""

from __future__ import division

import sklearn
import sklearn.linear_model
import sklearn.ensemble
from sklearn.cross_validation import KFold

import numpy as np
import random

#### TODO
#### - Distributions should have cross validated noise levels
#### - Implement some more serious model checks
#### - Implement stepwise regression expert that cross validates over depth

##############################################
#                                            #
#                   Data                     #
#                                            #
##############################################


class XYDataSet():
    """Data set wrapper - in the future it will do clever memory things etc."""

    def __init__(self):
        self.X_labels = None
        self.y_label = None
        self.X = None
        self.y = None
        self.cv_indices = None

    def set_cv_indices(self, cv_indices):
        """Cross validation scheme"""
        self.cv_indices = cv_indices

    def load_from_file(self, fname):
        # Load numeric data - assume first row is header, comma delimited
        data = np.loadtxt(fname, delimiter=',', skiprows=1, ndmin=2)
        self.X = data[:, :-1]
        self.y = data[:, -1].flatten()
        # Load labels
        with open(fname, 'r') as data_file:
            labels = data_file.readline().strip().split(',')
            self.X_labels = labels[:-1]
            self.y_label = labels[-1]

    def subsets(self, indices_list):
        """Given list of indices it returns lists of subset data sets"""
        data_sets = []
        for indices in indices_list:
            new_data_set = XYDataSet()
            new_data_set.X_labels = self.X_labels
            new_data_set.y_label = self.y_label
            new_data_set.X = self.X[indices, :]
            new_data_set.y = self.y[indices]
            data_sets.append(new_data_set)
        return data_sets

    def input_subset(self, subset):
        """Subsets the input variables and returns the data set"""
        #### FIXME - This should call a copy routine
        new_data_set = XYDataSet()
        new_data_set.X_labels = [self.X_labels[index] for index in subset]
        new_data_set.y_label = self.y_label
        new_data_set.X = self.X[:, subset]
        new_data_set.y = self.y
        new_data_set.cv_indices = self.cv_indices
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

##############################################
#                                            #
#                Utilities                   #
#                                            #
##############################################


def lin_mod_txt_description(coef, data):
    """Simple description of a linear model"""
    description = ''
    description += 'The output %s:' % data.y_label
    # Sort predictors by size of coefficient
    coef_names_and_values = zip(data.X_labels, coef)
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
    summary = 'A linear model with %d active inputs' % n_predictors
    return summary, description


def BIC(dist, data, n_params):
    MSE = sklearn.metrics.mean_squared_error(data.y, dist.conditional_mean(data))
    n = data.X.shape[0]
    return n * np.log(MSE) + n_params * np.log(n)

##############################################
#                                            #
#              Distributions                 #
#                                            #
##############################################


class SKLearnModelPlusGaussian(object):
    """Conditional distribution based on sklearn model with iid Gaussian noise"""

    def __init__(self, model, sd):
        self.model = model
        self.sd = sd

    def conditional_mean(self, data):
        return self.model.predict(data.X)

    def conditional_sample(self, data):
        return self.conditional_mean(data) + self.sd * np.random.randn(data.X.shape[0], 1)


class SKLearnModelInputFilteredPlusGaussian(object):
    """Conditional distribution based on sklearn model with iid Gaussian noise"""
    # FIXME - IRL this should be create with a pre-processing object

    def __init__(self, model, sd, subset):
        self.model = model
        self.sd = sd
        self.subset = subset

    def conditional_mean(self, data):
        return self.model.predict(data.input_subset(self.subset).X)

    def conditional_sample(self, data):
        return self.conditional_mean(data) + self.sd * np.random.randn(data.X.shape[0], 1)

##############################################
#                                            #
#                  Models                    #
#                                            #
##############################################


class SKLearnModel(object):
    """Wrapper for sklearn models"""

    def __init__(self, base_class):
        self.sklearn_class = base_class
        self.model = self.sklearn_class()
        self.data = None
        self.conditional_distributions = []
        self.knowledge_base = []

    def load_data(self, data):
        assert isinstance(data, XYDataSet)
        self.data = data

    def run(self):
        self.model.fit(self.data.X, self.data.y)
        y_hat = self.model.predict(self.data.X)
        sd = np.sqrt((sklearn.metrics.mean_squared_error(self.data.y, y_hat)))
        self.conditional_distributions = [SKLearnModelPlusGaussian(self.model, sd)]
        self.generate_descriptions()

    def generate_descriptions(self):
        # I can be replaced with a generic description routine - e.g. average predictive comparisons
        raise RuntimeError('Description not implemented')


class SKLinearModel(SKLearnModel):
    """Simple linear regression model based on sklearn implementation"""

    def __init__(self):
        super(SKLinearModel, self).__init__(lambda: sklearn.linear_model.LinearRegression(fit_intercept=True,
                                                                                          normalize=False,
                                                                                          copy_X=True))

    def generate_descriptions(self):
        summary, description = lin_mod_txt_description(coef=self.model.coef_, data=self.data)
        self.knowledge_base.append(dict(label='summary', text=summary,
                                        distribution=self.conditional_distributions[0], data=self.data))
        self.knowledge_base.append(dict(label='description', text=description,
                                        distribution=self.conditional_distributions[0], data=self.data))


class SKLassoReg(SKLearnModel):
    """Lasso trained linear regression model"""

    def __init__(self):
        super(SKLassoReg, self).__init__(sklearn.linear_model.LassoLarsCV)

    def generate_descriptions(self):
        summary, description = lin_mod_txt_description(coef=self.model.coef_, data=self.data)
        self.knowledge_base.append(dict(label='summary', text=summary,
                                        distribution=self.conditional_distributions[0], data=self.data))
        self.knowledge_base.append(dict(label='description', text=description,
                                        distribution=self.conditional_distributions[0], data=self.data))

class SKLearnRandomForestReg(SKLearnModel):

    def __init__(self):
        super(SKLearnRandomForestReg, self).__init__(lambda: sklearn.ensemble.RandomForestRegressor(
                                                                n_estimators=100))

    def generate_descriptions(self):
        self.knowledge_base.append(dict(label='summary', text='I am random forest',
                                        distribution=self.conditional_distributions[0], data=self.data))
        self.knowledge_base.append(dict(label='description', text='I am still random forest',
                                        distribution=self.conditional_distributions[0], data=self.data))


class BICBackwardsStepwiseLin(object):
    """BIC guided backwards stepwise linear regression"""

    def __init__(self):
        self.model = SKLinearModel()
        self.data = None
        self.knowledge_base = []
        self.conditional_distributions = []
        self.subset = []

    def load_data(self, data):
        assert isinstance(data, XYDataSet)
        self.data = data

    def run(self):
        self.subset = range(len(self.data.X_labels))
        self.model.load_data(self.data)
        self.model.run()
        current_BIC = BIC(self.model.conditional_distributions[0], self.data, len(self.model.model.coef_))
        improvement = True
        while improvement and (len(self.subset) > 0):
            improvement = False
            best_BIC = current_BIC
            # Try removing all input variables
            for i in range(len(self.subset)):
                temp_subset = self.subset[:i] + self.subset[(i+1):]
                temp_data_set = self.data.input_subset(temp_subset)
                temp_model = SKLinearModel()
                temp_model.load_data(temp_data_set)
                temp_model.run()
                temp_BIC = BIC(temp_model.conditional_distributions[0], temp_data_set, len(temp_model.model.coef_))
                if temp_BIC < best_BIC:
                    best_model = temp_model
                    best_subset = temp_subset
                    best_BIC = temp_BIC
            if best_BIC < current_BIC:
                improvement = True
                self.model = best_model
                self.subset = best_subset
                current_BIC = best_BIC
        y_hat = self.model.conditional_distributions[0].conditional_mean(self.data.input_subset(self.subset))
        sd = np.sqrt((sklearn.metrics.mean_squared_error(self.data.y, y_hat)))
        #### FIXME - model.model is clearly ugly :)
        self.conditional_distributions = [SKLearnModelInputFilteredPlusGaussian(self.model.model, sd, self.subset)]
        self.generate_descriptions()

    def generate_descriptions(self):
        summary, description = lin_mod_txt_description(coef=self.model.model.coef_,
                                                       data=self.data.input_subset(self.subset))
        self.knowledge_base.append(dict(label='summary', text=summary,
                                        distribution=self.conditional_distributions[0], data=self.data))
        self.knowledge_base.append(dict(label='description', text=description,
                                        distribution=self.conditional_distributions[0], data=self.data))

##############################################
#                                            #
#             Cross validation               #
#                                            #
##############################################


class CrossValidationExpert(object):
    def __init__(self, model_class):
        """
        :type data: XYDataSet
        """
        self.model_class = model_class
        self.model = None
        self.data = None
        self.knowledge_base = []
        self.conditional_distributions = []

    def load_data(self, data):
        assert isinstance(data, XYDataSet)
        self.data = data

    def run(self):
        # Calculate cross validated RMSE
        RMSE_sum = None
        fold_count = 0

        for train_data, test_data in self.data.cv_subsets:
            temp_model = self.model_class()
            temp_model.load_data(train_data)
            temp_model.run()
            distributions = temp_model.conditional_distributions

            fold_count += 1
            if RMSE_sum is None:
                RMSE_sum = np.zeros(len(distributions))
            for (i, distribution) in enumerate(distributions):
                RMSE_sum[i] += np.sqrt(sklearn.metrics.mean_squared_error(test_data.y,
                                                                          distribution.conditional_mean(test_data)))

        cv_RMSE = RMSE_sum / fold_count
        # Train on full data
        self.model = self.model_class()
        self.model.load_data(self.data)
        self.model.run()
        # Extract knowledge about the model
        self.knowledge_base += self.model.knowledge_base
        # Save facts about CV-RMSE to knowledge base
        for (score, distribution) in zip(cv_RMSE, self.model.conditional_distributions):
            self.knowledge_base.append(dict(label='CV-RMSE', distribution=distribution, value=score, data=self.data))
            # Modify noise levels of model if appropriate
            if isinstance(distribution, SKLearnModelInputFilteredPlusGaussian) or \
               isinstance(distribution, SKLearnModelPlusGaussian):
                distribution.sd = score

##############################################
#                                            #
#                Diagnostics                 #
#                                            #
##############################################


class RegressionDiagnosticsExpert():
    def __init__(self):
        self.conditional_distribution = None
        self.data = None
        self.knowledge_base = []

    def clear(self):
        self.conditional_distribution = None
        self.data = None
        self.knowledge_base = []

    def load_data(self, data):
        assert isinstance(data, XYDataSet)
        self.data = data

    def load_model(self, model):
        self.conditional_distribution = model

    def run(self):
        # Calculate RMSE
        RMSE = np.sqrt(sklearn.metrics.mean_squared_error(self.data.y,
                                                          self.conditional_distribution.conditional_mean(self.data)))
        # Save this to the knowledge base
        self.knowledge_base.append(dict(label='RMSE', distribution=self.conditional_distribution,
                                        value=RMSE, data=self.data))
        # Perform some sort of hypothesis test
        pass

##############################################
#                                            #
#                  Manager                   #
#                                            #
##############################################


class Manager():
    def __init__(self):
        self.data = None
        self.knowledge_base = []
        self.clear()

    def clear(self):
        self.data = None
        self.knowledge_base = []

    def load_data(self, data):
        assert isinstance(data, XYDataSet)
        self.data = data

    def run(self):
        #### This code is messy as is
        #### e.g. knowledge is completely unstructured and may contain duplicates

        # Partition data in train / test
        #### FIXME - Random or user specified partition?
        # train_indices = range(0, int(np.floor(len(self.data.y) / 2)))
        # test_indices  = range(int(np.floor(len(self.data.y) / 2)), int(len(self.data.y)))
        random_perm = range(len(self.data.y))
        random.shuffle(random_perm)
        train_indices = random_perm[:int(np.floor(len(self.data.y) / 2))]
        test_indices  = random_perm[int(np.floor(len(self.data.y) / 2)):]
        data_sets = self.data.subsets([train_indices, test_indices])
        train_data = data_sets[0]
        test_data = data_sets[1]
        # Create folds of training data
        train_folds = KFold(len(train_data.y), n_folds=5, indices=False)
        train_data.set_cv_indices(train_folds)
        # Initialise list of models and experts
        experts = [CrossValidationExpert(SKLinearModel),
                   CrossValidationExpert(SKLassoReg),
                   CrossValidationExpert(BICBackwardsStepwiseLin),
                   CrossValidationExpert(SKLearnRandomForestReg)]
        # Train the models
        print('Experts running')
        for expert in experts:
            expert.load_data(train_data)
            expert.run()
        # Get them to report back
        print('Experts reporting facts')
        self.knowledge_base = []
        for expert in experts:
            self.knowledge_base += expert.knowledge_base
        #### FIXME Ideally duplicates would be removed - this would be fastest if knowledge was hashable - currently not
        # Select / order
        self.cv_dists = []
        for fact in self.knowledge_base:
            if fact['label'] == 'CV-RMSE':
                self.cv_dists.append((fact['distribution'], fact['value']))
        self.cv_dists = sorted(self.cv_dists, key=lambda a_fact : a_fact[1])
        # Run model diagnostics
        print('Running diagnostics')
        for fact in self.cv_dists:
            checking_expert = RegressionDiagnosticsExpert()
            checking_expert.load_data(test_data)
            checking_expert.load_model(fact[0])
            checking_expert.run()
            self.knowledge_base += checking_expert.knowledge_base
        # Report
        print('Creating report')

        print('\nList of models in order of cross validated error (best first)\n')
        for (i, fact) in enumerate(self.cv_dists):
            dist = fact[0]
            for other_fact in self.knowledge_base:
                if (other_fact['label'] == 'summary') and (other_fact['distribution'] == dist):
                    print('%d: %s' % ((i+1), other_fact['text']))

        print('\nFull description of best model\n')
        dist = self.cv_dists[0][0]
        for other_fact in self.knowledge_base:
            if (other_fact['label'] == 'description') and (other_fact['distribution'] == dist):
                print(other_fact['text'])

        print('\nThose cross validated errors in full\n')
        for fact in self.cv_dists:
            print(fact[1])

        print('\nTest set errors\n')
        for cv_fact in self.cv_dists:
            dist = cv_fact[0]
            for fact in self.knowledge_base:
                # Look for test set RMSE
                if (fact['label'] == 'RMSE') and (fact['data'] == test_data) and (fact['distribution'] == dist):
                    print(fact['value'])

    @property
    def knowledge(self):
        return self.knowledge_base

##############################################
#                                            #
#                   Main                     #
#                                            #
##############################################


def main():
    data = XYDataSet()
    # data.load_from_file('../data/test-lin/simple-01.csv')
    # data.load_from_file('../data/test-lin/uci-slump-test.csv')
    data.load_from_file('../data/test-lin/uci-housing.csv')
    manager = Manager()
    manager.load_data(data)
    manager.run()

if __name__ == "__main__":
    main()