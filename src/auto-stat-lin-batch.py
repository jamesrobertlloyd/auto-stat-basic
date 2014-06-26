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
from sklearn.metrics import mean_squared_error as MSE
from sklearn.cross_decomposition import CCA

from scipy import stats

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

import numpy as np

import random

#### TODO
#### - Visuals!
#### - Calibrated outlier detection - or kolmogorov smirnov test?
#### - Make sampling several times more efficient - I think it could be a bottleneck
#### - Identify other bottlenecks
#### - Make manager prefer simplicity
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


def rank(x):
    """
    :type x: np.array
    """
    return x.argsort().argsort()


def RDC(x, y, k=10, s=0.2):
    """Randomised dependency criterion"""
    # FIXME - this should be tied rank
    x = x.flatten()
    y = y.flatten()
    x = np.vstack((rank(x) / len(x), np.ones(len(x)))).T
    y = np.vstack((rank(y) / len(y), np.ones(len(y)))).T
    x = np.sin(0.5 * s * np.dot(x, np.random.randn(2, k)))
    y = np.sin(0.5 * s * np.dot(y, np.random.randn(2, k)))
    x = np.hstack((x, np.ones((x.shape[0], 1))))
    y = np.hstack((y, np.ones((y.shape[0], 1))))
    temp_CCA = CCA()
    temp_CCA.fit(x, y)
    x, y = temp_CCA.transform(x, y)
    return abs(stats.pearsonr(x[:, 0], y[:, 0])[0])

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
        return self.model.predict(data.X).ravel()

    def conditional_sample(self, data):
        return (self.conditional_mean(data) + (self.sd * np.random.randn(data.X.shape[0], 1)).ravel()).ravel()


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
        return (self.conditional_mean(data) + (self.sd * np.random.randn(data.X.shape[0], 1)).ravel()).ravel()

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

    def generate_figures(self):
        # Plot training data against fit
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(1,1,1) # one row, one column, first plot
        y_hat = self.conditional_distributions[0].conditional_mean(self.data)
        sorted_y_hat = np.sort(y_hat)
        ax.plot(sorted_y_hat, sorted_y_hat, color="blue")
        ax.scatter(y_hat, self.data.y, color="red", marker="o")
        ax.set_title("Training data against fit")
        ax.set_xlabel("Model fit")
        ax.set_ylabel("Training data")
        fig.savefig("../temp-report/figures/lin-train-fit.pdf")
        plt.close()
        # Plot data against all dimensions
        for dim in range(self.data.X.shape[1]):
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(1,1,1) # one row, one column, first plot
            ax.scatter(self.data.X[:, dim], self.data.y, color="red", marker="o")
            ax.set_title("Training data against %s" % self.data.X_labels[dim])
            ax.set_xlabel(self.data.X_labels[dim])
            ax.set_ylabel("Training data")
            fig.savefig("../temp-report/figures/lin-train-%s.pdf" % self.data.X_labels[dim])
            plt.close()
        # Plot rest of model against fit
        for dim in range(self.data.X.shape[1]):
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(1,1,1) # one row, one column, first plot
            y_hat = self.conditional_distributions[0].conditional_mean(self.data)
            component_fit = self.model.coef_[dim] * self.data.X[:, dim].ravel()
            partial_resid = self.data.y - (y_hat - component_fit)
            plot_idx = np.argsort(self.data.X[:, dim].ravel())
            ax.plot(self.data.X[plot_idx, dim], component_fit[plot_idx], color="blue")
            ax.scatter(self.data.X[:, dim], partial_resid, color="red", marker="o")
            ax.set_title("Partial residual against %s" % self.data.X_labels[dim])
            ax.set_xlabel(self.data.X_labels[dim])
            ax.set_ylabel("Partial residual")
            fig.savefig("../temp-report/figures/lin-partial-resid-%s.pdf" % self.data.X_labels[dim])
            plt.close()
        # Plot residuals against each dimension
        for dim in range(self.data.X.shape[1]):
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(1,1,1) # one row, one column, first plot
            y_hat = self.conditional_distributions[0].conditional_mean(self.data)
            resid = self.data.y - y_hat
            ax.scatter(self.data.X[:, dim], resid, color="red", marker="o")
            ax.set_title("Residuals against %s" % self.data.X_labels[dim])
            ax.set_xlabel(self.data.X_labels[dim])
            ax.set_ylabel("Residuals")
            fig.savefig("../temp-report/figures/lin-resid-%s.pdf" % self.data.X_labels[dim])
            plt.close()

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
        # FIXME - The following line is a bit of a hack
        self.model.generate_figures()
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
        self.boot_iters = 1000

    def clear(self):
        self.conditional_distribution = None
        self.data = None
        self.knowledge_base = []

    def load_data(self, data):
        assert isinstance(data, XYDataSet)
        self.data = data

    def load_model(self, model):
        self.conditional_distribution = model

    # TODO - this can be abstracted nicely - just functions of residuals and model fit / data

    def RMSE_test(self):
        # TODO - turn this into chi squared version of test for heteroscedasticity?
        # Compute statistics on data
        y_hat = self.conditional_distribution.conditional_mean(self.data)
        RMSE = np.sqrt(MSE(self.data.y, y_hat))
        # Calculate sampling distribution
        sample_RMSEs = np.zeros(self.boot_iters)
        for i in range(self.boot_iters):
            y_rep = self.conditional_distribution.conditional_sample(self.data)
            sample_RMSEs[i] = np.sqrt(MSE(y_rep, y_hat))
        # Calculate p value
        p_RMSE = np.sum(sample_RMSEs > RMSE) / self.boot_iters
        # Generate a description of this fact
        description = 'RMSE of %f which yields a p-value of %f' % (RMSE, p_RMSE)
        # Save this to the knowledge base
        self.knowledge_base.append(dict(label='RMSE', distribution=self.conditional_distribution,
                                        value=RMSE, data=self.data, description=description, p_value=p_RMSE))

    def corr_test(self):
        """Test correlation of residuals with fit term"""
        # Compute statistics on data
        y_hat = self.conditional_distribution.conditional_mean(self.data)
        corr = abs(stats.pearsonr(y_hat, self.data.y - y_hat)[0])
        # Calculate sampling distribution
        sample_corrs = np.zeros(self.boot_iters)
        for i in range(self.boot_iters):
            y_rep = self.conditional_distribution.conditional_sample(self.data)
            sample_corrs[i] = abs(stats.pearsonr(y_hat, y_rep - y_hat)[0])
        # Calculate p value
        p_corr = np.sum(sample_corrs > corr) / self.boot_iters
        # Generate a description of this fact
        description = 'Correlation of %f which yields a p-value of %f' % (corr, p_corr)
        # Save this to the knowledge base
        self.knowledge_base.append(dict(label='corr-test', distribution=self.conditional_distribution,
                                        value=corr, data=self.data, description=description, p_value=p_corr))

    def RDC_test(self):
        """Test correlation of residuals with fit term using randomised dependence coefficient"""
        # Compute statistics on data
        y_hat = self.conditional_distribution.conditional_mean(self.data)
        corr = RDC(y_hat, self.data.y - y_hat)
        # Calculate sampling distribution
        sample_corrs = np.zeros(self.boot_iters)
        for i in range(self.boot_iters):
            y_rep = self.conditional_distribution.conditional_sample(self.data)
            sample_corrs[i] = RDC(y_hat, y_rep - y_hat)
        # Calculate p value
        p_corr = np.sum(sample_corrs > corr) / self.boot_iters
        # Generate a description of this fact
        description = 'RDC of %f which yields a p-value of %f' % (corr, p_corr)
        # Save this to the knowledge base
        self.knowledge_base.append(dict(label='RDC-test', distribution=self.conditional_distribution,
                                        value=corr, data=self.data, description=description, p_value=p_corr))

    def corr_test_multi_dim(self):
        """Test correlation of residuals with inputs using randomised dependence coefficient"""
        for dim in range(self.data.X.shape[1]):
            # Compute statistics on data
            y_hat = self.conditional_distribution.conditional_mean(self.data)
            corr = abs(stats.pearsonr(self.data.X[:,dim], self.data.y - y_hat)[0])
            # Calculate sampling distribution
            sample_corrs = np.zeros(self.boot_iters)
            for i in range(self.boot_iters):
                y_rep = self.conditional_distribution.conditional_sample(self.data)
                sample_corrs[i] = abs(stats.pearsonr(self.data.X[:,dim], y_rep - y_hat)[0])
            # Calculate p value
            p_corr = np.sum(sample_corrs > corr) / self.boot_iters
            # Generate a description of this fact
            description = 'Correlation of %f on %s which yields a p-value of %f' % (corr, self.data.X_labels[dim], p_corr)
            # Save this to the knowledge base
            self.knowledge_base.append(dict(label='corr-test-dim', distribution=self.conditional_distribution,
                                            value=corr, data=self.data, description=description, p_value=p_corr))

    def RDC_test_multi_dim(self):
        """Test correlation of residuals with inputs using randomised dependence coefficient"""
        for dim in range(self.data.X.shape[1]):
            # Compute statistics on data
            y_hat = self.conditional_distribution.conditional_mean(self.data)
            corr = RDC(self.data.X[:,dim], self.data.y - y_hat)
            # Calculate sampling distribution
            sample_corrs = np.zeros(self.boot_iters)
            for i in range(self.boot_iters):
                y_rep = self.conditional_distribution.conditional_sample(self.data)
                sample_corrs[i] = RDC(self.data.X[:,dim], y_rep - y_hat)
            # Calculate p value
            p_corr = np.sum(sample_corrs > corr) / self.boot_iters
            # Generate a description of this fact
            description = 'RDC of %f on %s which yields a p-value of %f' % (corr, self.data.X_labels[dim], p_corr)
            # Save this to the knowledge base
            self.knowledge_base.append(dict(label='RDC-test-dim', distribution=self.conditional_distribution,
                                            value=corr, data=self.data, description=description, p_value=p_corr))

    def corr_test_multi_dim_data(self):
        """Test correlation of residuals with inputs using randomised dependence coefficient"""
        for dim in range(self.data.X.shape[1]):
            # Compute statistics on data
            y_hat = self.conditional_distribution.conditional_mean(self.data)
            corr = abs(stats.pearsonr(self.data.X[:,dim], self.data.y)[0])
            # Calculate sampling distribution
            sample_corrs = np.zeros(self.boot_iters)
            for i in range(self.boot_iters):
                y_rep = self.conditional_distribution.conditional_sample(self.data)
                sample_corrs[i] = abs(stats.pearsonr(self.data.X[:,dim], y_rep)[0])
            # Calculate p value
            p_corr = np.sum(sample_corrs > corr) / self.boot_iters
            # Generate a description of this fact
            description = 'Correlation on data of %f on %s which yields a p-value of %f' % (corr, self.data.X_labels[dim], p_corr)
            # Save this to the knowledge base
            self.knowledge_base.append(dict(label='corr-test-dim-data', distribution=self.conditional_distribution,
                                            value=corr, data=self.data, description=description, p_value=p_corr))

    def RDC_test_multi_dim_data(self):
        """Test correlation of residuals with inputs using randomised dependence coefficient"""
        for dim in range(self.data.X.shape[1]):
            # Compute statistics on data
            y_hat = self.conditional_distribution.conditional_mean(self.data)
            corr = RDC(self.data.X[:,dim], self.data.y)
            # Calculate sampling distribution
            sample_corrs = np.zeros(self.boot_iters)
            for i in range(self.boot_iters):
                y_rep = self.conditional_distribution.conditional_sample(self.data)
                sample_corrs[i] = RDC(self.data.X[:,dim], y_rep)
            # Calculate p value
            p_corr = np.sum(sample_corrs > corr) / self.boot_iters
            # Generate a description of this fact
            description = 'RDC on data of %f on %s which yields a p-value of %f' % (corr, self.data.X_labels[dim], p_corr)
            # Save this to the knowledge base
            self.knowledge_base.append(dict(label='RDC-test-dim-data', distribution=self.conditional_distribution,
                                            value=corr, data=self.data, description=description, p_value=p_corr))

    def benjamini_hochberg(self, alpha=0.05):
        """Orders p-values in facts and records which facts are discoveries"""
        # FIXME - Currently assumes that knowledge base only contains p values
        # Sort facts
        self.knowledge_base.sort(key=lambda fact: fact['p_value'])
        # Apply BH procedure
        discoveries = []
        for (i, fact) in enumerate(self.knowledge_base):
            if fact['p_value'] <= alpha * (i + 1) / len(self.knowledge_base):
                discoveries.append(fact)
            else:
                break
        self.knowledge_base.append(dict(label='BH-discoveries', alpha=alpha, discoveries=discoveries))

    def run(self):
        self.RMSE_test()
        self.corr_test()
        # self.RDC_test()
        # self.corr_test_multi_dim()
        # self.RDC_test_multi_dim()
        # self.corr_test_multi_dim_data()
        # self.RDC_test_multi_dim_data()
        # self.benjamini_hochberg(alpha=0.05)

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
        experts = [CrossValidationExpert(SKLinearModel)]#,
                   #CrossValidationExpert(SKLassoReg),
                   #CrossValidationExpert(BICBackwardsStepwiseLin),
                   #CrossValidationExpert(SKLearnRandomForestReg)]
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
        for fact in [self.cv_dists[0]]:# FIXME - this is a hack to only check the best model
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

        print('\nModel criticism discoveries\n')
        for fact in self.knowledge_base:
            if fact['label'] == 'BH-discoveries':
                for discovery in fact['discoveries']:
                    print(discovery['description'])

        # print('\nThose cross validated errors in full\n')
        # for fact in self.cv_dists:
        #     print(fact[1])
        #
        # print('\nTest set errors\n')
        # for cv_fact in self.cv_dists:
        #     dist = cv_fact[0]
        #     for fact in self.knowledge_base:
        #         # Look for test set RMSE
        #         if (fact['label'] == 'RMSE') and (fact['data'] == test_data) and (fact['distribution'] == dist):
        #             print(fact['value'])

        # print('\nTest set error descriptions\n')
        # for cv_fact in self.cv_dists:
        #     dist = cv_fact[0]
        #     for fact in self.knowledge_base:
        #         # Look for test set RMSE
        #         if (fact['label'] == 'RMSE') and (fact['data'] == test_data) and (fact['distribution'] == dist):
        #             print(fact['description'])
        #
        # print('\nCorrelation test descriptions\n')
        # for cv_fact in self.cv_dists:
        #     dist = cv_fact[0]
        #     for fact in self.knowledge_base:
        #         # Look for test set RMSE
        #         if (fact['label'] == 'corr-test') and (fact['data'] == test_data) and (fact['distribution'] == dist):
        #             print(fact['description'])
        #
        # print('\nCorrelation dimension test descriptions\n')
        # for cv_fact in self.cv_dists:
        #     dist = cv_fact[0]
        #     for fact in self.knowledge_base:
        #         # Look for test set RMSE
        #         if (fact['label'] == 'corr-test-dim') and (fact['data'] == test_data) and (fact['distribution'] == dist):
        #             print(fact['description'])
        #
        # print('\nCorrelation on data dimension test descriptions\n')
        # for cv_fact in self.cv_dists:
        #     dist = cv_fact[0]
        #     for fact in self.knowledge_base:
        #         # Look for test set RMSE
        #         if (fact['label'] == 'corr-test-dim-data') and (fact['data'] == test_data) and (fact['distribution'] == dist):
        #             print(fact['description'])
        #
        # print('\nRDC test descriptions\n')
        # for cv_fact in self.cv_dists:
        #     dist = cv_fact[0]
        #     for fact in self.knowledge_base:
        #         # Look for test set RMSE
        #         if (fact['label'] == 'RDC-test') and (fact['data'] == test_data) and (fact['distribution'] == dist):
        #             print(fact['description'])
        #
        # print('\nRDC dimension test descriptions\n')
        # for cv_fact in self.cv_dists:
        #     dist = cv_fact[0]
        #     for fact in self.knowledge_base:
        #         # Look for test set RMSE
        #         if (fact['label'] == 'RDC-test-dim') and (fact['data'] == test_data) and (fact['distribution'] == dist):
        #             print(fact['description'])
        #
        # print('\nRDC on data dimension test descriptions\n')
        # for cv_fact in self.cv_dists:
        #     dist = cv_fact[0]
        #     for fact in self.knowledge_base:
        #         # Look for test set RMSE
        #         if (fact['label'] == 'RDC-test-dim-data') and (fact['data'] == test_data) and (fact['distribution'] == dist):
        #             print(fact['description'])

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
