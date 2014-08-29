"""
Statistical expert autonomous agent classes

This will also contain code for distribution objects since their code may be very much intertwined with that of
experts that perform inference

Created August 2014

@authors: James Robert Lloyd (james.robert.lloyd@gmail.com)
"""

import numpy as np

import sklearn
import sklearn.linear_model
import sklearn.ensemble
from sklearn.cross_validation import KFold

# from multiprocessing import Process
from threading import Thread
from multiprocessing import Queue as multi_q
from Queue import Empty as q_Empty

import time

from agent import Agent, start_communication
from data import XYDataSet, XSeqDataSet
# import util

##############################################
#                                            #
#              Distributions                 #
#                                            #
##############################################


class MeanPlusGaussian(object):
    """Conditional distribution - mean plus iid Gaussian noise"""

    def __init__(self, mean, sd):
        self.mean = mean
        self.sd = sd

    def clear_cache(self):
        # FIXME - this is not how caching should work - SUCH A HACK
        pass

    def conditional_mean(self, data):
        return self.mean * np.ones((data.X.shape[0])).ravel()

    def conditional_sample(self, data):
        return (self.conditional_mean(data) + (self.sd * np.random.randn(data.X.shape[0], 1)).ravel()).ravel()


class SKLearnModelPlusGaussian(object):
    """Conditional distribution based on sklearn model with iid Gaussian noise"""

    def __init__(self, model, sd):
        self.model = model
        self.sd = sd
        self.conditional_mean_ = None

    def clear_cache(self):
        # FIXME - this is not how caching should work - SUCH A HACK
        self.conditional_mean_ = None

    def conditional_mean(self, data):
        # FIXME - this is not how caching should work - SUCH A HACK
        if self.conditional_mean_ is None:
            self.conditional_mean_ = self.model.predict(data.X).ravel()
        return self.conditional_mean_

    def conditional_sample(self, data):
        return (self.conditional_mean(data) + (self.sd * np.random.randn(data.X.shape[0], 1)).ravel()).ravel()


class SKLearnModelInputFilteredPlusGaussian(object):
    """Conditional distribution based on sklearn model with iid Gaussian noise"""
    # FIXME - IRL this should be created with a pre-processing object

    def __init__(self, model, sd, subset):
        self.model = model
        self.sd = sd
        self.subset = subset
        self.conditional_mean_ = None

    def clear_cache(self):
        # FIXME - this is not how caching should work - SUCH A HACK
        self.conditional_mean_ = None

    def conditional_mean(self, data):
        # FIXME - this is not how caching should work - SUCH A HACK
        if self.conditional_mean_ is None:
            self.conditional_mean_ = self.model.predict(data.input_subset(self.subset).X)
        return self.conditional_mean_

    def conditional_sample(self, data):
        return (self.conditional_mean(data) + (self.sd * np.random.randn(data.X.shape[0], 1)).ravel()).ravel()

##############################################
#                                            #
#                  Models                    #
#                                            #
##############################################


class DistributionModel(object):
    """Wrapper for a distribution"""
    def __init__(self, dist):
        self.conditional_distributions = [dist]


class SKLearnModel(Agent):
    """Wrapper for sklearn models"""

    def __init__(self, base_class, *args, **kwargs):
        super(SKLearnModel, self).__init__(*args, **kwargs)

        self.sklearn_class = base_class
        self.model = self.sklearn_class()
        self.data = None
        self.conditional_distributions = []
        self.knowledge_base = []

    def load_data(self, data):
        assert isinstance(data, XYDataSet)
        self.data = data

    def fit(self):
        self.model.fit(self.data.X, self.data.y)
        y_hat = self.model.predict(self.data.X)
        sd = np.sqrt((sklearn.metrics.mean_squared_error(self.data.y, y_hat)))
        self.conditional_distributions = [SKLearnModelPlusGaussian(self.model, sd)]
        # self.generate_descriptions()

    @staticmethod
    def generate_descriptions():
        # I can be replaced with a generic description routine - e.g. average predictive comparisons
        raise RuntimeError('Description not implemented')


class SKLinearModel(SKLearnModel):
    """Simple linear regression model based on sklearn implementation"""

    def __init__(self, *args, **kwargs):
        super(SKLinearModel, self).__init__(lambda: sklearn.linear_model.LinearRegression(fit_intercept=True,
                                                                                          normalize=False,
                                                                                          copy_X=True),
                                            *args, **kwargs)


class SKLassoReg(SKLearnModel):
    """Lasso trained linear regression model"""

    def __init__(self, *args, **kwargs):
        super(SKLassoReg, self).__init__(sklearn.linear_model.LassoLarsCV, *args, **kwargs)


class SKLearnRandomForestReg(SKLearnModel):
    """Good ol' random forest"""

    def __init__(self, *args, **kwargs):
        super(SKLearnRandomForestReg, self).__init__(lambda: sklearn.ensemble.RandomForestRegressor(n_estimators=100),
                                                     *args, **kwargs)

##############################################
#                                            #
#               Meta experts                 #
#                                            #
##############################################


class CrossValidationExpert(Agent):
    """
    Takes an expert as input, assumes the expert learns some number of conditional distributions, cross validates the
    performance of these distributions, returns the distributions and cross validation scores to its parent
    """
    def __init__(self, sub_expert_class, *args, **kwargs):
        """
        :type data: XSeqDataSet
        """
        super(CrossValidationExpert, self).__init__(*args, **kwargs)

        self.sub_expert_class = sub_expert_class
        self.sub_expert = None
        self.data = None
        self.conditional_distributions = []

    def load_data(self, data):
        assert isinstance(data, XSeqDataSet)
        self.data = data

    def cross_validate(self):
        # Quick hack - turn into XY data set for initial testing purposes
        self.data = self.data.convert_to_XY()
        # Set up cross validation scheme
        train_folds = KFold(self.data.X.shape[0], n_folds=5, indices=False)
        self.data.set_cv_indices(train_folds)
        # Calculate cross validated RMSE
        RMSE_sum = None
        var_explained_sum = None
        fold_count = 0

        for train_data, test_data in self.data.cv_subsets:
            if self.termination_pending:
                break
            temp_expert = self.sub_expert_class()
            temp_expert.load_data(train_data)
            temp_expert.fit()
            distributions = temp_expert.conditional_distributions

            fold_count += 1
            if RMSE_sum is None:
                RMSE_sum = np.zeros(len(distributions))
                var_explained_sum = np.zeros(len(distributions))
            for (i, distribution) in enumerate(distributions):
                RMSE_sum[i] += np.sqrt(sklearn.metrics.mean_squared_error(test_data.y,
                                                                          distribution.conditional_mean(test_data)))
                var_explained_sum[i] += 100 * (1 - (sklearn.metrics.mean_squared_error(test_data.y,
                                                                    distribution.conditional_mean(test_data)) /
                                                    np.var(test_data.y)))
        if not self.termination_pending:

            cv_RMSE = RMSE_sum / fold_count
            cv_var_explained = var_explained_sum / fold_count
            # Train on full data
            self.sub_expert = self.sub_expert_class()
            self.sub_expert.load_data(self.data)
            self.sub_expert.fit()
            # Report results of cross validation
            for (rmse_score, var_score, distribution) in zip(cv_RMSE, cv_var_explained,
                                                             self.sub_expert.conditional_distributions):
                # Modify noise levels of model if appropriate
                if isinstance(distribution, SKLearnModelInputFilteredPlusGaussian) or \
                   isinstance(distribution, SKLearnModelPlusGaussian):
                    distribution.sd = rmse_score
                self.outbox.append(dict(label='CV-RMSE', distribution=distribution, value=rmse_score,
                                        var_explained=var_score, data=self.data))

    def next_action(self):
        if not self.termination_pending:
            if not self.data is None:
                self.cross_validate()
                self.terminated = True
            else:
                time.sleep(1)


class DataDoublingExpert(Agent):
    """
    Follows a data doubling strategy to turn any other expert into an anytime system
    Assumes that the sub expert will terminate of its own accord
    """
    def __init__(self, sub_expert_class, *args, **kwargs):
        """
        :type data: XSeqDataSet
        """
        super(DataDoublingExpert, self).__init__(*args, **kwargs)

        self.sub_expert_class = sub_expert_class
        self.expert_queue = multi_q()
        self.queues_to_children = [multi_q()]

        self.data = None
        self.conditional_distributions = []

        self.data_size = 100
        self.state = 'run'

        self.terminate_next_run = False
        self.epoch = 0

    def load_data(self, data):
        assert isinstance(data, XSeqDataSet)
        self.data = data

    def run_sub_expert(self):
        # Should I terminate?
        if self.terminate_next_run:
            self.terminated = True
        else:
            self.epoch += 1
            # Reduce data size if appropriate - if so this is the last run
            if self.data_size >= self.data.X.shape[0]:
                self.data_size = self.data.X.shape[0]
                self.terminate_next_run = True
            # Create a new expert and hook up communication
            sub_expert = self.sub_expert_class()
            sub_expert.inbox_q = self.queues_to_children[0]
            sub_expert.outbox_q = self.expert_queue
            # Create and load data
            subset_data = self.data.subsets([range(self.data_size)])[0]
            sub_expert.load_data(subset_data)
            # Create and launch sub expert process
            p = Thread(target=start_communication, args=(sub_expert,))
            p.start()
            # Delete local reference to sub_expert to avoid confusion
            del sub_expert
            self.child_processes = [p]
            self.state = 'wait'
            # Make the data larger for next time
            self.data_size *= 2

    def wait_for_sub_expert(self):
        time.sleep(0.1)
        if not self.child_processes[0].is_alive():
            # Sub expert has finished - time to run the next expert after reading any final messages
            self.state = 'run'
        while not self.termination_pending:
            try:
                message = self.expert_queue.get_nowait()
                # Message received, add some additional details
                message['epoch'] = self.epoch
                message['sender'] = self.name
                self.outbox.append(message)
            except q_Empty:
                break

    def next_action(self):
        if not self.termination_pending:
            if not self.data is None:
                if self.state == 'run':
                    self.run_sub_expert()
                elif self.state == 'wait':
                    self.wait_for_sub_expert()
            else:
                time.sleep(1)