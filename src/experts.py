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
from sklearn.mixture import GMM
from sklearn.cross_validation import KFold
from scipy import stats

# from multiprocessing import Process
from threading import Thread
from multiprocessing import Queue as multi_q
from Queue import Empty as q_Empty

import time

from agent import Agent, start_communication
from data import XSeqDataSet, XYDataSet
import util

##############################################
#                                            #
#              Distributions                 #
#                                            #
##############################################


class Independent1dGaussians(object):
    # TODO - This should derive from 1d objects and an appropriate DAG
    """Independent Gaussians"""

    def __init__(self, means, stds):
        self.means = means
        self.stds = stds

    def conditional_sample(self, data):
        assert isinstance(data, XSeqDataSet)
        return np.tile(self.means, (data.arrays['X'].shape[0], 1)) + \
               np.tile(self.stds, (data.arrays['X'].shape[0], 1)) * \
               np.random.randn(*data.arrays['X'].shape)

    def llh(self, data):
        llh = 0
        for j in range(data.arrays['X'].shape[1]):
            llh += np.sum(stats.norm.logpdf(data.arrays['X'][:, j], loc=self.means[j], scale=self.stds[j]))
        return llh


class Independent1dUniforms(object):
    # TODO - This should derive from 1d objects and an appropriate DAG
    """Independent uniforms"""

    def __init__(self, lefts, rights):
        self.lefts = lefts
        self.rights = rights

    def conditional_sample(self, data):
        assert isinstance(data, XSeqDataSet)
        return np.tile(self.lefts, (data.arrays['X'].shape[0], 1)) + \
               np.tile(self.rights - self.lefts, (data.arrays['X'].shape[0], 1)) * \
               np.random.random(data.arrays['X'].shape)

    def llh(self, data):
        # if np.all(np.tile(self.lefts, (data.arrays['X'].shape[0], 1)) <= data.arrays['X']) and \
        #    np.all(np.tile(self.rights, (data.arrays['X'].shape[0], 1)) >= data.arrays['X']):
        #     return - data.arrays['X'].shape[0] * np.sum(np.log(self.rights - self.lefts))
        # else:
        #     return -np.Inf
        llh = 0
        for j in range(data.arrays['X'].shape[1]):
            llh += np.sum(stats.uniform(loc=self.lefts[j],
                                        scale=self.rights[j] - self.lefts[j]).logpdf(data.arrays['X'][:, j]))
        return llh


class MoG(object):
    """Mixture of Gaussians"""

    def __init__(self, weights, means, stds, sklearn_mog=None):
        # print weights.size
        self.weights = weights
        self.means = means
        self.stds = stds
        self.sklearn_mog = sklearn_mog

    def conditional_sample(self, data):
        assert isinstance(data, XSeqDataSet)
        sample = np.zeros(data.arrays['X'].shape)
        for i in range(sample.shape[0]):
            if self.weights.size == 1:
                cluster = 0
            else:
                cluster = np.random.randint(low=0, high=self.weights.size-1)
            sample[i] = self.means[cluster] + np.random.randn(1, data.arrays['X'].shape[1]) * self.stds[cluster]
        return sample

    def llh(self, data):
        return np.sum(self.sklearn_mog.score(data.arrays['X']))


class SKLearnModelPlusGaussian(object):
    """Conditional distribution based on sklearn model with iid Gaussian noise"""

    def __init__(self, model, sd):
        self.model = model
        self.sd = sd

    def conditional_mean(self, data):
        return self.model.predict(data.arrays['X']).ravel()

    def conditional_sample(self, data):
        return (self.conditional_mean(data) + (self.sd * np.random.randn(data.X.shape[0], 1)).ravel()).ravel()

    def llh(self, data):
        data_minus_mean = data.arrays['Y'].ravel() - self.conditional_mean(data)
        llh = np.sum(stats.norm.logpdf(data_minus_mean, loc=0, scale=self.sd))
        return llh


class RegressionDAG(object):
    """Output caused by inputs"""

    def __init__(self, output_index, input_distribution, output_distribution):
        # print weights.size
        self.output_index = output_index
        self.input_distribution = input_distribution
        self.output_distribution = output_distribution

    def conditional_sample(self, data):
        assert isinstance(data, XSeqDataSet)
        sample = np.zeros(data.arrays['X'].shape)

        input_indices = list(range(self.output_index)) + \
                        list(range(self.output_index + 1, data.arrays['X'].shape[1], 1))

        inputs = data.variable_subsets(input_indices)
        outputs = data.variable_subsets([self.output_index])

        X_data = inputs

        XY_data = XYDataSet()
        XY_data.labels['X'] = inputs.labels['X']
        XY_data.labels['Y'] = outputs.labels['X']
        XY_data.arrays['X'] = inputs.arrays['X']
        XY_data.arrays['Y'] = outputs.arrays['X']

        X_sample = self.input_distribution.conditional_sample(X_data)
        XY_data.arrays['X'] = X_sample
        Y_sample = self.output_distribution.conditional_samples(XY_data)

        sample[:, input_indices] = X_sample
        sample[:, self.output_index] = Y_sample
        return sample

    def llh(self, data):
        assert isinstance(data, XSeqDataSet)
        sample = np.zeros(data.arrays['X'].shape)

        input_indices = list(range(self.output_index)) + \
                        list(range(self.output_index + 1, data.arrays['X'].shape[1], 1))

        inputs = data.variable_subsets(input_indices)
        outputs = data.variable_subsets([self.output_index])

        X_data = inputs

        XY_data = XYDataSet()
        XY_data.labels['X'] = inputs.labels['X']
        XY_data.labels['Y'] = outputs.labels['X']
        XY_data.arrays['X'] = inputs.arrays['X']
        XY_data.arrays['Y'] = outputs.arrays['X']

        llh = self.input_distribution.llh(X_data)
        llh += self.output_distribution.llh(XY_data)
        return llh

##############################################
#                                            #
#                 Scorers                    #
#                                            #
##############################################


class ZeroScorer(object):
    """Dummy class for scoring stuff"""
    def __init__(self):
        pass

    def score(self, data, distribution):
        return 0


class MMDScorer(object):
    """Maximum mean discrepancy of two distributions estimated from samples"""
    def __init__(self, lengthscales):
        self.lengthscales = lengthscales

    def score(self, data, distribution):
        X = data.arrays['X']
        Y = distribution.conditional_sample(data)  # In the future only manipulated inputs will be given
        return util.MMD(X, Y, self.lengthscales)


class LLHScorer(object):
    """Pointwise log likelihood"""
    def __init__(self):
        pass

    @staticmethod
    def score(data, distribution):
        # Converted to per data point llh
        return distribution.llh(data) / data.arrays['X'].shape[0]

##############################################
#                                            #
#                  Models                    #
#                                            #
##############################################


# TODO - is this used anywhere?
class DistributionModel(object):
    """Wrapper for a distribution"""
    def __init__(self, dist):
        self.conditional_distributions = [dist]


class IndependentGaussianLearner(Agent):
    """
    Fits independent Gaussians to data
    """

    def __init__(self, *args, **kwargs):
        super(IndependentGaussianLearner, self).__init__(*args, **kwargs)

        self.data = None
        self.conditional_distributions = []

    def load_data(self, data):
        assert isinstance(data, XSeqDataSet)
        self.data = data

    def fit(self):
        means = np.mean(self.data.arrays['X'], 0)
        stds = np.std(self.data.arrays['X'], 0)
        # TODO - This should be composed of 1d Gaussians and a DAG
        self.conditional_distributions = [Independent1dGaussians(means=means, stds=stds)]

    @staticmethod
    def generate_descriptions():
        # I can be replaced with a generic description routine - e.g. average predictive comparisons
        raise RuntimeError('Description not implemented')


class IndependentUniformLearner(Agent):
    """
    Fits independent uniforms to data
    """

    def __init__(self, *args, **kwargs):
        super(IndependentUniformLearner, self).__init__(*args, **kwargs)

        self.data = None
        self.conditional_distributions = []

    def load_data(self, data):
        assert isinstance(data, XSeqDataSet)
        self.data = data

    def fit(self):
        # TODO - this is not a good way to learn uniforms
        lefts = np.min(self.data.arrays['X'], 0)
        rights = np.max(self.data.arrays['X'], 0)
        widths = rights - lefts
        lefts = lefts - 0.1  * widths
        rights = rights + 0.1 * widths
        # TODO - This should be composed of 1d uniforms and a DAG
        self.conditional_distributions = [Independent1dUniforms(lefts=lefts, rights=rights)]

    @staticmethod
    def generate_descriptions():
        # I can be replaced with a generic description routine - e.g. average predictive comparisons
        raise RuntimeError('Description not implemented')


class MoGLearner(Agent):
    """
    Fits mixture of Gaussians to data
    """

    def __init__(self, *args, **kwargs):
        super(MoGLearner, self).__init__(*args, **kwargs)

        self.data = None
        self.conditional_distributions = []

    def load_data(self, data):
        assert isinstance(data, XSeqDataSet)
        self.data = data

    def fit(self):
        best_model = None
        best_BIC = None
        for n_components in range(1, 11, 1):
            sk_learner = GMM(n_components=n_components, n_iter=250, n_init=5)
            sk_learner.fit(self.data.arrays['X'])
            BIC = sk_learner.bic(self.data.arrays['X'])
            # print BIC
            if (best_BIC is None) or (best_BIC > BIC):
                best_BIC = BIC
                best_model = sk_learner
        weights = best_model.weights_
        means = best_model.means_
        # stds = np.sqrt(1 / best_model.precs_)
        stds = np.sqrt(best_model.covars_)
        self.conditional_distributions = [MoG(weights=weights, means=means, stds=stds, sklearn_mog=best_model)]

    @staticmethod
    def generate_descriptions():
        # I can be replaced with a generic description routine - e.g. average predictive comparisons
        raise RuntimeError('Description not implemented')


class SKLearnModelLearner(Agent):
    """Wrapper for sklearn regression models"""

    def __init__(self, base_class, *args, **kwargs):
        super(SKLearnModelLearner, self).__init__(*args, **kwargs)
        self.sklearn_class = base_class
        self.model = self.sklearn_class()
        self.data = None
        self.conditional_distributions = []
        self.knowledge_base = []

    def load_data(self, data):
        assert isinstance(data, XYDataSet)
        self.data = data

    def fit(self):
        self.model.fit(self.data.arrays['X'], self.data.arrays['Y'].ravel())
        y_hat = self.model.predict(self.data.arrays['X'])
        sd = np.sqrt((sklearn.metrics.mean_squared_error(self.data.arrays['Y'], y_hat)))
        self.conditional_distributions = [SKLearnModelPlusGaussian(self.model, sd)]

    @staticmethod
    def generate_descriptions(self):
        # I can be replaced with a generic description routine - e.g. average predictive comparisons
        raise RuntimeError('Description not implemented')


class SKLinearModel(SKLearnModelLearner):
    """Simple linear regression model based on sklearn implementation"""

    def __init__(self):
        super(SKLinearModel, self).__init__(lambda: sklearn.linear_model.LinearRegression(fit_intercept=True,
                                                                                          normalize=False,
                                                                                          copy_X=True))


class SKLASSO(SKLearnModelLearner):
    """Simple linear regression model based on sklearn implementation"""

    def __init__(self):
        super(SKLASSO, self).__init__(lambda: sklearn.ensemble.RandomForestRegressor(n_estimators=100))

class RegressionLearner(Agent):
    """
    Fits something to explain all inputs then explains one of the variables as conditional on the rest
    Let's hope that this becomes an example of a generic DAG learning unit
    """

    def __init__(self, input_learner, output_learner, *args, **kwargs):
        super(RegressionLearner, self).__init__(*args, **kwargs)

        self.data = None
        self.conditional_distributions = []

        self.input_learner = input_learner
        self.output_learner = output_learner

    def load_data(self, data):
        assert isinstance(data, XSeqDataSet)
        self.data = data

    def fit(self):
        best_llh = -np.Inf
        best_distribution = None
        # best_index = None

        for input_var in range(self.data.arrays['X'].shape[1]):
            inputs = self.data.variable_subsets(list(range(input_var)) +
                                                list(range(input_var + 1, self.data.arrays['X'].shape[1], 1)))
            outputs = self.data.variable_subsets([input_var])
            input_agent = self.input_learner()

            input_agent.load_data(inputs)
            input_agent.fit()

            XY_data = XYDataSet()
            XY_data.labels['X'] = inputs.labels['X']
            XY_data.labels['Y'] = outputs.labels['X']
            XY_data.arrays['X'] = inputs.arrays['X']
            XY_data.arrays['Y'] = outputs.arrays['X']

            output_agent = self.output_learner()
            output_agent.load_data(XY_data)
            output_agent.fit()

            conditional_distribution = RegressionDAG(input_var, input_agent.conditional_distributions[0],
                                                                output_agent.conditional_distributions[0])

            score = LLHScorer.score(self.data, conditional_distribution)
            if score > best_llh:
                best_llh = score
                best_distribution = conditional_distribution
                # best_index = input_var

        # print best_index

        self.conditional_distributions = [best_distribution]

    @staticmethod
    def generate_descriptions():
        # I can be replaced with a generic description routine - e.g. average predictive comparisons
        raise RuntimeError('Description not implemented')


##############################################
#                                            #
#               Meta experts                 #
#                                            #
##############################################


class SamplesCrossValidationExpert(Agent):
    """
    Takes an expert as input, assumes the expert learns some number of (conditional) distributions, cross validates the
    performance of these distributions by asking them to produce samples and then passing them to some scoring function,
    returns the distributions and cross validation scores to its parent
    """
    def __init__(self, sub_expert_class, scoring_expert, n_folds=5, *args, **kwargs):
        super(SamplesCrossValidationExpert, self).__init__(*args, **kwargs)

        self.sub_expert_class = sub_expert_class
        self.sub_expert = None
        self.data = None
        self.scoring_expert = scoring_expert
        self.conditional_distributions = []

        self.n_folds = n_folds

    def load_data(self, data):
        assert isinstance(data, XSeqDataSet)
        self.data = data

    def cross_validate(self):
        # Set up cross validation scheme
        train_folds = KFold(self.data.arrays['X'].shape[0], n_folds=self.n_folds, indices=False)
        self.data.set_cv_indices(train_folds)
        # Calculate cross validated scores
        scores = None
        fold_count = 0

        for (fold, (train_data, test_data)) in enumerate(self.data.cv_subsets):
            # print('CV')
            if self.termination_pending:
                # print('Received termination call during CV')
                break
            temp_expert = self.sub_expert_class()
            temp_expert.load_data(train_data)
            temp_expert.fit()
            distributions = temp_expert.conditional_distributions

            fold_count += 1
            if scores is None:
                scores = np.zeros((self.n_folds, len(distributions)))
            for (i, distribution) in enumerate(distributions):
                scores[fold, i] = self.scoring_expert.score(data=test_data, distribution=distribution)

        if not self.termination_pending:

            # Train on full data
            self.sub_expert = self.sub_expert_class()
            self.sub_expert.load_data(self.data)
            self.sub_expert.fit()
            distributions = self.sub_expert.conditional_distributions
            # Report results of cross validation
            for (i, distribution) in enumerate(distributions):
                self.outbox.append(dict(label='CV-samples', distribution=distribution, scores=scores[:, i],
                                        scoring_expert=self.scoring_expert, data=self.data))

    def next_action(self):
        if not self.termination_pending:
            if not self.data is None:
                self.cross_validate()
                self.terminated = True
            else:
                time.sleep(1)
        # if self.termination_pending or self.terminated:
        #     print 'Cross validater will terminate'


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
            if self.data_size >= self.data.arrays['X'].shape[0]:
                self.data_size = self.data.arrays['X'].shape[0]
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
        # if self.termination_pending or self.terminated:
        #     print 'Doubler will terminate'