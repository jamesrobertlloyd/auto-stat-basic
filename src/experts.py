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

import time

from agent import Agent
from data import XYDataSet, XSeqDataSet
import util

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
        # self.generate_descriptions()

    @staticmethod
    def generate_descriptions():
        # I can be replaced with a generic description routine - e.g. average predictive comparisons
        raise RuntimeError('Description not implemented')


class SKLinearModel(SKLearnModel):
    """Simple linear regression model based on sklearn implementation"""

    def __init__(self):
        super(SKLinearModel, self).__init__(lambda: sklearn.linear_model.LinearRegression(fit_intercept=True,
                                                                                          normalize=False,
                                                                                          copy_X=True))

    # def generate_descriptions(self):
    #     summary, description = lin_mod_txt_description(coef=self.model.coef_, data=self.data)
    #     self.knowledge_base.append(dict(label='summary', text=summary,
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #     self.knowledge_base.append(dict(label='description', text=description,
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #     self.knowledge_base.append(dict(label='method', text='Full linear model',
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #     self.knowledge_base.append(dict(label='active-inputs', value=self.data.X.shape[1],
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #
    # def generate_figures(self):
    #     # Plot training data against fit
    #     fig = plt.figure(figsize=(5, 4))
    #     ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    #     y_hat = self.conditional_distributions[0].conditional_mean(self.data)
    #     sorted_y_hat = np.sort(y_hat)
    #     ax.plot(sorted_y_hat, sorted_y_hat, color="blue")
    #     ax.scatter(y_hat, self.data.y, color="red", marker="o")
    #     ax.set_title("Training data against fit")
    #     ax.set_xlabel("Model fit")
    #     ax.set_ylabel("Training data")
    #     fig.savefig(os.path.join(self.data.path, 'report', 'figures', "lin-train-fit.pdf"))
    #     plt.close()
    #     # Plot data against all dimensions
    #     for dim in range(self.data.X.shape[1]):
    #         fig = plt.figure(figsize=(5, 4))
    #         ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    #         ax.scatter(self.data.X[:, dim], self.data.y, color="red", marker="o")
    #         ax.set_title("Training data against %s" % self.data.X_labels[dim])
    #         ax.set_xlabel(self.data.X_labels[dim])
    #         ax.set_ylabel("Training data")
    #         fig.savefig(os.path.join(self.data.path, 'report', 'figures', "lin-train-%s.pdf" % self.data.X_labels[dim].replace(' ', '')))
    #         plt.close()
    #     # Plot rest of model against fit
    #     for dim in range(self.data.X.shape[1]):
    #         fig = plt.figure(figsize=(5, 4))
    #         ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    #         y_hat = self.conditional_distributions[0].conditional_mean(self.data)
    #         component_fit = self.model.coef_[dim] * self.data.X[:, dim].ravel()
    #         partial_resid = self.data.y - (y_hat - component_fit)
    #         plot_idx = np.argsort(self.data.X[:, dim].ravel())
    #         ax.plot(self.data.X[plot_idx, dim], component_fit[plot_idx], color="blue")
    #         ax.scatter(self.data.X[:, dim], partial_resid, color="red", marker="o")
    #         ax.set_title("Partial residual against %s" % self.data.X_labels[dim])
    #         ax.set_xlabel(self.data.X_labels[dim])
    #         ax.set_ylabel("Partial residual")
    #         fig.savefig(os.path.join(self.data.path, 'report', 'figures', "lin-partial-resid-%s.pdf" % self.data.X_labels[dim].replace(' ', '')))
    #         plt.close()
    #     # Plot residuals against each dimension
    #     for dim in range(self.data.X.shape[1]):
    #         fig = plt.figure(figsize=(5, 4))
    #         ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    #         y_hat = self.conditional_distributions[0].conditional_mean(self.data)
    #         resid = self.data.y - y_hat
    #         ax.scatter(self.data.X[:, dim], resid, color="red", marker="o")
    #         ax.set_title("Residuals against %s" % self.data.X_labels[dim])
    #         ax.set_xlabel(self.data.X_labels[dim])
    #         ax.set_ylabel("Residuals")
    #         fig.savefig(os.path.join(self.data.path, 'report', 'figures', "lin-resid-%s.pdf" % self.data.X_labels[dim].replace(' ', '')))
    #         plt.close()
    #     # FIXME - this is in the wrong place
    #     self.generate_tex()
    #
    # def generate_tex(self):
    #     tex_summary, tex_full = lin_mod_tex_description(coef=self.model.coef_, data=self.data,
    #                                             y_hat=self.conditional_distributions[0].conditional_mean(self.data))
    #     self.knowledge_base.append(dict(label='tex-summary', text=tex_summary,
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #     self.knowledge_base.append(dict(label='tex-description', text=tex_full,
    #                                     distribution=self.conditional_distributions[0], data=self.data))


class SKLassoReg(SKLearnModel):
    """Lasso trained linear regression model"""

    def __init__(self):
        super(SKLassoReg, self).__init__(sklearn.linear_model.LassoLarsCV)

    # def generate_descriptions(self):
    #     summary, description = lin_mod_txt_description(coef=self.model.coef_, data=self.data)
    #     self.knowledge_base.append(dict(label='summary', text=summary,
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #     self.knowledge_base.append(dict(label='description', text=description,
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #     self.knowledge_base.append(dict(label='method', text='LASSO',
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #     self.knowledge_base.append(dict(label='active-inputs', value=np.sum(self.model.coef_ != 0),
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #
    # def generate_figures(self):
    #     # Plot training data against fit
    #     fig = plt.figure(figsize=(5, 4))
    #     ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    #     y_hat = self.conditional_distributions[0].conditional_mean(self.data)
    #     sorted_y_hat = np.sort(y_hat)
    #     ax.plot(sorted_y_hat, sorted_y_hat, color="blue")
    #     ax.scatter(y_hat, self.data.y, color="red", marker="o")
    #     ax.set_title("Training data against fit")
    #     ax.set_xlabel("Model fit")
    #     ax.set_ylabel("Training data")
    #     fig.savefig(os.path.join(self.data.path, 'report', 'figures', "lasso-train-fit.pdf"))
    #     plt.close()
    #     # Plot data against all dimensions
    #     for dim in range(self.data.X.shape[1]):
    #         fig = plt.figure(figsize=(5, 4))
    #         ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    #         ax.scatter(self.data.X[:, dim], self.data.y, color="red", marker="o")
    #         ax.set_title("Training data against %s" % self.data.X_labels[dim])
    #         ax.set_xlabel(self.data.X_labels[dim])
    #         ax.set_ylabel("Training data")
    #         fig.savefig(os.path.join(self.data.path, 'report', 'figures', "lasso-train-%s.pdf" % self.data.X_labels[dim].replace(' ', '')))
    #         plt.close()
    #     # Plot rest of model against fit
    #     for dim in range(self.data.X.shape[1]):
    #         fig = plt.figure(figsize=(5, 4))
    #         ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    #         y_hat = self.conditional_distributions[0].conditional_mean(self.data)
    #         component_fit = self.model.coef_[dim] * self.data.X[:, dim].ravel()
    #         partial_resid = self.data.y - (y_hat - component_fit)
    #         plot_idx = np.argsort(self.data.X[:, dim].ravel())
    #         ax.plot(self.data.X[plot_idx, dim], component_fit[plot_idx], color="blue")
    #         ax.scatter(self.data.X[:, dim], partial_resid, color="red", marker="o")
    #         ax.set_title("Partial residual against %s" % self.data.X_labels[dim])
    #         ax.set_xlabel(self.data.X_labels[dim])
    #         ax.set_ylabel("Partial residual")
    #         fig.savefig(os.path.join(self.data.path, 'report', 'figures', "lasso-partial-resid-%s.pdf" % self.data.X_labels[dim].replace(' ', '')))
    #         plt.close()
    #     # Plot residuals against each dimension
    #     for dim in range(self.data.X.shape[1]):
    #         fig = plt.figure(figsize=(5, 4))
    #         ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    #         y_hat = self.conditional_distributions[0].conditional_mean(self.data)
    #         resid = self.data.y - y_hat
    #         ax.scatter(self.data.X[:, dim], resid, color="red", marker="o")
    #         ax.set_title("Residuals against %s" % self.data.X_labels[dim])
    #         ax.set_xlabel(self.data.X_labels[dim])
    #         ax.set_ylabel("Residuals")
    #         fig.savefig(os.path.join(self.data.path, 'report', 'figures', "lasso-resid-%s.pdf" % self.data.X_labels[dim].replace(' ', '')))
    #         plt.close()
    #     # FIXME - this is in the wrong place
    #     self.generate_tex()
    #
    # def generate_tex(self):
    #     tex_summary, tex_full = lin_mod_tex_description(coef=self.model.coef_, data=self.data, id='lasso',
    #                                             y_hat=self.conditional_distributions[0].conditional_mean(self.data))
    #     self.knowledge_base.append(dict(label='tex-summary', text=tex_summary,
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #     self.knowledge_base.append(dict(label='tex-description', text=tex_full,
    #                                     distribution=self.conditional_distributions[0], data=self.data))


class SKLearnRandomForestReg(SKLearnModel):

    def __init__(self):
        super(SKLearnRandomForestReg, self).__init__(lambda: sklearn.ensemble.RandomForestRegressor(n_estimators=100))

    # def generate_descriptions(self):
    #     self.knowledge_base.append(dict(label='summary', text='I am random forest',
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #     self.knowledge_base.append(dict(label='description', text='I am still random forest',
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #
    # def generate_figures(self):
    #     # Plot training data against fit
    #     fig = plt.figure(figsize=(5, 4))
    #     ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    #     y_hat = self.conditional_distributions[0].conditional_mean(self.data)
    #     sorted_y_hat = np.sort(y_hat)
    #     ax.plot(sorted_y_hat, sorted_y_hat, color="blue")
    #     ax.scatter(y_hat, self.data.y, color="red", marker="o")
    #     ax.set_title("Training data against fit")
    #     ax.set_xlabel("Model fit")
    #     ax.set_ylabel("Training data")
    #     fig.savefig(os.path.join(self.data.path, 'report', 'figures', "rf-train-fit.pdf"))
    #     plt.close()


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
        current_BIC = util.BIC(self.model.conditional_distributions[0], self.data, len(self.model.model.coef_))
        improvement = True
        best_model = None  # Removing warning message
        best_subset = None  # Removing warning message
        while improvement and (len(self.subset) > 0):
            improvement = False
            best_BIC = current_BIC
            # Try removing all input variables
            for i in range(len(self.subset)):
                temp_subset = self.subset[:i] + self.subset[(i+1):]
                if len(temp_subset) > 0:
                    temp_data_set = self.data.input_subset(temp_subset)
                    temp_model = SKLinearModel()
                    temp_model.load_data(temp_data_set)
                    temp_model.run()
                    temp_BIC = util.BIC(temp_model.conditional_distributions[0], temp_data_set, len(temp_model.model.coef_))
                    if temp_BIC < best_BIC:
                        best_model = temp_model
                        best_subset = temp_subset
                        best_BIC = temp_BIC
                else:
                    temp_dist = MeanPlusGaussian(mean=self.data.y.mean(), sd=0)
                    temp_BIC = util.BIC(temp_dist, self.data, 0)
                    if temp_BIC < best_BIC:
                        best_model = DistributionModel(temp_dist)
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
        if len(self.subset) > 0:
            self.conditional_distributions = [SKLearnModelInputFilteredPlusGaussian(self.model.model, sd, self.subset)]
        else:
            self.conditional_distributions = [self.model.conditional_distributions[0]]
            self.conditional_distributions[0].sd = sd
        # self.generate_descriptions()

    # def generate_descriptions(self):
    #     if len(self.subset) > 0:
    #         summary, description = lin_mod_txt_description(coef=self.model.model.coef_,
    #                                                        data=self.data.input_subset(self.subset))
    #     else:
    #         summary, description = lin_mod_txt_description(coef=[],
    #                                                        data=self.data.input_subset(self.subset))
    #     self.knowledge_base.append(dict(label='summary', text=summary,
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #     self.knowledge_base.append(dict(label='description', text=description,
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #     self.knowledge_base.append(dict(label='method', text='BIC stepwise',
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #     self.knowledge_base.append(dict(label='active-inputs', value=len(self.subset),
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #
    # def generate_figures(self):
    #     # Plot training data against fit
    #     fig = plt.figure(figsize=(5, 4))
    #     ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    #     y_hat = self.conditional_distributions[0].conditional_mean(self.data)
    #     sorted_y_hat = np.sort(y_hat)
    #     ax.plot(sorted_y_hat, sorted_y_hat, color="blue")
    #     ax.scatter(y_hat, self.data.y, color="red", marker="o")
    #     ax.set_title("Training data against fit")
    #     ax.set_xlabel("Model fit")
    #     ax.set_ylabel("Training data")
    #     fig.savefig(os.path.join(self.data.path, 'report', 'figures', "bic-train-fit.pdf"))
    #     plt.close()
    #     # Plot data against all dimensions
    #     for dim in range(self.data.X.shape[1]):
    #         fig = plt.figure(figsize=(5, 4))
    #         ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    #         ax.scatter(self.data.X[:, dim], self.data.y, color="red", marker="o")
    #         ax.set_title("Training data against %s" % self.data.X_labels[dim])
    #         ax.set_xlabel(self.data.X_labels[dim])
    #         ax.set_ylabel("Training data")
    #         fig.savefig(os.path.join(self.data.path, 'report', 'figures', "bic-train-%s.pdf" % self.data.X_labels[dim].replace(' ', '')))
    #         plt.close()
    #     # Plot rest of model against fit
    #     dim_count = 0 # FIXME - this is hacky
    #     for dim in range(self.data.X.shape[1]):
    #         if dim in self.subset:
    #             fig = plt.figure(figsize=(5, 4))
    #             ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    #             y_hat = self.conditional_distributions[0].conditional_mean(self.data)
    #             component_fit = self.model.model.coef_[dim_count] * self.data.X[:, dim].ravel() # FIXME model.model
    #             partial_resid = self.data.y - (y_hat - component_fit)
    #             plot_idx = np.argsort(self.data.X[:, dim].ravel())
    #             ax.plot(self.data.X[plot_idx, dim], component_fit[plot_idx], color="blue")
    #             ax.scatter(self.data.X[:, dim], partial_resid, color="red", marker="o")
    #             ax.set_title("Partial residual against %s" % self.data.X_labels[dim])
    #             ax.set_xlabel(self.data.X_labels[dim])
    #             ax.set_ylabel("Partial residual")
    #             fig.savefig(os.path.join(self.data.path, 'report', 'figures', "bic-partial-resid-%s.pdf" % self.data.X_labels[dim].replace(' ', '')))
    #             plt.close()
    #             dim_count += 1
    #     # Plot residuals against each dimension
    #     for dim in range(self.data.X.shape[1]):
    #         fig = plt.figure(figsize=(5, 4))
    #         ax = fig.add_subplot(1,1,1) # one row, one column, first plot
    #         y_hat = self.conditional_distributions[0].conditional_mean(self.data)
    #         resid = self.data.y - y_hat
    #         ax.scatter(self.data.X[:, dim], resid, color="red", marker="o")
    #         ax.set_title("Residuals against %s" % self.data.X_labels[dim])
    #         ax.set_xlabel(self.data.X_labels[dim])
    #         ax.set_ylabel("Residuals")
    #         fig.savefig(os.path.join(self.data.path, 'report', 'figures', "bic-resid-%s.pdf" % self.data.X_labels[dim].replace(' ', '')))
    #         plt.close()
    #     # FIXME - this is in the wrong place
    #     self.generate_tex()
    #
    # def generate_tex(self):
    #     coefs = [0] * len(self.data.X_labels)
    #     for (i, dim) in enumerate(self.subset):
    #         coefs[dim] = self.model.model.coef_[i] # FIXME model.model
    #     tex_summary, tex_full = lin_mod_tex_description(coef=coefs, data=self.data, id='bic',
    #                                                 y_hat=self.conditional_distributions[0].conditional_mean(self.data))
    #     self.knowledge_base.append(dict(label='tex-summary', text=tex_summary,
    #                                     distribution=self.conditional_distributions[0], data=self.data))
    #     self.knowledge_base.append(dict(label='tex-description', text=tex_full,
    #                                     distribution=self.conditional_distributions[0], data=self.data))

##############################################
#                                            #
#             Cross validation               #
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
        # Calculate cross validated RMSE
        RMSE_sum = None
        var_explained_sum = None
        fold_count = 0

        for train_data, test_data in self.data.cv_subsets:
            if self.termination_pending:
                break
            temp_expert = self.sub_expert_class()
            temp_expert.load_data(train_data)
            temp_expert.run()
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
            self.sub_expert.run()
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