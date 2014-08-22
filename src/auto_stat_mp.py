"""
Basic version of automatic statistician implementing linear models and a couple of model building strategies.
This version implements some basic concurrency

Created August 2014

@authors: James Robert Lloyd (james.robert.lloyd@gmail.com)
"""

from __future__ import division

import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True}) # Prevents labels getting chopped - but can cause graph squashing

import sklearn
import sklearn.linear_model
import sklearn.ensemble
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.cross_decomposition import CCA

from multiprocessing import Process, Pipe
from multiprocessing import Queue as multi_q
from Queue import Empty as q_Empty

from scipy import stats
import numpy as np
import random
import subprocess
import time
import os
import shutil
import re
import sys

#### TODO
#### - Write a to do list

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
        self.knowledge_base.append(dict(label='method', text='Full linear model',
                                        distribution=self.conditional_distributions[0], data=self.data))
        self.knowledge_base.append(dict(label='active-inputs', value=self.data.X.shape[1],
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
        fig.savefig(os.path.join(self.data.path, 'report', 'figures', "lin-train-fit.pdf"))
        plt.close()
        # Plot data against all dimensions
        for dim in range(self.data.X.shape[1]):
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(1,1,1) # one row, one column, first plot
            ax.scatter(self.data.X[:, dim], self.data.y, color="red", marker="o")
            ax.set_title("Training data against %s" % self.data.X_labels[dim])
            ax.set_xlabel(self.data.X_labels[dim])
            ax.set_ylabel("Training data")
            fig.savefig(os.path.join(self.data.path, 'report', 'figures', "lin-train-%s.pdf" % self.data.X_labels[dim].replace(' ', '')))
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
            fig.savefig(os.path.join(self.data.path, 'report', 'figures', "lin-partial-resid-%s.pdf" % self.data.X_labels[dim].replace(' ', '')))
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
            fig.savefig(os.path.join(self.data.path, 'report', 'figures', "lin-resid-%s.pdf" % self.data.X_labels[dim].replace(' ', '')))
            plt.close()
        # FIXME - this is in the wrong place
        self.generate_tex()

    def generate_tex(self):
        tex_summary, tex_full = lin_mod_tex_description(coef=self.model.coef_, data=self.data,
                                                y_hat=self.conditional_distributions[0].conditional_mean(self.data))
        self.knowledge_base.append(dict(label='tex-summary', text=tex_summary,
                                        distribution=self.conditional_distributions[0], data=self.data))
        self.knowledge_base.append(dict(label='tex-description', text=tex_full,
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
        self.knowledge_base.append(dict(label='method', text='LASSO',
                                        distribution=self.conditional_distributions[0], data=self.data))
        self.knowledge_base.append(dict(label='active-inputs', value=np.sum(self.model.coef_ != 0),
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
        fig.savefig(os.path.join(self.data.path, 'report', 'figures', "lasso-train-fit.pdf"))
        plt.close()
        # Plot data against all dimensions
        for dim in range(self.data.X.shape[1]):
            fig = plt.figure(figsize=(5, 4))
            ax = fig.add_subplot(1,1,1) # one row, one column, first plot
            ax.scatter(self.data.X[:, dim], self.data.y, color="red", marker="o")
            ax.set_title("Training data against %s" % self.data.X_labels[dim])
            ax.set_xlabel(self.data.X_labels[dim])
            ax.set_ylabel("Training data")
            fig.savefig(os.path.join(self.data.path, 'report', 'figures', "lasso-train-%s.pdf" % self.data.X_labels[dim].replace(' ', '')))
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
            fig.savefig(os.path.join(self.data.path, 'report', 'figures', "lasso-partial-resid-%s.pdf" % self.data.X_labels[dim].replace(' ', '')))
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
            fig.savefig(os.path.join(self.data.path, 'report', 'figures', "lasso-resid-%s.pdf" % self.data.X_labels[dim].replace(' ', '')))
            plt.close()
        # FIXME - this is in the wrong place
        self.generate_tex()

    def generate_tex(self):
        tex_summary, tex_full = lin_mod_tex_description(coef=self.model.coef_, data=self.data, id='lasso',
                                                y_hat=self.conditional_distributions[0].conditional_mean(self.data))
        self.knowledge_base.append(dict(label='tex-summary', text=tex_summary,
                                        distribution=self.conditional_distributions[0], data=self.data))
        self.knowledge_base.append(dict(label='tex-description', text=tex_full,
                                        distribution=self.conditional_distributions[0], data=self.data))

##############################################
#                                            #
#                  Manager                   #
#                                            #
##############################################


class Manager():
    def __init__(self, pipe):
        self.pipe_to_parent = pipe  # Communication to application
        self.data = None
        self.expert_processes = []
        self.expert_pipes = []

    def load_data(self, data):
        # assert isinstance(data, XSeqDataSet)
        self.data = data

    def stop_children(self):
        # Send message to all children
        for pipe in self.expert_pipes:
            pipe.send('stop')
        # Attempt to join all child processes, else terminate them

    def init_data(self):
        

    def run(self):
        # TODO - this should be a minimal interface that just turns messages from the parent into actions...
        # TODO - ...this will allow us to test the module in parallel / single threaded environments
        # Write placeholder report
        self.pipe_to_parent.send('Your report is being prepared')
        # Partition data in train / test
        random_perm = range(len(self.data.y))
        random.shuffle(random_perm)
        proportion_train = 0.5
        train_indices = random_perm[:int(np.floor(len(self.data.y) * proportion_train))]
        test_indices  = random_perm[int(np.floor(len(self.data.y) * proportion_train)):]
        data_sets = self.data.subsets([train_indices, test_indices])
        train_data = data_sets[0]
        test_data = data_sets[1]
        # Create folds of training data
        train_folds = KFold(len(train_data.y), n_folds=5, indices=False)
        train_data.set_cv_indices(train_folds)
        # Initialise list of experts
        # experts = [CrossValidationExpert(SKLinearModel),
        #            CrossValidationExpert(SKLassoReg),
        #            CrossValidationExpert(BICBackwardsStepwiseLin)]#,
        #            # CrossValidationExpert(SKLearnRandomForestReg)]
        experts = [SKLinearModel(),
                   SKLassoReg()]
        # Load data into experts
        for expert in experts:
            expert.load_data(train_data)
        # Start experts running in separate processes and set up communication
        self.expert_processes = []
        self.expert_pipes = []
        for expert in experts:
            pipe_to_expert, pipe_to_manager = Pipe()
            self.expert_pipes.append(pipe_to_expert)
            expert.pipe_to_parent = pipe_to_manager
            p = Process(target=run_agent, args=(expert,))
            p.start()
            self.expert_processes.append(p)
        # Remove local reference to experts
        del experts
        # Start communication loop
        while True:
            # Any messages from parent?
            if self.pipe_to_parent.poll():
                message = self.pipe_to_parent.recv()
                if message == 'stop':

                    break


        self.pipe_to_parent.send('A message')
        self.pipe_to_parent.send('finished')

##############################################
#                                            #
#          Multiprocessing target            #
#                                            #
##############################################


def run_agent(agent):
    try:
        agent.run()
    except:
        print "Thread for %s exited with '%s'" % (agent, sys.exc_info())

##############################################
#                                            #
#                   Main                     #
#                                            #
##############################################


def main():
    # Setup
    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    # Load data
    data = None
    # data = XSeqDataSet()
    # data.load_from_file('../data/test-lin/simple-01.csv')
    # Setup up manager and communication
    pipe_to_manager, pipe_to_main = Pipe()
    manager = Manager(pipe=pipe_to_main)
    manager.load_data(data)
    # Start manager in new process
    p = Process(target=run_agent, args=(manager,))
    p.start()
    # Delete the local version of the manager to avoid confusion
    del manager
    # Listen to the manager until it finishes or crashes
    while True:
        if pipe_to_manager.poll():
            message = pipe_to_manager.recv()
            print(message)
            if message == 'finished':
                break
        elif not p.is_alive():
            print('Manager has crashed')
            break
        time.sleep(1)


if __name__ == "__main__":
    main()