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

from multiprocessing import Process, Pipe, Queue
from Queue import Empty as q_Empty

from scipy import stats
import numpy as np
import random
import subprocess
import time
import os
import shutil
import re

#### TODO
#### - Write a to do list

##############################################
#                                            #
#                  Manager                   #
#                                            #
##############################################


class Manager():
    # def __init__(self, pipe):
    #     self.pipe_to_app = pipe  # Communication to application
    #     self.data = None

    # def load_data(self, data):
    #     assert isinstance(data, XSeqDataSet)
    #     self.data = data

    def run(self):
        # Write placeholder report
        pass

##############################################
#                                            #
#          Multiprocessing targets           #
#                                            #
##############################################

def

##############################################
#                                            #
#                   Main                     #
#                                            #
##############################################


def main():
    # Setup
    np.random.seed(1)
    random.seed(1)
    # Load data
    data = XDataSet()
    data.load_from_file('../data/test-lin/simple-01.csv')
    # Setup up manager and communication
    pipe_to_manager, pipe_to_main = Pipe()
    manager = Manager()
    manager.load_data(data)
    manager.run()


if __name__ == "__main__":
    main()