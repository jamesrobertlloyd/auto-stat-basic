"""
Basic version of automatic statistician implementing linear models and a couple of model building strategies.
This version implements some basic concurrency

Created August 2014

@authors: James Robert Lloyd (james.robert.lloyd@gmail.com)
"""

from __future__ import division

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
# import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})  # Prevents labels getting chopped - but can cause graph squashing

from sklearn.cross_validation import KFold
# from sklearn.cross_decomposition import CCA

from multiprocessing import Process
from threading import Thread
from multiprocessing import Queue as multi_q
from Queue import Queue as thread_q
from Queue import Empty as q_Empty

# from scipy import stats
# import numpy as np
from numpy import floor
from numpy.random import seed as np_rand_seed
import random
# import subprocess
import time
# import os
# import shutil
# import re
import sys
import traceback

# import util
from data import XSeqDataSet
from agent import Agent
import experts
import latex_util

#### TODO
#### - Agents should have a max lifespan at birth to reduce risk of orphans
#### - Stop using NIPS template - switch to something generic
#### - Make HTML versions of all tex output

#### FIXME
#### - XSeqDataSet will not uniqify labels when columns > 100

##############################################
#                                            #
#                  Manager                   #
#                                            #
##############################################


class Manager(Agent):
    def __init__(self, *args, **kwargs):
        super(Manager, self).__init__(*args, **kwargs)

        self.data = None
        self.expert_queue = multi_q()
        self.criticism_queue = multi_q()

        self.experts = []
        self.expert_reports = []
        self.updated = False

        self.state = 'init'

    def load_data(self, data):
        assert isinstance(data, XSeqDataSet)
        self.data = data

    def init_data_and_experts(self):
        # Write placeholder report
        self.outbox.append(dict(label='report', text='Your report is being prepared', tex=latex_util.waiting_tex()))
        # Partition data in train / test
        n_data = self.data.X.shape[0]
        random_perm = range(n_data)
        random.shuffle(random_perm)
        proportion_train = 0.5
        train_indices = random_perm[:int(floor(n_data * proportion_train))]
        test_indices  = random_perm[int(floor(n_data * proportion_train)):]
        data_sets = self.data.subsets([train_indices, test_indices])
        train_data = data_sets[0]
        test_data = data_sets[1]
        # Create folds of training data
        train_folds = KFold(train_data.X.shape[0], n_folds=5, indices=False)
        train_data.set_cv_indices(train_folds)
        # Initialise list of experts
        # self.experts = [DummyModel(number=5, max_actions=10),
        #                 DummyModel(number=7, max_actions=5)]
        self.experts = [experts.CrossValidationExpert(experts.SKLinearModel),
                        experts.CrossValidationExpert(experts.SKLassoReg)]
        # Load data into experts
        for expert in self.experts:
            expert.load_data(train_data)
        # Start experts running in separate processes and set up communication
        for expert in self.experts:
            q_to_child = multi_q()
            self.queues_to_children.append(q_to_child)
            expert.inbox_q = q_to_child
            expert.outbox_q = self.expert_queue
            p = Process(target=start_communication, args=(expert,))
            p.start()
            self.child_processes.append(p)
        # Remove local reference to experts
        del self.experts
        self.state = 'wait for experts'

    def wait_for_experts(self):
        time.sleep(0.1)
        self.updated = False
        while True:
            try:
                self.expert_reports.append(self.expert_queue.get_nowait())
                self.updated = True
            except q_Empty:
                break
        self.state = 'produce report'

    def produce_report(self):
        if self.updated:
            report = '\n'
            for expert_report in self.expert_reports:
                report += str(expert_report) + '\n'
            self.outbox.append(report)
        self.state = 'wait for experts'

    def next_action(self):
        if not self.termination_pending:
            if self.state == 'init':
                if not self.data is None:
                    self.init_data_and_experts()
            elif self.state == 'wait for experts':
                self.wait_for_experts()
            elif self.state == 'produce report':
                self.produce_report()

##############################################
#                                            #
#          Multiprocessing target            #
#                                            #
##############################################


def start_communication(agent):
    try:
        agent.communicate()
    except:
        traceback_object = sys.exc_info()[2]
        print("Thread for %s exited with '%s'" % (agent, sys.exc_info()))
        traceback.print_tb(traceback_object)

##############################################
#                                            #
#                   Main                     #
#                                            #
##############################################


def main():
    # Setup
    seed = 1
    np_rand_seed(seed)
    random.seed(seed)
    # Load data
    data = XSeqDataSet()
    data.load_from_file('../data/test-lin/simple-01.csv')
    # Setup up manager and communication
    queue_to_manager = thread_q()
    queue_to_main = thread_q()
    manager = Manager(inbox_q=queue_to_manager, outbox_q=queue_to_main)
    manager.load_data(data)
    # Start manager in new process / thread
    p = Thread(target=start_communication, args=(manager,))
    p.start()
    # Delete the local version of the manager to avoid confusion
    del manager
    # Listen to the manager until it finishes or crashes
    count = 0
    while True:
        if not p.is_alive():
            break
        while True:
            try:
                print(queue_to_main.get_nowait())
            except q_Empty:
                break
        count += 1
        if count > 100:
            queue_to_manager.put(dict(label='terminate'))
        time.sleep(0.1)
    p.join()


if __name__ == "__main__":
    main()