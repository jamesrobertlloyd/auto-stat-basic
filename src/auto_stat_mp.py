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

from multiprocessing import Process
from threading import Thread
from multiprocessing import Queue as multi_q
from Queue import Queue as thread_q
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
#                  Agent                     #
#                                            #
##############################################


class Agent(object):
    def __init__(self, inbox_q=None, outbox_q=None, communication_sleep=1, child_timeout=60):
        """
        Implements a basic communication and action loop
         - Get incoming messages
         - Perform next action
         - Send outgoing messages
         - Check to see if terminated
        :type inbox_q: thread_q
        :type outbox_q: thread_q
        """
        self.inbox = []
        self.outbox = []
        self.inbox_q = inbox_q
        self.outbox_q = outbox_q

        self.child_processes = []
        self.queues_to_children = []

        self.communication_sleep = communication_sleep
        self.child_timeout = child_timeout

        self.terminated = False

    def get_inbox_q(self):
        """Transfer items from inbox queue into local inbox"""
        while True:
            try:
                self.inbox.append(self.inbox_q.get_nowait())
            except q_Empty:
                break
        # TODO : This might be a good place for generic message processing e.g. pause, terminate, clear messages

    def next_action(self):
        """Inspect messages and state and perform next action checking if process stopped or paused"""
        pass

    def flush_outbox(self):
        """Send all pending messages to parent if communication queue exists"""
        if not self.outbox_q is None:
            while len(self.outbox) > 0:
                self.outbox_q.put(self.outbox.pop(0))

    def terminate_children(self):
        # Send message to all children to terminate
        for q in self.queues_to_children:
            q.put(dict(label='terminate'))
        # Attempt to join all child processes but always terminate them - terminate fast if possible
        timeout = min(1, self.child_timeout)
        while timeout <= self.child_timeout:
            for p in self.child_processes:
                p.join(timeout=timeout)
                if hasattr(p, 'terminate'):
                    p.terminate()
            timeout *= 2

    def tidy_up(self):
        """Run anything pertinent before termination"""
        self.terminate_children()

    def clear_inbox(self):
        self.inbox = []

    @property
    def termination_pending(self):
        """Checks all messages for termination instruction"""
        result = False
        for message in self.inbox:
            try:
                if message['label'].lower() == 'terminate':
                    result = True
                    break
            except:
                pass
        return result

    def communicate(self):
        """Receive incoming messages, perform actions as appropriate and send outgoing messages"""
        while True:
            self.get_inbox_q()
            self.next_action()
            self.flush_outbox()
            if self.terminated or self.termination_pending:
                self.tidy_up()
                break
            time.sleep(self.communication_sleep)

##############################################
#                                            #
#                  Models                    #
#                                            #
##############################################


class DummyModel(Agent):
    def __init__(self, number=5, max_actions=10, *args, **kwargs):
        super(DummyModel, self).__init__(*args, **kwargs)

        self.number = number
        self.action_count = 0
        self.max_actions = max_actions

    def next_action(self):
        time.sleep(self.number)
        self.action_count += 1
        self.outbox.append(str(self.number * self.action_count))
        if self.action_count >= self.max_actions:
            self.terminated = True

##############################################
#                                            #
#                  Manager                   #
#                                            #
##############################################


class Manager(Agent):
    def __init__(self, *args, **kwargs):
        super(Manager, self).__init__(*args, **kwargs)

        self.data = None
        self.model_queue = multi_q()
        self.criticism_queue = multi_q()

        self.experts = []
        self.expert_reports = []
        self.updated = False

        self.state = 'init'

    def load_data(self, data):
        # assert isinstance(data, XSeqDataSet)
        self.data = data

    def init_data_and_experts(self):
        # Write placeholder report
        self.outbox.append(dict(label='report', text='Your report is being prepared'))
        # Partition data in train / test
        # random_perm = range(len(self.data.y))
        # random.shuffle(random_perm)
        # proportion_train = 0.5
        # train_indices = random_perm[:int(np.floor(len(self.data.y) * proportion_train))]
        # test_indices  = random_perm[int(np.floor(len(self.data.y) * proportion_train)):]
        # data_sets = self.data.subsets([train_indices, test_indices])
        # train_data = data_sets[0]
        # test_data = data_sets[1]
        # # Create folds of training data
        # train_folds = KFold(len(train_data.y), n_folds=5, indices=False)
        # train_data.set_cv_indices(train_folds)
        # Initialise list of experts
        self.experts = [DummyModel(number=5, max_actions=10),
                        DummyModel(number=7, max_actions=5)]
        # Load data into experts
        # for expert in experts:
        #     expert.load_data(train_data)
        # Start experts running in separate processes and set up communication
        for expert in self.experts:
            q_to_child = multi_q()
            self.queues_to_children.append(q_to_child)
            expert.inbox_q = q_to_child
            expert.outbox_q = self.model_queue
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
                self.expert_reports.append(self.model_queue.get_nowait())
                self.updated = True
            except q_Empty:
                break
        self.state = 'produce report'

    def produce_report(self):
        if self.updated:
            report = '\n'
            for expert_report in self.expert_reports:
                report += expert_report + '\n'
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
    data = 0
    # data = XSeqDataSet()
    # data.load_from_file('../data/test-lin/simple-01.csv')
    # Setup up manager and communication
    queue_to_manager = multi_q()
    queue_to_main = multi_q()
    manager = Manager(inbox_q=queue_to_manager, outbox_q=queue_to_main)
    manager.load_data(data)
    # Start manager in new process
    p = Process(target=start_communication, args=(manager,))
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
        if count > 400:
            queue_to_manager.put(dict(label='terminate'))
        time.sleep(0.1)
    p.join()


if __name__ == "__main__":
    main()