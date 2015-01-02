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

# from sklearn.cross_decomposition import CCA

from multiprocessing import Process
from threading import Thread
from multiprocessing import Queue as multi_q
from Queue import Empty as q_Empty

import numpy as np
from numpy.random import seed as np_rand_seed
import random
# import subprocess
import time
# import os
# import shutil
# import re
import sys
import select


# import util
from data import XSeqDataSet
from agent import Agent, start_communication
import experts
import latex_util
import make_graphs as gr

#### TODO
#### - Nonparametric density estimate, (these are nice to haves - FA, Linear DAG)
#### - And what about also including something with t - distributed errors
#### - Pretty plots - Stop using NIPS template - switch to something generic - switch to HTML?
#### - Model checks appropriate for generative models - how do I ensure I still have power?
#### - Try scoring by MMD (arc kernel required for missing data?) - might need efficient variants of MMD
#### - Agents should have a max lifespan at birth to reduce risk of orphans

#### FIXME
#### - XSeqDataSet will not uniqify labels when columns > 100
#### - Random forest causes thread problems when asked to shutdown before completion - don't know why


class Manager(Agent):
    def __init__(self, *args, **kwargs):
        super(Manager, self).__init__(*args, **kwargs)

        self.data = None
        self.test_indices = None
        self.train_indices = None
        self.expert_queue = multi_q()
        self.criticism_queue = multi_q()

        self.experts = []
        self.all_dist_msgs = []
        self.up_to_date_dist_msgs = []
        self.updated = False

        self.state = 'init'
        self.latex_dir = './pdf-template' # template for report
        self.temp_dir = None # directory for output

    def load_data(self, data):
        assert isinstance(data, XSeqDataSet)
        self.data = data

    def init_data_and_experts(self):
        # Preparing report things
        self.templatefl = "pdf_template.jinja"
        self.template = latex_util.get_latex_template(self.templatefl)
        summary = [{} for x in range(self.data.arrays['X'].shape[1])]
        for i in range(self.data.arrays['X'].shape[1]):
            summary[i]['label'] = self.data.labels['X'][i]
            summary[i]['min'] = np.min(self.data.arrays['X'][:,i])
            summary[i]['med'] = np.median(self.data.arrays['X'][:,i])
            summary[i]['max'] = np.max(self.data.arrays['X'][:,i])
        self.templateVars = {"data_name" : self.data.name,
                             "n_inputs" : self.data.arrays['X'].shape[1],
                             "n_rows" : self.data.arrays['X'].shape[0],
                             "summary" : summary
                             }
        self.temp_dir = latex_util.initialise_tex(self.latex_dir)
        latex_util.update_tex(self.temp_dir, self.template.render({
                    'title' : 'Your report is being prepared',
                    'body' : 'This usually takes a couple of minutes - at most ten minutes.'}))

        # Write placeholder message
        self.outbox.append(dict(label='report', 
                                text='Your report is being prepared', 
                                loc=self.temp_dir))

        # Partition data in train / test
        n_data = self.data.arrays['X'].shape[0]
        random_perm = range(n_data)
        random.shuffle(random_perm)
        proportion_train = 0.5
        self.train_indices = random_perm[:int(np.floor(n_data * proportion_train))]
        self.test_indices = random_perm[int(np.floor(n_data * proportion_train)):]
        train_data, test_data = self.data.subsets([self.train_indices, self.test_indices])

        # test_data = data_sets[1]
        # Initialise list of experts
        # self.experts = [DummyModel(number=5, max_actions=10),
        #                 DummyModel(number=7, max_actions=5)]
        # self.experts = [experts.DataDoublingExpert(lambda:
        #                                            experts.CrossValidationExpert(experts.SKLearnRandomForestReg))]
        # self.experts = [experts.DataDoublingExpert(lambda: experts.CrossValidationExpert(experts.SKLassoReg))]
        # self.experts = [experts.DataDoublingExpert(lambda: experts.SamplesCrossValidationExpert(experts.SKLinearModel)),
        #                 experts.DataDoublingExpert(lambda: experts.SamplesCrossValidationExpert(experts.SKLassoReg)),
        #                 experts.DataDoublingExpert(lambda:
        #                                            experts.SamplesCrossValidationExpert(experts.SKLearnRandomForestReg))]
        # TODO - revisit this heuristic for lengthscale selection - especially if data not continuous
        # scoring_expert = experts.MMDScorer(lengthscales=np.std(self.data.arrays['X'], 0))
        scoring_expert = experts.LLHScorer()
        # self.experts = [experts.DataDoublingExpert(lambda:
        #                                            experts.SamplesCrossValidationExpert(
        #                                                experts.IndependentGaussianLearner,
        #                                                scoring_expert)),
        #                 experts.DataDoublingExpert(lambda:
        #                                            experts.SamplesCrossValidationExpert(
        #                                                experts.MoGLearner,
        #                                                scoring_expert)),
        #                 experts.DataDoublingExpert(lambda:
        #                                            experts.SamplesCrossValidationExpert(
        #                                                lambda:
        #                                                experts.RegressionLearner(experts.IndependentGaussianLearner,
        #                                                                          experts.SKLinearModel),
        #                                                scoring_expert)),
        #                 experts.DataDoublingExpert(lambda:
        #                                            experts.SamplesCrossValidationExpert(
        #                                                lambda:
        #                                                experts.RegressionLearner(experts.IndependentGaussianLearner,
        #                                                                          experts.SKLASSO),
        #                                                scoring_expert)),
        #                 experts.DataDoublingExpert(lambda:
        #                                            experts.SamplesCrossValidationExpert(
        #                                                lambda: experts.RegressionLearner(experts.MoGLearner,
        #                                                                                  experts.SKLinearModel),
        #                                                scoring_expert)),
        #                 experts.DataDoublingExpert(lambda:
        #                                            experts.SamplesCrossValidationExpert(
        #                                                lambda: experts.RegressionLearner(experts.MoGLearner,
        #                                                                                  experts.SKLASSO),
        #                                                scoring_expert))]
        self.experts = [experts.DataDoublingExpert(lambda:
                                                   experts.SamplesCrossValidationExpert(
                                                       experts.MoGLearner,
                                                       scoring_expert)),
                        experts.DataDoublingExpert(lambda:
                                                   experts.SamplesCrossValidationExpert(
                                                       lambda:
                                                       experts.RegressionLearner(experts.IndependentGaussianLearner,
                                                                                 experts.SKLinearModel),
                                                       scoring_expert))]
        # Load data into experts and set name
        for (i, expert) in enumerate(self.experts):
            expert.name = '%dof%d' % (i + 1, len(self.experts))
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
                self.all_dist_msgs.append(self.expert_queue.get_nowait())
                self.updated = True
            except q_Empty:
                break
        self.state = 'produce report'


    def produce_report(self):
        if self.updated:
            senders = set([message['sender'] for message in self.all_dist_msgs])
            maxscores = {x:-np.inf for x in senders}
            maxscore = -np.inf
            topdists = {x:None for x in senders} # highest score for each sender
            shortdescs = {x:None for x in senders}

            xs = {x:[] for x in senders}
            ys = {x:[] for x in senders}
            xarr = {x:[] for x in senders}
            for message in self.all_dist_msgs:
                sender = message['sender']
                distribution = message['distribution']
                if distribution.avscore > maxscores[sender]:
                    topdists[sender] = distribution
                    shortdescs[sender] = distribution.shortdescrip
                    maxscores[sender] = distribution.avscore
                    if distribution.avscore > maxscore:
                        topsender = sender # highest overall score
                        maxscore = distribution.avscore
                xs[sender].append(distribution.data_size)
                ys[sender].append(distribution.avscore)
                xarr[sender].append(sorted(distribution.scores))
            #print self.all_dist_msgs[-1]
            topdist = topdists[topsender]

            # make graphs
            topdist.make_graphs(self.data.subsets([self.train_indices, self.test_indices])[0],
                                self.temp_dir)
            if len(self.all_dist_msgs) / float(len(topdists)) > 1:
                gr.learning_curve(xs, xarr, shortdescs, self.temp_dir) # make the learning curve


            self.templateVars.update({'messages' : self.all_dist_msgs,
                                      'topdists' : topdists.values(),
                                      'topdist' : topdist,
                                      'outdir' : self.temp_dir})

            latex_util.update_tex(self.temp_dir, self.template.render(self.templateVars))

            self.outbox.append({'label' : 'report', 
                                'text' : 'Your report has been updated', 
                                'loc' : self.temp_dir
                                })
            
        self.state = 'check for life'

    def check_life(self):
        self.terminated = True
        for child in self.child_processes:
            if child.is_alive():
                self.terminated = False
                break
        if self.terminated == True:
            self.wait_for_experts() # get final messages from pipe
            self.produce_report()
        self.state = 'wait for experts'

    def next_action(self):
        if not self.termination_pending:
            if self.state == 'init':
                if not self.data is None:
                    self.init_data_and_experts()
            elif self.state == 'check for life':
                self.check_life()
            elif self.state == 'wait for experts':
                self.wait_for_experts()
            elif self.state == 'produce report':
                self.produce_report()
        # if self.termination_pending or self.terminated:
        #     print 'Manager will terminate'


def main():
    # Setup
    print('\nPress enter to terminate at any time.\n')
    seed = 1
    np_rand_seed(seed)
    random.seed(seed)
    # Load data
    data = XSeqDataSet()
    # data.load_from_file('../data/test-lin/simple-09.csv')
    # data.load_from_file('../data/test-lin/uci-compressive-strength.csv')
    data.load_from_file('../data/test-lin/iris_labels.csv')
    # data.load_from_file('../data/test-lin/stovesmoke-no-outliers.csv')
    # Setup up manager and communication
    queue_to_manager = multi_q()
    queue_to_main = multi_q()
    manager = Manager(inbox_q=queue_to_manager, outbox_q=queue_to_main)
    manager.load_data(data)
    # Start manager in new process / thread
    p = Thread(target=start_communication, args=(manager,))
    p.start()
    # Delete the local version of the manager to avoid confusion
    del manager
    # Listen to the manager until it finishes or crashes or user types input
    while True:
        if not p.is_alive():
            break
        while True:
            try:
                comms = queue_to_main.get_nowait()
                if comms['label'] == 'report':
                    report_loc = comms['loc']
                    print comms['text']
            except q_Empty:
                break
        # Wait for one second to see if any keyboard input
        i, o, e = select.select([sys.stdin], [], [], 1)
        if i:
            print('\n\nTerminating')
            sys.stdin.readline() # Read whatever was typed to stdin
            if p.is_alive(): # avoid sending messages to dead processes
                queue_to_manager.put(dict(label='terminate',sentby='main'))
                p.join()
            break

    print 'Report is complete'
    latex_util.compile_tex(report_loc)
    print "Report is here :" + report_loc



if __name__ == "__main__":
    main()
