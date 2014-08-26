"""
Generic autonomous agents classes for automatic statistician

Created August 2014

@authors: James Robert Lloyd (james.robert.lloyd@gmail.com)
"""

from Queue import Queue as thread_q
from Queue import Empty as q_Empty

import time


class Agent(object):
    def __init__(self, inbox_q=None, outbox_q=None, communication_sleep=1, child_timeout=60):
        """
        Implements a basic communication and action loop
         - Get incoming messages from parent
         - Perform next action
         - Send outgoing messages to parent
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
        if not self.inbox_q is None:  # Check we are not in single threaded mode
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
        self.get_inbox_q()
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