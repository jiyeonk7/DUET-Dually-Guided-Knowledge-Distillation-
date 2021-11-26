import os
import sys
import time
import logging
import unittest


class Logger:
    def __init__(self, log_dir, filename='log.txt'):
        self.log_dir = log_dir
        self.logger = logging.getLogger('StarRec - ' + filename)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # File handler
        # self.log_dir = self.get_log_dir(log_dir)
        self.fh = logging.FileHandler(os.path.join(log_dir, filename))
        self.fh.setLevel(logging.DEBUG)
        fh_format = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.fh.setFormatter(fh_format)
        self.logger.addHandler(self.fh)

        # Console handler
        self.ch = logging.StreamHandler(sys.stdout)
        self.ch.setLevel(logging.INFO)
        ch_format = logging.Formatter('%(message)s')
        self.ch.setFormatter(ch_format)
        self.logger.addHandler(self.ch)

    def info(self, msg):
        self.logger.info(msg)

    def close(self):
        self.logger.removeHandler(self.fh)
        self.logger.removeHandler(self.ch)
        logging.shutdown()

def get_log_dir(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_dirs = os.listdir(log_dir)
    if len(log_dirs) == 0:
        idx = 0
    else:
        idx_list = sorted([int(d.split('_')[0]) for d in log_dirs])
        idx = idx_list[-1] + 1

    cur_log_dir = '%d_%s' % (idx, time.strftime('%Y%m%d-%H%M'))
    full_log_dir = os.path.join(log_dir, cur_log_dir)
    if not os.path.exists(full_log_dir):
        os.mkdir(full_log_dir)

    return full_log_dir


class TestLogger(unittest.TestCase):
    def runTest(self):
        # Build a logger.
        logger = Logger('..//saves')
        logger.info('Logger started.')


if __name__ == '__main__':
    unittest.main()

