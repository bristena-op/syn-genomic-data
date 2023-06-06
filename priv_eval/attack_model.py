"""Parent class for all privacy attacks"""
from os import path

import logging
from logging.config import fileConfig
dirname = path.dirname(__file__)
logconfig = path.join(dirname, 'logging_config.ini')
fileConfig(logconfig)
logger = logging.getLogger(__name__)


class PrivacyAttack(object):

    def train(self, synA, labels):
        """Train attack to distinguish between synthetic data with target in original data or not"""
        return NotImplementedError('Method needs to be overwritten by a subclass.')

    def attack(self, synT):
        """Make a guess about target in or out"""
        return NotImplementedError('Method needs to be overwritten by a subclass.')

    def get_probability_of_success(self, synT, secret):
        """Calculate probability of successfully guessing target's secret"""
        return NotImplementedError('Method needs to be overwritten by a subclass.')