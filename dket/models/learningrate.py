"""Learning rate decay functions."""

import logging

import tensorflow as tf

from dket import configurable


class ExpLearningRateDecayFn(configurable.Configurable):
    """Exponential learning rate decay function."""

    DECAY_STEPS_PK = 'decay_steps'
    DECAY_RATE_PK = 'decay_rate'
    STAIRCASE_PK = 'staircase'

    def __init__(self, mode, params):
        train = tf.contrib.learn.ModeKeys.TRAIN
        if mode != train:
            logging.warning(
                """learning rate decay is expecting mode `%s`"""
                """, found `%s` instead.""", train, mode)
        super(ExpLearningRateDecayFn, self).__init__(mode, params)

    @classmethod
    def get_default_params(cls):
        return {
            cls.DECAY_STEPS_PK: 1000,
            cls.DECAY_RATE_PK: 0.96,
            cls.STAIRCASE_PK: True,
        }

    def _validate_params(self, params):
        decay_steps = params[self.DECAY_STEPS_PK]
        if decay_steps <= 0:
            msg = '{} must be a positive integer.'.format(self.DECAY_STEPS_PK)
            logging.critical(msg)
            raise ValueError(msg)
        decay_rate = params[self.DECAY_RATE_PK]
        if decay_rate <= 0.0 or decay_rate > 1.0:
            msg = '{} must be a float between 0.0 and 1.0'.format(self.DECAY_RATE_PK)
            logging.critical(msg)
            raise ValueError(msg)
        
        logging.debug('decay rate: %d', decay_rate)
        logging.debug('decay steps: %f', decay_steps)
        logging.debug('staircase: %s', str(params[self.STAIRCASE_PK]))
        return params

    def compute(self, learning_rate, global_step):
        """Compute the exponential decay of the learning rate."""
        return tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=global_step,
            decay_steps=self._params[self.DECAY_STEPS_PK],
            decay_rate=self._params[self.DECAY_RATE_PK],
            staircase=self._params[self.STAIRCASE_PK])
