"""Loss functions."""

import logging

import tensorflow as tf

from liteflow.losses import categorical_crossentropy as xentropy

from dket import configurable


class XEntropy(configurable.Configurable):
    """Categorical crossentropy loss function."""

    _TRAIN = tf.contrib.learn.ModeKeys.TRAIN

    def __init__(self, mode, params):
        if mode != self._TRAIN:
            logging.warning(
                'loss function expects mode `%s`, found `%s` instead',
                self._TRAIN, mode)
        super(XEntropy, self).__init__(mode, params)

    @classmethod
    def get_default_params(cls):
        return {}

    def _validate_params(self, params):
        if params:
            logging.warning('parameters will be ignored.')
        return {}

    def compute(self, targets, predictions, weights=None):
        """Computes the categorical crossentropy."""
        values, weights = xentropy(targets, predictions, weights=weights)
        values = tf.multiply(values, weights)
        return tf.reduce_sum(values) / tf.reduce_sum(weights)
