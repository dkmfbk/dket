"""Gradient clipping components."""

import logging

import tensorflow as tf

from dket import configurable


class ClipByValueFn(configurable.Configurable):
    """Implement a clip by value function."""

    MIN_VALUE_PK = 'clip_min_value'
    MAX_VALUE_PK = 'clip_max_value'

    @classmethod
    def get_default_params(cls):
        return {
            cls.MIN_VALUE_PK: -5.0,
            cls.MAX_VALUE_PK: 5.0
        }

    def _validate_params(self, params):
        min_value = params[self.MIN_VALUE_PK]
        max_value = params[self.MAX_VALUE_PK]

        msg = '{} min value cannot be `None`.'
        if min_value is None:
            msg = msg.format(self.MIN_VALUE_PK)
            logging.critical(msg)
            raise ValueError(msg)
        if max_value is None:
            msg = msg.format(self.MAX_VALUE_PK)
            logging.critical(msg)
            raise ValueError(msg)

        if min_value >= max_value:
            msg = '{} should be less than {}, found {} and {} instead.'\
                .format(self.MIN_VALUE_PK, self.MAX_VALUE_PK, min_value, max_value)
            logging.critical(msg)
            raise ValueError(msg)
        return params

    def compute(self, grads):
        """Clip a tensor representing gradients."""
        min_value = self._params[self.MIN_VALUE_PK]
        max_value = self._params[self.MAX_VALUE_PK]
        return tf.clip_by_value(grads, min_value, max_value)
