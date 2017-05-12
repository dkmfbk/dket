"""Loss functions for `dket` models."""

import functools

import tensorflow as tf

from dket import ops


class Loss(object):
    """Base implementation of a loss function."""

    def __init__(self, func, accept_logits=True):
        """Initialize the Loss object."""
        self._func = func
        self._accept_logits = accept_logits

    @property
    def accept_logits(self):
        """True if logits must be provided."""
        return self._accept_logits

    @property
    def func(self):
        """The wrapped function."""
        return self._func

    def compute(self, truth, predicted, weights=1.0):
        """Compute the loss invoking the inner function."""
        return self._func(truth, predicted, weights=weights)

    def __call__(self, truth, predicted, weights=1.0):
        return self.compute(truth, predicted, weights=weights)

    @staticmethod
    def softmax_cross_entropy(scope=None, collection=tf.GraphKeys.LOSSES):
        """Loss function implementing a sparse softmax cross entropy."""
        return Loss(
            functools.partial(
                ops.softmax_xent_with_logits,
                scope=scope, loss_collection=collection))
