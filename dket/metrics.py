"""Downstream metrics."""

import numpy as np
import editdistance

def _avg(values):
    return sum([item * 1.0 for item in values]) / len(values)

class Metric(object):
    """Wrap a metric function into a moving average calclulation.
    
    The wrapped function `func` is intended to implement the inner logics
    for the metric evaluation on a single pair of target-prediction, which
    are sentences of the same length. Such function MUST have the following
    signature:

    Arguments:
      target: a `list` of `int` representing the gold truth labels.
      prediction: a `list` of `int` representing the predicted labels.

    Return:
      a `float` representing the value of the metric for the given
        target-prediction pair.
    """

    def __init__(self, func, name=None):
        self._func = func
        self._name = name
        self._values = []

    @property
    def name(self):
        """The metric name."""
        return self._name

    def average(self):
        """The average of the values seen so far."""
        if not self._values:
            return None
        return _avg(self._values)

    def min(self):
        """The minimum value seen so far."""
        if not self._values:
            return None
        return min(self._values)

    def max(self):
        """The maximum value seen so far."""
        if not self._values:
            return None
        return max(self._values)

    def reset(self):
        """Reset the moving average."""
        self._values.clear()
        return self

    def compute(self, targets, predictions, lengths=None):
        """Compute the average of the metric for a batch of examples and accumulates it.

        Arguments:
          targets: a `numpy` tensor of shape [batch, length] of `int` representing
            the gold truth labels.
          predictions: a `numpy` tensor of shape [batch, length, num_classes] of
            `float`, representing the predictions of the model.
          lengths: a `numpy` array of shape [batch] representing the actual lengths
            of the sentences in the batch.

        Returns:
          a `float` representing the average value of the metric over the batch.
        """
        values = []
        predictions = np.argmax(predictions, axis=-1)
        lengths = lengths if lengths is not None else [targets.shape[1]] * targets.shape[0]
        for target, prediction, length in zip(targets, predictions, lengths):
            target = list(target)[:length]
            prediction = list(prediction)[:length]
            values.append(self._func(target, prediction))
        self._values += values
        return _avg(values)

    @classmethod
    def editdistance(cls):
        """Build a Metric instance calculating the edit distance."""
        return Metric(func=editdistance.eval, name='EditDist')

    @classmethod
    def per_token_accuracy(cls):
        """Build a Metric instance calculating the per-token accuracy."""
        return Metric(func=per_token_accuracy, name='Acc')

    @classmethod
    def per_sentence_accuracy(cls):
        """Build a Metric instance calculating the per-sentence accuracy."""
        return Metric(func=per_sentence_accuracy, name='PSAcc')


def per_token_accuracy(target, prediction):
    """Per token accuracy."""

    if len(target) != len(prediction):
        raise ValueError(
            """`target` and `prediction` are supposed to have the same lengths """
            """(found %d and %d instead).""" % (len(target), len(prediction)))

    value = 0.0
    length = len(target) * 1.0
    for exp, act in zip(target, prediction):
        if exp == act:
            value += 1.0
    return value / length


def per_sentence_accuracy(target, prediction):
    """Per sentence accuracy."""

    if len(target) != len(prediction):
        raise ValueError(
            """`target` and `prediction` are supposed to have the same lengths """
            """(found %d and %d instead).""" % (len(target), len(prediction)))

    for exp, act in zip(target, prediction):
        if exp != act:
            return 0.0
    return 1.0
