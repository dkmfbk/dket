"""Downstream metrics."""

import numpy as np
import editdistance as editdist_

def _avg(values):
    return sum([item * 1.0 for item in values]) / len(values)

class Metric(object):
    """Wrap a metric function into a moving average calclulation."""

    def __init__(self, func):
        self._func = func
        self._values = []

    @property
    def average(self):
        """The average of the values seen so far."""
        if not self._values:
            return None
        return _avg(self._values)

    @property
    def min(self):
        """The minimum value seen so far."""
        if not self._values:
            return None
        return min(self._values)

    @property
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
        """Compute the average of the metric for a batch of examples.

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
        values = self._func(targets, predictions, lengths=lengths)
        self._values += values
        return _avg(values)

    @classmethod
    def editdistance(cls):
        """Build a Metric instance calculating the edit distance."""
        return Metric(func=editdistance)


def editdistance(targets, predictions, lengths=None):
    """Edit distance between target and predictions.

    Arguments:
      targets: a `numpy` tensor of shape [batch, length] of `int` representing
        the gold truth labels.
      predictions: a `numpy` tensor of shape [batch, length, num_classes] of
        `float`, representing the predictions of the model.
      lengths: a `numpy` array of shape [batch] representing the actual lengths
        of the sentences in the batch.

    Returns:
      a list of `int` of length `batch` representing the edit distances for
        each sequence in the batch.
    """
    distances = []
    predictions = np.argmax(predictions, axis=-1)
    lengths = lengths if lengths is not None else [targets.shape[1]] * targets.shape[0]
    for target, prediction, length in zip(targets, predictions, lengths):
        target = target[:length]
        prediction = prediction[:length]
        distances.append(editdist_.eval(target, prediction))
    return distances
