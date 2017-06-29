"""Downstream metrics."""

import numpy as np
import editdistance as editdist_

def _avg(values):
    return sum([item * 1.0 for item in values]) / len(values)

class Metric(object):
    """..."""

    def __init__(self, func):
        self._func = func
        self._values = []

    @property
    def average(self):
        """..."""
        return _avg(self._values)

    def reset(self):
        """..."""
        self._values.clear()
        return self

    def compute(self, targets, predictions, lengths=None):
        """..."""
        values = self._func(targets, predictions, lengths=lengths)
        self._values += values
        return _avg(values)

    @classmethod
    def editdistance(cls):
        """."""
        return Metric(func=editdistance)


def editdistance(targets, predictions, lengths=None):
    """Edit distance between target and predictions."""
    distances = []
    predictions = np.argmax(predictions, axis=-1)
    for target, prediction, length in zip(targets, predictions, lengths):
        target = target[:length]
        prediction = prediction[:length]
        distances.append(editdist_.eval(target, prediction))
    return distances
