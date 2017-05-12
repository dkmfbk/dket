"""Metrics for `dket` model evaluation."""

import itertools

import tensorflow as tf


class Metrics(object):
    """A function used to judge performances of the model."""

    def __init__(self, metrics):
        """Initialize a new instance.

        Arguments:
          metrics: a list of callable objects implementing the
            same signature of the `Metric.compute` method.
        """
        self._metrics = metrics

    def __call__(self, truth, predicted):
        """Wrapper for the `compute` method."""
        return self.compute(truth, predicted)

    def compute(self, truth, predicted):
        """Compute a set of evaluation metrics on the results.

        Arguments:
          truth: a `Tensor` representing the gold truth.
          predicted:  a `Tensor` of same shape and dtype than `truth` representing
            actual predictions.

        Returns:
          a `list` of `Op` objects representing the evaluation metrics for the model.
        """
        return list(itertools.chain(*[m(truth, predicted) for m in self._metrics]))

    @staticmethod
    def batch_mean_accuracy():
        """Compute the mean accuracy on a batch.

        Returns:
          a `Metrics` instance that returns a list with one `Op` representing
          the average accuracy of the prediction w.r.t. the gold truth. Both
          the `truth` and `predicted` tensors are intended to be of type `tf.int32`
          and to have the same shape and are intended to be sparse representation
          of classification labels.
        """
        def _accuracy(truth, predicted):
            acc = tf.reduce_mean(
                tf.cast(
                    tf.equal(truth, predicted),
                    tf.float32),
                name='accuracy')
            return [acc]
        return Metrics([_accuracy])
