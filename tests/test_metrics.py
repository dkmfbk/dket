"""Test module for the dket.metrics module."""

import unittest

import mock
import numpy as np

from dket import metrics


def pad(lists, padding=0):
    """Pads a list of lists."""
    maxlen = max([len(l) for l in lists])
    def _pad_list(l):  # pylint: disable=C0103
        return l + ([padding] * (maxlen - len(l)))
    return [_pad_list(l) for l in lists]


def onehot(indices, num_classes):
    return np.eye(num_classes)[indices]
    """One-hot projection of a sparse numpy array."""


class TestMetric(unittest.TestCase):
    """Test case for the Metric class."""

    def test_inner_func_usage(self):
        """Test the proper wrapping of the `func` argument."""
        num_classes = 10
        targets_exp = [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]
        predictions_exp = [[3, 2, 1], [4, 3, 2, 1], [5, 4, 3, 2, 1]]
        lengths = [3, 4, 5]
        values = [0.1, 0.2, 0.3]
        result_exp = sum(values) / len(values)

        def _func(target, prediction):
            target_exp = targets_exp.pop(0)
            prediction_exp = predictions_exp.pop(0)
            self.assertEqual(target_exp, target)
            self.assertEqual(prediction_exp, prediction)
            return values.pop(0)

        targets = np.asarray(pad(targets_exp))
        predictions = onehot(np.asarray(pad(predictions_exp)), num_classes)

        metric = metrics.Metric(func=_func)
        self.assertIsNone(metric.min())
        self.assertIsNone(metric.max())
        self.assertIsNone(metric.average())

        result_act = metric.compute(targets, predictions, lengths=lengths)
        self.assertEqual(result_exp, result_act)

    def test_default(self):
        """Default test case for the Metric class."""
        num_classes = 10
        targets_01 = np.asarray([[0, 1, 2], [0, 1, 2]])
        predictions_01 = onehot(np.asarray([[0, 1, 4], [0, 1, 4]]), num_classes)
        lengths_01 = np.asarray([3, 3])
        values_01 = [0, 1]
        avg_01 = 0.5
        min_01 = 0
        max_01 = 1

        targets_02 = np.asarray([[9, 7, 5], [5, 7, 9]])
        predictions_02 = onehot(np.asarray([[8, 6, 4], [4, 6, 8]]), num_classes)
        lengths_02 = None
        values_02 = [2, 4]
        avg_02 = 3
        min_02 = 2
        max_02 = 4

        avg_tot = 1.75
        min_tot = min_01
        max_tot = max_02

        func = mock.Mock()
        func.side_effect = values_01 + values_02

        metric = metrics.Metric(func=func)

        act_avg_01 = metric.compute(targets_01, predictions_01, lengths=lengths_01)
        self.assertEqual(avg_01, act_avg_01)
        self.assertEqual(2, func.call_count)
        # partial.
        self.assertEqual(avg_01, metric.average())
        self.assertEqual(min_01, metric.min())
        self.assertEqual(max_01, metric.max())

        act_avg_02 = metric.compute(targets_02, predictions_02, lengths=lengths_02)
        self.assertEqual(avg_02, act_avg_02)
        self.assertEqual(4, func.call_count)
        # total.
        self.assertEqual(avg_tot, metric.average())
        self.assertEqual(min_tot, metric.min())
        self.assertEqual(max_tot, metric.max())



    def test_reset(self):
        """Test case with reset."""

        num_classes = 10
        targets_01 = np.asarray([[0, 1, 2], [0, 1, 2]])
        predictions_01 = onehot(np.asarray([[0, 1, 4], [0, 1, 4]]), num_classes)
        lengths_01 = np.asarray([3, 3])
        values_01 = [0, 1]
        avg_01 = 0.5
        min_01 = 0
        max_01 = 1

        targets_02 = np.asarray([[9, 7, 5], [5, 7, 9]])
        predictions_02 = onehot(np.asarray([[8, 6, 4], [4, 6, 8]]), num_classes)
        lengths_02 = None
        values_02 = [2, 4]
        avg_02 = 3
        min_02 = 2
        max_02 = 4

        func = mock.Mock()
        func.side_effect = values_01 + values_02

        metric = metrics.Metric(func=func)
        
        act_avg_01 = metric.compute(targets_01, predictions_01, lengths=lengths_01)
        self.assertEqual(avg_01, act_avg_01)
        self.assertEqual(2, func.call_count)
        # partial (1).
        self.assertEqual(avg_01, metric.average())
        self.assertEqual(min_01, metric.min())
        self.assertEqual(max_01, metric.max())

        # Reset!
        metric.reset()
        self.assertIsNone(metric.average())
        self.assertIsNone(metric.min())
        self.assertIsNone(metric.max())

        act_avg_02 = metric.compute(targets_02, predictions_02, lengths=lengths_02)
        self.assertEqual(avg_02, act_avg_02)
        self.assertEqual(4, func.call_count)
        # partial (2).
        self.assertEqual(avg_02, metric.average())
        self.assertEqual(min_02, metric.min())
        self.assertEqual(max_02, metric.max())

if __name__ == '__main__':
    unittest.main()
