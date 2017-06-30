"""Test module for the dket.metrics module."""

import unittest

import mock
import numpy as np
import editdistance as editdist_

from dket import metrics


class TestMetric(unittest.TestCase):
    """Test case for the Metric class."""

    def test_default(self):
        """Default test case for the Metric class."""

        targets_01 = [[0, 1, 2], [0, 1, 2]]
        predictions_01 = [[0, 1, 4], [0, 1, 4]]
        lengths_01 = [3, 3]
        values_01 = [0, 1]
        avg_01 = 0.5
        min_01 = 0
        max_01 = 1

        targets_02 = [[9, 7, 5], [5, 7, 9]]
        predictions_02 = [[8, 6, 4], [4, 6, 8]]
        lengths_02 = None
        values_02 = [2, 4]
        avg_02 = 3
        # min_02 = 2
        max_02 = 4

        avg_tot = 1.75
        min_tot = min_01
        max_tot = max_02

        func = mock.Mock()
        func.side_effect = [values_01, values_02]

        metric = metrics.Metric(func=func)

        act_avg_01 = metric.compute(targets_01, predictions_01, lengths=lengths_01)
        self.assertEqual(avg_01, act_avg_01)
        self.assertEqual(1, func.call_count)
        func.assert_called_with(targets_01, predictions_01, lengths=lengths_01)
        # partial.
        self.assertEqual(avg_01, metric.average)
        self.assertEqual(min_01, metric.min)
        self.assertEqual(max_01, metric.max)

        act_avg_02 = metric.compute(targets_02, predictions_02, lengths=lengths_02)
        self.assertEqual(avg_02, act_avg_02)
        self.assertEqual(2, func.call_count)
        func.assert_called_with(targets_02, predictions_02, lengths=lengths_02)
        # total.
        self.assertEqual(avg_tot, metric.average)
        self.assertEqual(min_tot, metric.min)
        self.assertEqual(max_tot, metric.max)

    def test_reset(self):
        """Test case with reset."""

        targets_01 = [[0, 1, 2], [0, 1, 2]]
        predictions_01 = [[0, 1, 4], [0, 1, 4]]
        lengths_01 = [3, 3]
        values_01 = [0, 1]
        avg_01 = 0.5
        min_01 = 0
        max_01 = 1

        targets_02 = [[9, 7, 5], [5, 7, 9]]
        predictions_02 = [[8, 6, 4], [4, 6, 8]]
        lengths_02 = None
        values_02 = [2, 4]
        avg_02 = 3
        min_02 = 2
        max_02 = 4

        func = mock.Mock()
        func.side_effect = [values_01, values_02]

        metric = metrics.Metric(func=func)

        act_avg_01 = metric.compute(targets_01, predictions_01, lengths=lengths_01)
        self.assertEqual(avg_01, act_avg_01)
        self.assertEqual(1, func.call_count)
        func.assert_called_with(targets_01, predictions_01, lengths=lengths_01)
        # partial (1).
        self.assertEqual(avg_01, metric.average)
        self.assertEqual(min_01, metric.min)
        self.assertEqual(max_01, metric.max)

        metric.reset()
        self.assertIsNone(metric.average)
        self.assertIsNone(metric.min)
        self.assertIsNone(metric.max)

        act_avg_02 = metric.compute(targets_02, predictions_02, lengths=lengths_02)
        self.assertEqual(avg_02, act_avg_02)
        self.assertEqual(2, func.call_count)
        func.assert_called_with(targets_02, predictions_02, lengths=lengths_02)
        # partial (2).
        self.assertEqual(avg_02, metric.average)
        self.assertEqual(min_02, metric.min)
        self.assertEqual(max_02, metric.max)


class TestEditDistance(unittest.TestCase):
    """Test case for the editdistance function."""

    def test_default(self):
        """Default test case."""
        num_classes = 10
        targets_p = np.array([[1, 2, 3, 4, 0, 0, 0],
                              [1, 2, 3, 4, 5, 6, 7],
                              [1, 2, 0, 0, 0, 0, 0]])
        predictions_p_sparse = np.array(
            [[1, 2, 9, 3, 9, 9, 0],
             [1, 2, 4, 5, 6, 0, 0],
             [1, 3, 2, 9, 9, 9, 0]])
        predictions_p = np.eye(num_classes)[predictions_p_sparse]
        lengths = np.array([4, 7, 3])

        targets = [[1, 2, 3, 4], [1, 2, 3, 4, 5, 6, 7], [1, 2, 0]]
        predictions = [[1, 2, 9, 3], [1, 2, 4, 5, 6, 0, 0], [1, 3, 2]]
        exp_distances = [editdist_.eval(t, p) for t, p in zip(targets, predictions)]
        act_distances = metrics.editdistance(targets_p, predictions_p, lengths=lengths)
        self.assertEqual(exp_distances, act_distances)

    def test_no_lengths(self):
        """Test case with no length."""
        num_classes = 6
        targets = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        predictions = [[1, 2, 3, 4], [1, 2, 4, 5], [4, 3, 2, 1]]
        exp_distances = [editdist_.eval(t, p) for t, p in zip(targets, predictions)]

        targets_np = np.array(targets)
        predictions_np_sparse = np.array(predictions)
        predictions_np = np.eye(num_classes)[predictions_np_sparse]
        act_distances = metrics.editdistance(targets_np, predictions_np, lengths=None)
        self.assertEqual(exp_distances, act_distances)

if __name__ == '__main__':
    unittest.main()
