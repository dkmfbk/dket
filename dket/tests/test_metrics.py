"""Test module for `dket.metrics` module."""

import mock

import numpy as np
import tensorflow as tf

from dket import metrics


class TestMetrics(tf.test.TestCase):
    """Test case for the `dket.metrics.Metrics` class."""

    def test_default(self):
        """Test the default behaviour of the `dket.metrics.Metrics` class."""

        target = tf.constant([0, 1, 2], dtype=tf.int32, name='target')
        output = tf.constant([9, 23, 47], dtype=tf.int32, name='output')

        metrics_a = mock.Mock()
        metrics_a_res_01 = tf.no_op(name='A01')
        metrics_a_res_02 = tf.no_op(name='A02')
        metrics_a.side_effect = [[metrics_a_res_01, metrics_a_res_02]]

        metrics_b = mock.Mock()
        metrics_b_res_01 = tf.no_op(name='B01')
        metrics_b_res_02 = tf.no_op(name='B02')
        metrics_b.side_effect = [[metrics_b_res_01, metrics_b_res_02]]

        expected = [metrics_a_res_01, metrics_a_res_02,
                    metrics_b_res_01, metrics_b_res_02]
        expected = sorted(expected, key=lambda item: item.name)

        metric = metrics.Metrics([metrics_a, metrics_b])
        results = metric.compute(target, output)
        results = sorted(results, key=lambda item: item.name)

        self.assertEqual(expected, results)
        metrics_a.assert_called_once_with(target, output, weights=None)
        metrics_b.assert_called_once_with(target, output, weights=None)


class TestMeanCategoricalAccuracy(tf.test.TestCase):
    """Base test case for the `dket.ops.mean_categorical_accuracy` function."""

    def test_unweighted(self):
        """Test the mean categorical accuracy on unweighted tensors."""
        target = tf.constant([[2, 1, 0, 0]], dtype=tf.int32)
        output = tf.constant(
            [[[0.1, 0.1, 0.8],
              [0.1, 0.1, 0.8],
              [0.1, 0.8, 0.1],
              [0.8, 0.1, 0.8]]],
            dtype=tf.float32)
        accuracy = metrics.mean_categorical_accuracy(target, output)
        expected = 0.5
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(accuracy)
        self.assertAllClose(expected, actual)

    def test_weighted(self):
        """Test the mean categorical accuracy on unweighted tensors."""
        target = tf.constant([[2, 1, 0, 0]], dtype=tf.int32)
        weights = tf.placeholder(dtype=tf.float32, shape=target.shape)
        output = tf.constant(
            [[[0.1, 0.1, 0.8],
              [0.1, 0.1, 0.8],
              [0.1, 0.8, 0.1],
              [0.8, 0.1, 0.8]]],
            dtype=tf.float32)
        accuracy = metrics.mean_categorical_accuracy(target, output, weights=weights)

        # pylint: disable=I0011,E1101
        weights_and_expected = [
            (np.asarray([[2, 1, 1, 0]], dtype=np.float32), 0.5000),
            (np.asarray([[1, 1, 1, 0]], dtype=np.float32), 0.3333),
        ]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for weights_np, expected in weights_and_expected:
                actual = sess.run(accuracy, {weights: weights_np})
                self.assertAllClose(expected, actual, atol=1e-4)


if __name__ == '__main__':
    tf.test.main()
