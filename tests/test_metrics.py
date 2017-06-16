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

        func = mock.Mock()
        result = tf.constant(23, dtype=tf.int32, name='A01')
        func.side_effect = [result]

        metric = metrics.Metrics('awesome_metric', func)
        results = metric.compute(target, output)

        self.assertEqual(result, results)
        func.assert_called_once_with(target, output, weights=None)


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
