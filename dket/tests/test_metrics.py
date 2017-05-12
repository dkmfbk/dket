"""Test module for `dket.metrics` module."""

import mock

import tensorflow as tf

from dket import metrics as M


class TestMetrics(tf.test.TestCase):
    """Test case for the `dket.metrics.Metrics` class."""

    def test_default(self):
        """Test the default behaviour of the `dket.metrics.Metrics` class."""

        truth = tf.constant([0, 1, 2], dtype=tf.int32, name='truth')
        preds = tf.constant([9, 23, 47], dtype=tf.int32, name='preds')

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

        metrics = M.Metrics([metrics_a, metrics_b])
        results = metrics.compute(truth, preds)
        results = sorted(results, key=lambda item: item.name)

        self.assertEqual(expected, results)
        metrics_a.assert_called_once_with(truth, preds)
        metrics_b.assert_called_once_with(truth, preds)

    def test_accuracy(self):
        """Test the instance created via the `batch_mean_accuracy` factory method."""

        truth = tf.constant([0, 1, 2, 1, 0], dtype=tf.int32, name='truth')
        preds = tf.constant([1, 1, 2, 3, 0], dtype=tf.int32, name='preds')
        expected = 3.0 / 5.0

        results = M.Metrics.batch_mean_accuracy().compute(truth, preds)
        self.assertEqual(1, len(results))

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(results[0])
            self.assertAllClose(expected, actual)


if __name__ == '__main__':
    tf.test.main()