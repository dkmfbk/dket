"""Test module for the `dket.loss` module."""

import tensorflow as tf

from dket import loss as L


class TestLoss(tf.test.TestCase):
    """Test case for the loss function."""

    def _softmax_cross_entropy_rank_truth_larger(self):
        loss = L.Loss.softmax_cross_entropy()
        truth = tf.constant([[0, 0, 1], [0, 1, 0]], dtype=tf.int32)
        logits = tf.constant([[0.1, 0.9], [0.9, 0.1]], dtype=tf.float32)
        self.assertRaises(ValueError, loss, truth, logits)

    def _softmax_cross_entropy_rank_truth_too_small(self):
        loss = L.Loss.softmax_cross_entropy()
        truth = tf.constant([[0, 0, 1], [0, 1, 0]], dtype=tf.int32)
        logits = tf.constant(
            [[0.1, 0.9, 0.5, 0.5], [0.9, 0.1, 0.5, 0.5]], dtype=tf.float32)
        self.assertRaises(ValueError, loss, truth, logits)

    def _softmax_cross_entropy_sparse(self):
        truth = tf.constant([[[0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1]],
                             [[1, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0],
                              [1, 0, 0, 0, 0]]], dtype=tf.int32)
        logits = tf.constant([[[0.1, 0.2, 0.5, 1.0, 2.0],
                               [0.1, 0.2, 0.5, 1.0, 2.0],
                               [0.1, 0.2, 0.5, 1.0, 2.0]],
                              [[0.1, 0.2, 0.5, 1.0, 2.0],
                               [0.1, 0.2, 0.5, 1.0, 2.0],
                               [0.1, 0.2, 0.5, 1.0, 2.0]]])
        loss_fn = L.Loss.softmax_cross_entropy()
        loss_op = loss_fn(truth, logits)
        expected = 2.14494224389
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(loss_op)
        self.assertAllClose(expected, actual)

    def _softmax_cross_entropy_dense(self):
        truth = tf.constant([[2, 1, 4], [0, 0, 0]], dtype=tf.int32)
        logits = tf.constant([[[0.1, 0.2, 0.5, 1.0, 2.0],
                               [0.1, 0.2, 0.5, 1.0, 2.0],
                               [0.1, 0.2, 0.5, 1.0, 2.0]],
                              [[0.1, 0.2, 0.5, 1.0, 2.0],
                               [0.1, 0.2, 0.5, 1.0, 2.0],
                               [0.1, 0.2, 0.5, 1.0, 2.0]]])
        loss_fn = L.Loss.softmax_cross_entropy()
        loss_op = loss_fn(truth, logits)
        expected = 2.14494224389
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(loss_op)
        self.assertAllClose(expected, actual)

    def test_softmax_cross_entropy(self):
        """Test the `Loss` instance created with the `softmax_cross_entropy` method."""
        self._softmax_cross_entropy_rank_truth_larger()
        self._softmax_cross_entropy_rank_truth_too_small()
        self._softmax_cross_entropy_sparse()
        self._softmax_cross_entropy_dense()


if __name__ == '__main__':
    tf.test.main()
