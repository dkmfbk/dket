"""Test module for `dket.ops` module."""

import tensorflow as tf

from dket import ops


class TestSoftmaxXentWithLogits(tf.test.TestCase):
    """Base test case for the `dket.ops.softmax_xent_with_logits`."""

    def test_rank_truth_larger(self):
        """ValueError with rank of the truth greater than rank of logits."""
        truth = tf.constant([[0, 0, 1], [0, 1, 0]], dtype=tf.int32)
        logits = tf.constant([[0.1, 0.9], [0.9, 0.1]], dtype=tf.float32)
        self.assertRaises(ValueError, ops.softmax_xent_with_logits, truth, logits)

    def test_rank_truth_too_small(self):
        """ValueError with rank of the logits - rank of truths > 1."""
        truth = tf.constant([[0, 0, 1], [0, 1, 0]], dtype=tf.int32)
        logits = tf.constant([[0.1, 0.9, 0.5, 0.5], [0.9, 0.1, 0.5, 0.5]], dtype=tf.float32)
        self.assertRaises(ValueError, ops.softmax_xent_with_logits, truth, logits)

    def test_sparse(self):
        """The truth rank is logits rank -1."""
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
        loss = ops.softmax_xent_with_logits(truth, logits)
        expected = 2.14494224389
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(loss)
        self.assertAllClose(expected, actual)

    def test_dense(self):
        """The truth rank equals to logits rank."""
        truth = tf.constant([[2, 1, 4], [0, 0, 0]], dtype=tf.int32)
        logits = tf.constant([[[0.1, 0.2, 0.5, 1.0, 2.0],
                               [0.1, 0.2, 0.5, 1.0, 2.0],
                               [0.1, 0.2, 0.5, 1.0, 2.0]],
                              [[0.1, 0.2, 0.5, 1.0, 2.0],
                               [0.1, 0.2, 0.5, 1.0, 2.0],
                               [0.1, 0.2, 0.5, 1.0, 2.0]]])
        loss = ops.softmax_xent_with_logits(truth, logits)
        expected = 2.14494224389
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(loss)
        self.assertAllClose(expected, actual)

    