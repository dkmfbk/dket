"""Test module for the `dket.loss` module."""

import tensorflow as tf

from dket import losses


class TestCategoricalCrossEntropy(tf.test.TestCase):
    """Base test case for the `dket.ops.softmax_xent`."""

    def test_default(self):
        """Default test case."""
        truth = tf.constant([[2, 1, 4], [0, 0, 0]], dtype=tf.int32)
        predictions = tf.constant(
            [[[0.07847758, 0.08673114, 0.11707479, 0.19302368, 0.52469283],
              [0.07847758, 0.08673114, 0.11707479, 0.19302368, 0.52469283],
              [0.07847758, 0.08673114, 0.11707479, 0.19302368, 0.52469283]],
             [[0.07847758, 0.08673114, 0.11707479, 0.19302368, 0.52469283],
              [0.07847758, 0.08673114, 0.11707479, 0.19302368, 0.52469283],
              [0.07847758, 0.08673114, 0.11707479, 0.19302368, 0.52469283]]])
        loss = losses.categorical_crossentropy(truth, predictions)
        expected = 2.14494224389
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run(loss)
        self.assertAllClose(expected, actual)

if __name__ == '__main__':
    tf.test.main()
