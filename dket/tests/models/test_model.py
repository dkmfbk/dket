"""Test module for the `dket.models.model` module."""

import mock

import tensorflow as tf

from dket import ops
from dket.models import model


class TestLoss(tf.test.TestCase):
    """Test case for the loss function."""

    def _softmax_cross_entropy_rank_truth_larger(self):
        loss = model.Loss.softmax_cross_entropy()
        truth = tf.constant([[0, 0, 1], [0, 1, 0]], dtype=tf.int32)
        logits = tf.constant([[0.1, 0.9], [0.9, 0.1]], dtype=tf.float32)
        self.assertRaises(ValueError, loss, truth, logits)

    def _softmax_cross_entropy_rank_truth_too_small(self):
        loss = model.Loss.softmax_cross_entropy()
        truth = tf.constant([[0, 0, 1], [0, 1, 0]], dtype=tf.int32)
        logits = tf.constant([[0.1, 0.9, 0.5, 0.5], [0.9, 0.1, 0.5, 0.5]], dtype=tf.float32)
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
        loss_fn = model.Loss.softmax_cross_entropy()
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
        loss_fn = model.Loss.softmax_cross_entropy()
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


class TestOptimizer(tf.test.TestCase):
    """Test case for the `dket.model.Optimizer` class."""

    def test_default(self):
        """Test for the `dket.model.Optimizer` class."""

        # ARRANGE.
        global_step = ops.get_or_create_global_step()
        coll = [tf.GraphKeys.TRAINABLE_VARIABLES]
        var_x = tf.Variable(23, dtype=tf.float32, trainable=True, collections=coll, name='x')
        var_y = tf.Variable(23, dtype=tf.float32, trainable=True, collections=coll, name='y')
        variables = [var_x, var_y]
        grad_x = tf.div(var_x, 2, name='grad_x')
        grad_y = tf.div(var_y, 2, name='grad_y')
        clip_grad_x = tf.multiply(0.1, grad_x, 'clip_grad_x')
        clip_grad_y = tf.multiply(0.5, grad_y, 'clip_grad_y')
        loss_op = tf.no_op(name='Loss')
        train_op = tf.no_op(name='Train')

        def _compute_gradients(*args, **kwargs):  # pylint: disable=I0011,W0613
            return [(grad_x, var_x), (grad_y, var_y), (None, var_x)]
        optimizer = mock.Mock()
        optimizer.compute_gradients.side_effect = _compute_gradients
        optimizer.apply_gradients.side_effect = [train_op]

        clip_map = {
            grad_x: clip_grad_x,
            grad_y: clip_grad_y
        }
        def _clip(grad):
            return clip_map.pop(grad)
        clip = mock.Mock()
        clip.side_effect = _clip

        summarize_map = {
            grad_x: 1,
            grad_y: 1,
            clip_grad_x: 1,
            clip_grad_y: 1
        }
        def _summarize(grad):
            summarize_map.pop(grad)
        summarize = mock.Mock()
        summarize.side_effect = _summarize

        # ACT.
        opt = model.Optimizer(optimizer, clip, colocate=False, summarize=summarize)
        result = opt.minimize(loss_op, variables=variables, global_step=global_step)

        # ASSERT.
        # The returned train_op is the expected one.
        self.assertEqual(result, train_op)

        # The `compute_gradients` method has been invoked for the given
        # loss op, with he given variables and with the proper value
        # for the `colocate_gradients_with_ops` flag.
        optimizer.compute_gradients.assert_called_once_with(
            loss_op, variables, colocate_gradients_with_ops=opt.colocate)

        # All the (not None) gradients have been mapped to their
        # clipped version -- i.e. the `clip_map` is empty.
        self.assertEqual(0, len(clip_map))

        # The `apply_gradients` method has been invoked with the proper
        # list of tuple of clipped gradients and variables.
        optimizer.apply_gradients.assert_called_once_with(
            [(clip_grad_x, var_x), (clip_grad_y, var_y)],
            global_step=global_step)

        # The `summarize` function has been invoked for each of the
        # gradients and for each of the clipped gradients -- i.e. the
        # summarize_map dictionary is emtpy.
        self.assertEqual(0, len(summarize_map))

    def test_optimizer_no_clip(self):
        """Test for the `dket.model.Optimizer` class withouth gradient clipping."""

        global_step = ops.get_or_create_global_step()
        coll = [tf.GraphKeys.TRAINABLE_VARIABLES]
        var_x = tf.Variable(23, dtype=tf.float32, trainable=True, collections=coll, name='x')
        var_y = tf.Variable(23, dtype=tf.float32, trainable=True, collections=coll, name='y')
        variables = [var_x, var_y]
        grad_x = tf.div(var_x, 2, name='grad_x')
        grad_y = tf.div(var_y, 2, name='grad_y')
        loss_op = tf.no_op(name='Loss')
        train_op = tf.no_op(name='Train')

        def _compute_gradients(*args, **kwargs):  # pylint: disable=I0011,W0613
            return [(grad_x, var_x), (grad_y, var_y), (None, var_x)]
        optimizer = mock.Mock()
        optimizer.compute_gradients.side_effect = _compute_gradients
        optimizer.apply_gradients.side_effect = [train_op]

         # ACT.
        opt = model.Optimizer(optimizer, clip=None, colocate=False, summarize=None)
        result = opt.minimize(loss_op, variables=variables, global_step=global_step)

        # ASSERT.
        # The returned train_op is the expected one.
        self.assertEqual(result, train_op)

        # The `compute_gradients` method has been invoked for the given
        # loss op, with he given variables and with the proper value
        # for the `colocate_gradients_with_ops` flag.
        optimizer.compute_gradients.assert_called_once_with(
            loss_op, variables, colocate_gradients_with_ops=opt.colocate)

        # The `apply_gradients` method has been invoked with the proper
        # list of tuple of clipped gradients and variables.
        optimizer.apply_gradients.assert_called_once_with(
            [(grad_x, var_x), (grad_y, var_y)],
            global_step=global_step)


    def test_optimizer_no_global_step(self):
        """Test Test for the `dket.model.Optimizer` withouth global_step."""

        _ = ops.get_or_create_global_step()
        coll = [tf.GraphKeys.TRAINABLE_VARIABLES]
        var_x = tf.Variable(23, dtype=tf.float32, trainable=True, collections=coll, name='x')
        var_y = tf.Variable(23, dtype=tf.float32, trainable=True, collections=coll, name='y')
        variables = [var_x, var_y]
        grad_x = tf.div(var_x, 2, name='grad_x')
        grad_y = tf.div(var_y, 2, name='grad_y')
        loss_op = tf.no_op(name='Loss')
        train_op = tf.no_op(name='Train')

        def _compute_gradients(*args, **kwargs):  # pylint: disable=I0011,W0613
            return [(grad_x, var_x), (grad_y, var_y), (None, var_x)]
        optimizer = mock.Mock()
        optimizer.compute_gradients.side_effect = _compute_gradients
        optimizer.apply_gradients.side_effect = [train_op]

         # ACT.
        opt = model.Optimizer(optimizer, clip=None, colocate=False, summarize=None)
        result = opt.minimize(loss_op, variables=variables)

        # ASSERT.
        # The returned train_op is the expected one.
        self.assertEqual(result, train_op)

        # The `compute_gradients` method has been invoked for the given
        # loss op, with he given variables and with the proper value
        # for the `colocate_gradients_with_ops` flag.
        optimizer.compute_gradients.assert_called_once_with(
            loss_op, variables, colocate_gradients_with_ops=opt.colocate)

        # The `apply_gradients` method has been invoked with the proper
        # list of tuple of clipped gradients and variables.
        optimizer.apply_gradients.assert_called_once_with(
            [(grad_x, var_x), (grad_y, var_y)], global_step=None)


    def test_optimizer_no_variables(self):
        """Test for the `dket.model.Optimizer` withouth explicit variable set."""

        global_step = ops.get_or_create_global_step()
        coll = [tf.GraphKeys.TRAINABLE_VARIABLES]
        var_x = tf.Variable(23, dtype=tf.float32, trainable=True, collections=coll, name='x')
        var_y = tf.Variable(23, dtype=tf.float32, trainable=True, collections=coll, name='y')
        variables = [var_x, var_y]
        grad_x = tf.div(var_x, 2, name='grad_x')
        grad_y = tf.div(var_y, 2, name='grad_y')
        loss_op = tf.no_op(name='Loss')
        train_op = tf.no_op(name='Train')

        def _compute_gradients(*args, **kwargs):  # pylint: disable=I0011,W0613
            return [(grad_x, var_x), (grad_y, var_y), (None, var_x)]
        optimizer = mock.Mock()
        optimizer.compute_gradients.side_effect = _compute_gradients
        optimizer.apply_gradients.side_effect = [train_op]

         # ACT.
        opt = model.Optimizer(optimizer, clip=None, colocate=False, summarize=None)
        result = opt.minimize(loss_op, variables=None, global_step=global_step)

        # ASSERT.
        # The returned train_op is the expected one.
        self.assertEqual(result, train_op)

        # The `compute_gradients` method has been invoked for the given
        # loss op, with he given variables and with the proper value
        # for the `colocate_gradients_with_ops` flag.
        optimizer.compute_gradients.assert_called_once_with(
            loss_op, variables, colocate_gradients_with_ops=opt.colocate)

        # The `apply_gradients` method has been invoked with the proper
        # list of tuple of clipped gradients and variables.
        optimizer.apply_gradients.assert_called_once_with(
            [(grad_x, var_x), (grad_y, var_y)], global_step=global_step)
