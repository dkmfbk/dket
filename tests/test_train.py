"""Test module for the `dket.train` module."""


import mock

import tensorflow as tf

from dket import configurable
from dket import ops
from dket import train


TRAIN = tf.contrib.learn.ModeKeys.TRAIN


class TOptimizer(train.Optimizer):
    """Optimizer for test purposes."""

    # disabled warning about not invoking super's __init__
    # pylint: disable=I0011,W0231
    def __init__(self, mode, params):
        self._mode = mode
        self._params = params
        self._mock = None

    @classmethod
    def get_default_params(cls):
        return {}
    
    def _validate_params(self, params):
        return params

    @property
    def mock(self):
        """Return the wrapped mock."""
        return self._mock

    @mock.setter
    def mock(self, value):
        self._mock = value

    def _build_optimizer(self):
        return self._mock


class TestOptimizer(tf.test.TestCase):
    """Test case for the `dket.optimizer.Optimizer` class."""

    @mock.patch('dket.configurable.factory')
    def test_default(self, factory):
        """Test default working settings."""

        loss_op = tf.no_op('loss')
        train_op = tf.no_op('train')
        gs = ops.get_or_create_global_step()  # pylint: disable=I0011,C0103
        variables_exp = [
           tf.Variable(23, dtype=tf.float32, trainable=True, name='x'),
           tf.Variable(23, dtype=tf.float32, trainable=True, name='y')]
        gradients_exp = [tf.div(var, 2, name='grad_' + str(i))
                         for i, var in enumerate(variables_exp)]
        clipped_gradients_exp = [tf.multiply(0.1 * i, grad, 'clip_grad_' + str(i))
                                 for i, grad in enumerate(gradients_exp)]
        clip_map = dict(zip(gradients_exp, clipped_gradients_exp))
        clipped_grad_to_vars = dict(zip(clipped_gradients_exp, variables_exp))
        decay_lr = tf.constant('0.099')

        lr = 1.0  # pylint: disable=I0011,C0103
        decay_clz = 'DECAY'
        decay_params = {'foo': 23}
        clip_clz = 'CLIP'
        clip_params = {'bar': 'baz'}
        params = {
            train.Optimizer.LR_PK: 1.0,
            train.Optimizer.LR_DECAY_CLASS_PK: decay_clz,
            train.Optimizer.LR_DECAY_PARAMS_PK: decay_params,
            train.Optimizer.CLIP_GRADS_CLASS_PK: clip_clz,
            train.Optimizer.CLIP_GRADS_PARAMS_PK: clip_params,
            train.Optimizer.COLOCATE_PK: True,
        }

        decay = mock.Mock()
        decay.side_effect = lambda *_: decay_lr

        clip = mock.Mock()
        def _clip(grad):
            cgrad = clip_map.pop(grad, None)
            self.assertIsNotNone(cgrad)
            return cgrad
        clip.side_effect = _clip

        clzs = set([decay_clz, clip_clz])
        def _factory(clz, mode, params, module):
            self.assertEqual(mode, TRAIN)
            self.assertIn(clz, clzs)
            clzs.remove(clz)
            if clz == decay_clz:
                self.assertEqual(decay_params, params)
                self.assertEqual(train, module)
                return decay
            if clz == clip_clz:
                self.assertEqual(clip_params, params)
                self.assertEqual(train, module)
                return clip
        factory.side_effect = _factory

        optimizer = mock.Mock()
        def _compute_gradients(loss, variables, colocate_gradients_with_ops):
            self.assertEqual(loss, loss_op)
            for var in variables:
                self.assertIn(var, variables_exp)
            self.assertEqual(len(variables_exp), len(variables))
            self.assertTrue(colocate_gradients_with_ops)
            return [(g, v) for g, v in zip(gradients_exp, variables_exp)]
        optimizer.compute_gradients.side_effect = _compute_gradients

        def _apply_gradients(gvs, global_step):
            self.assertEqual(gs, global_step)
            for cgrad, var in gvs:
                self.assertIn(cgrad, clipped_grad_to_vars)
                self.assertEqual(var, clipped_grad_to_vars[cgrad])
            return train_op
        optimizer.apply_gradients.side_effect = _apply_gradients

        topt = TOptimizer(TRAIN, params)
        topt.mock = optimizer
        train_op_act = topt.minimize(loss_op, None, global_step=gs)

        self.assertFalse(clzs)
        decay.assert_called_once_with(lr, gs)
        self.assertEqual(len(gradients_exp), clip.call_count)
        optimizer.compute_gradients.assert_called_once()
        optimizer.apply_gradients.assert_called_once()
        self.assertEqual(train_op, train_op_act)


class TestConfigurableTypeResolution(tf.test.TestCase):
    """Test that all the configurable types are actually resolved."""

    def _do_test(self, ctype):
        clz = ctype.__module__ + '.' + ctype.__name__
        params = ctype.get_default_params()
        instance = configurable.factory(clz, TRAIN, params)
        self.assertIsNotNone(instance)

    def test_default(self):
        """Test the dafult creation of all the configurable types in dket.train."""
        ctypes = [
            train.ExponentialLRDecayFn,
            train.GradClipByValueFn,
            train.SGD,
        ]
        for ctype in ctypes:
            self._do_test(ctype)

if __name__ == '__main__':
    tf.test.main()
