"""Training utilities."""

import abc
from collections import OrderedDict
import logging
import sys


import six
import tensorflow as tf
from liteflow import losses

from dket import configurable


_TRAIN_MODE_KEY = tf.contrib.learn.ModeKeys.TRAIN


class XEntropy(configurable.Configurable):
    """Categorical crossentropy loss function."""

    def __init__(self, mode, params):
        if mode != _TRAIN_MODE_KEY:
            logging.warning(
                'loss function expects mode `%s`, found `%s` instead',
                _TRAIN_MODE_KEY, mode)
        super(XEntropy, self).__init__(mode, params)

    @classmethod
    def get_default_params(cls):
        return {}

    def _validate_params(self, params):
        if params:
            logging.warning('parameters will be ignored.')
        return {}

    def compute(self, targets, predictions, weights=None):
        """Computes the categorical crossentropy."""
        values, weights = losses.categorical_crossentropy(
            targets, predictions, weights=weights)
        values = tf.multiply(values, weights)
        return tf.reduce_sum(values) / tf.reduce_sum(weights)



@six.add_metaclass(abc.ABCMeta)
class LRDecayFn(configurable.Configurable):
    """Learning rate decay function."""

    def __init__(self, mode, params):
        if mode != _TRAIN_MODE_KEY:
            logging.warning(
                """learning rate decay is expecting mode `%s`"""
                """, found `%s` instead.""", _TRAIN_MODE_KEY, mode)
        super(LRDecayFn, self).__init__(mode, params)
    
    @abc.abstractclassmethod
    def get_default_params(cls):
        raise NotImplementedError()

    @abc.abstractmethod
    def _validate_params(self, params):
        raise NotImplementedError()

    @abc.abstractmethod
    def compute(self, learning_rate, global_step):
        """Compute a decaying learning rate.

        Arguments:
          learning_rate: a `float` or a unit `Tensor` of `dtype=tf.float32`
            representing the initial value of the learning rate.
          global_step: a unit `Tensor` representing the model global step.
        
        Returns:
          a unit `Tensor` representing the decaying learning rate value.
        """
        raise NotImplementedError()


class ExponentialLRDecayFn(LRDecayFn):
    """Exponential learning rate decay function."""

    DECAY_STEPS_PK = 'decay_steps'
    DECAY_RATE_PK = 'decay_rate'
    STAIRCASE_PK = 'staircase'

    @classmethod
    def get_default_params(cls):
        return OrderedDict([
            (cls.DECAY_STEPS_PK, 1000),
            (cls.DECAY_RATE_PK, 0.96),
            (cls.STAIRCASE_PK, True),
        ])

    def _validate_params(self, params):
        decay_steps = params[self.DECAY_STEPS_PK]
        if decay_steps <= 0:
            msg = '{} must be a positive integer.'.format(self.DECAY_STEPS_PK)
            logging.critical(msg)
            raise ValueError(msg)
        decay_rate = params[self.DECAY_RATE_PK]
        if decay_rate <= 0.0 or decay_rate > 1.0:
            msg = '{} must be a float between 0.0 and 1.0'.format(self.DECAY_RATE_PK)
            logging.critical(msg)
            raise ValueError(msg)
        
        logging.debug('decay rate: %d', decay_rate)
        logging.debug('decay steps: %f', decay_steps)
        logging.debug('staircase: %s', str(params[self.STAIRCASE_PK]))
        return params

    def compute(self, learning_rate, global_step):
        """Compute the exponential decay of the learning rate."""
        return tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=global_step,
            decay_steps=self._params[self.DECAY_STEPS_PK],
            decay_rate=self._params[self.DECAY_RATE_PK],
            staircase=self._params[self.STAIRCASE_PK])


@six.add_metaclass(abc.ABCMeta)
class GradClipFn(configurable.Configurable):
    """Gradient clipping/normalization function."""

    def __init__(self, mode, params):
        if mode != _TRAIN_MODE_KEY:
            logging.warning(
                """learning rate decay is expecting mode `%s`"""
                """, found `%s` instead.""", _TRAIN_MODE_KEY, mode)
        super(GradClipFn, self).__init__(mode, params)

    @abc.abstractclassmethod
    def get_default_params(cls):
        raise NotImplementedError()

    @abc.abstractmethod
    def _validate_params(self, params):
        raise NotImplementedError()

    @abc.abstractmethod
    def compute(self, grads):
        """Clip/normalize the gradient."""
        raise NotImplementedError()

    
class GradClipByValueFn(GradClipFn):
    """Clip gradients by min/max value."""

    MIN_VALUE_PK = 'clip_min_value'
    MAX_VALUE_PK = 'clip_max_value'

    @classmethod
    def get_default_params(cls):
        return OrderedDict([
            (cls.MIN_VALUE_PK, -5.0),
            (cls.MAX_VALUE_PK, 5.0)
        ])

    def _validate_params(self, params):
        min_value = params[self.MIN_VALUE_PK]
        max_value = params[self.MAX_VALUE_PK]

        msg = '{} min value cannot be `None`.'
        if min_value is None:
            msg = msg.format(self.MIN_VALUE_PK)
            logging.critical(msg)
            raise ValueError(msg)
        if max_value is None:
            msg = msg.format(self.MAX_VALUE_PK)
            logging.critical(msg)
            raise ValueError(msg)

        if min_value >= max_value:
            msg = '{} should be less than {}, found {} and {} instead.'\
                .format(self.MIN_VALUE_PK, self.MAX_VALUE_PK, min_value, max_value)
            logging.critical(msg)
            raise ValueError(msg)
        return params

    def compute(self, grads):
        """Clip the gradients by min/max value."""
        min_value = self._params[self.MIN_VALUE_PK]
        max_value = self._params[self.MAX_VALUE_PK]
        return tf.clip_by_value(grads, min_value, max_value)


@six.add_metaclass(abc.ABCMeta)
class Optimizer(configurable.Configurable):
    """Base optimizer class.

    The default parameters for the configurable class are the following:
      lr: `float`, initial learning rate value.
      lr.decay.class: fully qualified name of a subclass of `LRDecayFn`
      lr.decay.params: parameters for the lr.decay.class configuration.
      clip.class: fully qualified name of a subclass of `GradClipFn`.
      clip.params: the gradient clipping function class params.
      colocate: `True` to colocate the gradient with the corresponding ops.
    """
    LR_PK = 'lr'
    LR_DECAY_CLASS_PK = 'lr.decay.class'
    LR_DECAY_PARAMS_PK = 'lr.decay.params'
    CLIP_GRADS_CLASS_PK = 'clip.class'
    CLIP_GRADS_PARAMS_PK = 'clip.params'
    COLOCATE_PK = 'colocate'

    def __init__(self, mode, params):
        train = tf.contrib.learn.ModeKeys.TRAIN
        if mode != train:
            logging.warning('Optimizer created with mode `{}`'.format(mode))
        super(Optimizer, self).__init__(mode, params)

        self._lr = None
        self._clip_fn = None
        self._optimizer = None
        self._grads, self._cgrads = [], []
 
    @property
    def learning_rate(self):
        """Unit `Tensor` representing the learning rate."""
        return self._lr

    @property
    def gradients(self):
        """List of tensor representing the gradients."""
        return self._grads
        
    @property
    def clipped_gradients(self):
        """List (maybe empty) of tensor representing the clipped/normalized gradients."""
        return self._cgrads

    @classmethod
    def get_default_params(cls):
        return OrderedDict()

    @abc.abstractmethod
    def _validate_params(self, params):
        raise NotImplementedError()

    @abc.abstractmethod
    def _build_optimizer(self):
        raise NotImplementedError()

    def _build_lr(self, global_step):
        lr = self._params[self.LR_PK]  # pylint: disable=I0011,C0103
        if self._params[self.LR_DECAY_CLASS_PK]:
            lr_decay_class = self._params[self.LR_DECAY_CLASS_PK]
            lr_decay_params = self._params[self.LR_DECAY_PARAMS_PK]
            lr_decay_fn = configurable.factory(
                lr_decay_class, self.mode, lr_decay_params, sys.modules[__name__])
            lr = lr_decay_fn(lr, global_step)  # pylint: disable=I0011,C0103
        return lr

    def _build_clip_fn(self):
        clip_class = self._params[self.CLIP_GRADS_CLASS_PK]
        clip_params = self._params[self.CLIP_GRADS_PARAMS_PK]
        if clip_class:
            return configurable.factory(
                clip_class, self.mode, clip_params, sys.modules[__name__])
        return None

    def minimize(self, loss, variables=None, global_step=None):
        """Minimize the loss.
        
        Arguments:
          loss: a unit `Tensor` representing the loss `Op` of the model.
          variables: a (optional) list ov variables w.r.t. compute the gradients.
            If `None`, the `tf.GraphKeys.TRAINABLE_VARIABLES` collection will be used.
          global_step: a unit `Tensor` representing the global step of the model.
        
        Return:
          an `Op` that, if evaluated, performs a training step on the model.
        """
        self._lr = self._build_lr(global_step)
        self._clip_fn = self._build_clip_fn()
        self._optimizer = self._build_optimizer()

        colocate = self._params[self.COLOCATE_PK]
        variables = variables or tf.trainable_variables()
        gvs = self._optimizer.compute_gradients(
            loss, variables, colocate_gradients_with_ops=colocate)
        gvs = [(g, v) for g, v in gvs if g is not None]
        self._grads = [gv[0] for gv in gvs]
        if self._clip_fn:
            gvs = [(self._clip_fn(g), v) for g, v in gvs]
            self._cgrads = [gv[0] for gv in gvs]
        return self._optimizer.apply_gradients(gvs, global_step=global_step)


class SGD(Optimizer):
    """Stochastic gradient descent optimizer."""

    @classmethod
    def get_default_params(cls):
        return OrderedDict([
            (cls.LR_PK, 0.1),
            (cls.LR_DECAY_CLASS_PK, ''),
            (cls.LR_DECAY_PARAMS_PK, OrderedDict()),
            (cls.CLIP_GRADS_CLASS_PK, ''),
            (cls.CLIP_GRADS_PARAMS_PK, OrderedDict()),
            (cls.COLOCATE_PK, True)
        ])

    def _validate_params(self, params):
        return params

    def _build_optimizer(self):
        return tf.train.GradientDescentOptimizer(self._lr)


class Adadelta(Optimizer):
    """AdaDelta optimizer."""

    RHO_PK = 'rho'
    EPSILON_PK = 'epsilon'

    @classmethod
    def get_default_params(cls):
        return OrderedDict([
            (cls.LR_PK, 0.001),
            (cls.RHO_PK, 0.95),
            (cls.EPSILON_PK, 1e-08),
            (cls.LR_DECAY_CLASS_PK, ''),
            (cls.LR_DECAY_PARAMS_PK, OrderedDict()),
            (cls.CLIP_GRADS_CLASS_PK, ''),
            (cls.CLIP_GRADS_PARAMS_PK, OrderedDict()),
            (cls.COLOCATE_PK, True)
        ])

    def _validate_params(self, params):
        return params

    def _build_optimizer(self):
        rho = self._params[self.RHO_PK]
        epsilon = self._params[self.EPSILON_PK]
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate=self._lr,
            rho=rho,
            epsilon=epsilon,
            use_locking=False,
            name="Adadelta")
        return optimizer
