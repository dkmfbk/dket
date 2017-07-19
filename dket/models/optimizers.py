"""Optimizers for the dket model."""

import abc
import logging

import six
import tensorflow as tf

from dket import configurable

@six.add_metaclass(abc.ABCMeta)
class Optimizer(configurable.Configurable):
    """Base optimizer."""
    
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

        self._lr = self._params[self.LR_PK]
        if self._params[self.LR_DECAY_CLASS_PK]:
            lr_decay_class = self._params[self.LR_DECAY_CLASS_PK]
            lr_decay_params = self._params[self.LR_DECAY_PARAMS_PK]
            self._lr_decay_fn = configurable.factory(
                lr_decay_class, self.mode, lr_decay_params)

        self._clip_fn = None
        clip_class = self._params[self.CLIP_GRADS_CLASS_PK]
        clip_params = self._params[self.CLIP_GRADS_PARAMS_PK]
        if clip_class:
            self._clip_fn = configurable.factory(
                clip_class, self.mode, clip_params)
        
        self._optimizer = None
        self._grads, self._cgrads = [], []
 
    @property
    def learning_rate(self):
        return self._lr

    @property
    def gradients(self):
        return self._grads
        
    @property
    def clipped_gradients(self):
        return self._cgrads

    @classmethod
    def get_default_params(cls):
        return {
            'lr': 0.1,
            'lr.decay.class': '',
            'lr.decay.params': {},
            'clip.class': '',
            'clip.params': {},
            'colocate': True,
        }
    
    @abc.abstractmethod
    def _validate_params(self, params):
        raise NotImplementedError()

    @abc.abstractmethod
    def _build_optimizer(self):
        raise NotImplementedError()
    
    def minimize(self, loss, variables=None, global_step=None):
        """Minimize the loss."""
        colocate = self._params[self.COLOCATE_PK]
        if self._lr_decay_fn is not None:
            self._lr = self._lr_decay_fn(self._lr, global_step)
        self._optimizer = self._build_optimizer()
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

    def _validate_params(self, params):
        return params

    def _build_optimizer(self):
        return tf.train.GradientDescentOptimizer(self._lr)