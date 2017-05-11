"""Model implementation for the `dket` system."""


import abc
import functools

import tensorflow as tf

from dket import ops

class Loss(object):
    """Base implementation of a loss function."""

    def __init__(self, func, accept_logits=True):
        """Initialize the Loss object."""
        self._func = func
        self._accept_logits = accept_logits

    @property
    def accept_logits(self):
        """True if logits must be provided."""
        return self._accept_logits

    @property
    def func(self):
        """The wrapped function."""
        return self._func

    def call(self, truth, predicted, weights=1.0):
        """Call the inner loss function."""
        return self._func(truth, predicted, weights=weights)

    def __call__(self, truth, predicted, weights=1.0):
        return self.call(truth, predicted, weights=weights)

    @staticmethod
    def softmax_cross_entropy(scope=None, collection=tf.GraphKeys.LOSSES):
        """Loss function implementing a sparse softmax cross entropy."""
        return Loss(
            functools.partial(
                ops.softmax_xent_with_logits,
                scope=scope, loss_collection=collection))


class Optimizer(object):
    """Base implementation of the optimizer function."""

    @staticmethod
    def _no_summary(var):
        pass

    def __init__(self, optimizer, clip, colocate=False, summarize=ops.summarize):
        """Initialize the new optimizer instance.
        """
        self._optimizer = optimizer
        self._clip = clip
        self._colocate = colocate
        self._summarize = summarize or Optimizer._no_summary

    @property
    def optimizer(self):
        """The wrapped `tf.train.Optimizer`."""
        return self._optimizer

    @property
    def clip(self):
        """The wrapped gradient clipping function (if any)."""
        return self._clip

    @property
    def colocate(self):
        """If `True`, try to colocate gradients and ops."""
        return self._colocate

    @property
    def summarize(self):
        """The summarization function invoked on gradients (and clipped)."""
        return self._summarize

    def minimize(self, loss, variables=None, global_step=None):
        """Minimize the loss w.r.t. the variables.
        """

        variables = variables or tf.trainable_variables()

        grads_and_vars = self._optimizer.compute_gradients(
            loss, variables, colocate_gradients_with_ops=self._colocate)

        # Remove the None gradiend which might have come from
        # variables actually not influencing the loss value.
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]

        for grad, _ in grads_and_vars:
            self._summarize(grad)

        if self._clip is not None:
            for i, grad_and_var in enumerate(grads_and_vars):
                grad, var = grad_and_var
                clipped_grad = self._clip(grad)
                self._summarize(clipped_grad)
                grads_and_vars[i] = (clipped_grad, var)

        return self.optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

    @staticmethod
    def sgd(learning_rate, clip=None, colocate=True, summarize=ops.summarize):
        """Create a Stochastic Gradient Descent optimizer.

        Arguments:
          aa

        Returns:
          bb
        """
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        return Optimizer(optimizer, clip=clip, colocate=colocate, summarize=summarize)



class BaseModel(object):
    """Base model implementation."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, hparams, loss, metrics, optimizer=None):
        self._global_step = ops.get_or_create_global_step()
        self._hparams = self._set_hparams(hparams)
        self._loss = loss
        self._optimizer = optimizer
        self._metrics = metrics
        self._trainable = self._optimizer is not None
        self._input = None
        self._target = None
        self._output = None
        self._loss_op = None
        self._train_op = None
        self._summary_op = None
        self._metrics_ops = None
        self._build_model()

    @abc.abstractmethod
    def get_default_hparams(self):
        """Returns the default `tf.contrib.training.HParams`"""
        raise NotImplementedError('This method must be implemented in subclasses')

    def _set_hparams(self, hparams):
        actual = hparams.values()
        default = self.get_default_hparams().values()
        merged = tf.contrib.training.HParams()
        for key, value in default.iteritems():
            if key in actual:
                value = actual[key]
            merged.add_hparam(key, value)
        return merged

    @abc.abstractmethod
    def _build_model(self):
        """Build the model."""
        raise NotImplementedError('This method must be implemented in subclasses')

    @property
    def global_step(self):
        """The global step of the model."""
        return self._global_step

    @property
    def hparams(self):
        """The full initialization HParams of the model."""
        return self._hparams

    @property
    def loss(self):
        """The loss function."""
        return self._loss

    @property
    def optimizer(self):
        """The optimizer of the model."""
        return self._optimizer

    @property
    def metrics(self):
        """The metrics object for the model."""
        return self._metrics

    @property
    def trainable(self):
        """`True` if the model is trainable."""
        return self._trainable

    @property
    def input(self):
        """A tensor or a dictionary of tensors represenitng the model input(s)."""
        return self._input

    @property
    def target(self):
        """A tensor representing the target output of the model."""
        return self._target

    @property
    def output(self):
        """A tensor representing the actual output of the model."""
        return self._output

    @property
    def loss_op(self):
        """The loss op of the model."""
        return self._loss

    @property
    def train_op(self):
        """The train op of the model."""
        return self._train_op

    @property
    def summary_op(self):
        """The summary op of the model."""
        return self._summary_op

    @property
    def metrics_ops(self):
        """A list of ops for evaluation."""
        return self._metrics_ops
