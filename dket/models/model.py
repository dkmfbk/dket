"""Model implementation for the `dket` system."""


import abc


import tensorflow as tf


class BaseModel(object):
    """Base model implementation."""

    __metaclass__ = abc.ABCMeta

    def __init__(self, hparams, loss, metrics, optimizer=None):
        self._global_step = tf.train.create_global_step()
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
