"""Model implementation for the `dket` system."""

import abc

import tensorflow as tf

from dket import ops


class BaseModel(object):
    """Base model implementation.

    This class implements a SINGLE TASK model where single task means one
    target tensor, one output, one loss. The model can handle multiple inputs.
    The model must be first fed with input(s) and target `Tensor`s and then
    built, specifying its topology configuration, the loss function, the
    optimization algorithm and evaluation metrics. If the optimization is not
    specified, the model will not be trainable.

    NOTA BENE for the implementors of concrete subclasses. What you need to do is
    to subclass the `BaseModel` class and to define its two abstract methods.abc

      * `_build_graph()` is the method which is in charge to build the graph. Implementing
        this method you MUST set the `self._logits` and the `self._output` tensors. This
        method is invoked as an abstract template from within the `.build()` method.
      * `get_default_hparams()` is the model that is in charge to generate the default
        `HParams` for the model. This method MUST be invokable at the very early stage
        of the building phase, when maybe no property has been set so it should be
        like a static method, totally independent from the state of the object.

    Example:
    ```python

    class MyModel(BaseModel):

        def _build_graph(self):
            # Build your graph from self._inputs to self._outputs.
            pass

        def get_default_hparams(self):
            # Build the default `HParams`.
            pass

    # Define the inputs and the target: can be placeholder
    # or tenrors coming from a dequeue operation.
    tensors = {
        'A': tf.placeholder(...),
        'B': tf.placeholder(...),
        'target': tf.placeholder(...)
    }

    # Define your `HParams` to be used in the model configuration:
    hparams = tf.contrib.training.HParams(...)

    # Define the loss function, which is a function accepting
    # the target and the output tenso as argumetns. It is strongly
    # suggested to use an instance of the `dket.loss.Loss` class.
    loss = ...

    # Define the optimizer. Same as for the loss, it could be better
    # to use some instance of the `dket.oprimizer.Optimizer` class.
    optimizer = ...

    # Define some evaluation metrics. Same as for the loss, it could be better
    # to use some instance of the `dket.metrics.Metrics` class.
    metrics = ...

    # Now you can build and feed the model:
    instance = MyModel().feed(tensors).build(hparams, loss, optimizer, metrics)
    ```
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._global_step = ops.get_or_create_global_step()
        self._hparams = None
        self._fed = False
        self._tensors = None
        self._inputs = None
        self._target = None
        self._logits = None
        self._output = None
        self._loss = None
        self._loss_op = None
        self._optimizer = None
        self._train_op = None
        self._trainable = False
        self._summary_op = None
        self._metrics = None
        self._metrics_ops = None
        self._built = False

    @abc.abstractmethod
    def _feed_helper(self, tensors):
        """Process the feed tensors and place inputs and target."""
        raise NotImplementedError('To be implemented in subclasses.')

    def feed(self, tensors):
        """Feed the model with input and target queues.

        Arguments:
          tensors: a `Tensor` or a dict of `str`, `Tensor` representing the model input(s).
          target: a `Tensor` representing the model output.

        Returns:
          the very same instance of the model.
        """

        if self._fed:
            raise RuntimeError(
                'Cannot feed a model that has already been fed.')

        if self._built:
            raise RuntimeError(
                'Cannot feed a model that has already been built.')

        if tensors is None:
            raise ValueError('`tensors` argument cannot be `None`')

        self._feed_helper(tensors)
        self._fed = True
        return self

    @property
    def fed(self):
        """`True` if the model has beed feed with queues."""
        return self._fed

    @property
    def feeding(self):
        """The feeding tensors."""
        return self._tensors

    def build(self, hparams, loss=None, optimizer=None, metrics=None):
        """Build the model instance.

        This method is the backbon of the model creation and performs many operations,
        leaving the graph creation to the asbtract template method `_build_graph()` which
        MUST be implemented in subclasses and MUST complete leaving the `self._logits`
        and `self._output` tensors defined. The method will set the `self.built` flag
        value to `True`.

        Arguments:
          hparams: a `tf.contrib.training.HParams` representing the configuration for
            the current model instance. Such settings will be merged with the default ones
            so that all the entries in the default one will be overwritten and all the
            entries that are not in the default one will be discarded.
          loss: a function accepting the `self.target` and `self.output` tensors as arguments
            and returning the loss op that will be placed in `self.loss_op` representing
            the loss function of the model.
          optimizer: a function accepting the `self.loss_op` tensor, an optional list
            of trainable variables (or getting the graph tf.GraphKeys.TRAINABLE_VARIABLES if such
            list is not provided) and the `global_step` as a named argument. If this argument
            is `None`, the `self.trainable` flag is set to `False`.
          metrics: a function accepting the `self.target` and `self.output` tensors as arguments
            and returning a list of ops representing evaluation metrics for the model.

        Returns:
          the very same instance of the model.

        Raises:
          RuntimeError: is the model has not been fed (i.e. `self.fed` is `False`) or
            if the method has already been invoked (i.e. `self.built` is `True`)
          ValueError: if `hparams` is `None` or if `optimizer` is provided without `loss`.

        Remarks:
          for the `loss` argument, you can use an instance of the `dket.loss.Loss` class,
          for the `optimizer` argument, you can use an instance od the `dket.optimizer.Optimizer`
          class and, finally, for the `metrics` argument you can use an instance of the
          `dket.metrics.Metrics` class.
        """
        if not self._fed:
            raise RuntimeError('The model has not been fed yes.')

        if self._built:
            raise RuntimeError('The model has already been built.')

        if loss is None and optimizer is not None:
            raise ValueError(
                'If `loss` is `None`, `optimizer` must be `None`.')

        if hparams is None:
            raise ValueError('`hparams` cannot be `None`.')

        self._hparams = self._set_hparams(hparams)
        self._loss = loss
        self._optimizer = optimizer
        self._metrics = metrics
        self._trainable = self._optimizer is not None
        self._build_graph()

        if self._loss:
            self._loss_op = self._loss(self.target, self._output)

        if self._optimizer:
            self._train_op = self._optimizer.minimize(
                self._loss_op, global_step=self._global_step)

        if self._metrics:
            self._metrics_ops = self._metrics(self.target, self.output)

        if self._trainable:
            self._summary_op = tf.summary.merge_all()
            if self._summary_op is None:
                self._summary_op = tf.no_op('NoSummary')

        self._built = True
        return self

    @abc.abstractmethod
    def get_default_hparams(self):
        """Returns the default `tf.contrib.training.HParams`.

        Remarks: this method will be called in a static-like scenario so
        so nothing (property, fields, ecc.) from the instance state should
        be used.
        """
        raise NotImplementedError(
            'This method must be implemented in subclasses')

    def _set_hparams(self, hparams):
        actual = hparams.values()
        default = self.get_default_hparams().values()
        merged = tf.contrib.training.HParams()
        for key, value in default.items():
            if key in actual:
                value = actual[key]
            merged.add_hparam(key, value)
        return merged

    @property
    def built(self):
        """`True` if the model has been built."""
        return self._built

    @abc.abstractmethod
    def _build_graph(self):
        """Build the (inference) graph."""
        raise NotImplementedError(
            'This method must be implemented in subclasses')

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
    def inputs(self):
        """A tensor or a dictionary of tensors represenitng the model input(s)."""
        return self._inputs

    @property
    def target(self):
        """A tensor representing the target output of the model."""
        return self._target

    @property
    def logits(self):
        """Unscaled log propabilities."""
        return self._logits

    @property
    def output(self):
        """A tensor representing the actual output of the model."""
        return self._output

    @property
    def loss_op(self):
        """The loss op of the model."""
        return self._loss_op

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
