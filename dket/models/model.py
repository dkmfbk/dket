"""Model implementation for the `dket` system."""

import abc

import tensorflow as tf

from liteflow import utils

from dket import data
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
      * `get_default_hparams()` is the classs method that is in charge to generate the default
        `HParams` for the model. Note that it is a classmethod and must be implemented
        in concrete subclasses with the @classmethod decorator.

    Example:
    ```python

    class MyModel(BaseModel):

        def _build_graph(self):
            # Build your graph from self._inputs to self._outputs.
            pass

        @classmethod
        def get_default_hparams(cls):
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

    # Define some evaluation metrics. Since they have to be used both
    # in training (batch-by-batch) and in evaluation (epoch-by-epoch)
    # they must be able to track both the batch values and the streaming
    # average, so you need to use `liteflow.metrics.StreamingMetric`.
    metrics = {
        'my_metric': liteflow.metrics.StreamingMetrics(...)
    }

    # Now you can build and feed the model:
    instance = MyModel().feed(tensors).build(hparams, loss, optimizer, metrics)
    ```
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, graph=None):
        self._graph = graph or tf.get_default_graph()
        self._global_step = ops.get_or_create_global_step(graph=self._graph)
        self._hparams = None
        self._fed = False
        self._tensors = None
        self._inputs = None
        self._target = None
        self._logits = None
        self._output = None
        self._output_mask = None
        self._loss = None
        self._loss_op = None
        self._optimizer = None
        self._train_op = None
        self._trainable = False
        self._summary_op = None
        self._metrics = None
        self._built = False

    @property
    def graph(self):
        """The graph in which the model has been created."""
        return self._graph

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
          metrics: a `dict` where the key is a string and the value is an instance
            of `liteflow.metrics.StreamingMetric`.

        Returns:
          the very same instance of the model.

        Raises:
          RuntimeError: is the model has not been fed (i.e. `self.fed` is `False`) or
            if the method has already been invoked (i.e. `self.built` is `True`)
          ValueError: if `hparams` is `None` or if `optimizer` is provided without `loss`.

        Remarks:
          for the `loss` argument, you can use an instance of the `dket.loss.Loss` class,
          for the `optimizer` argument, you can use an instance od the `dket.optimizer.Optimizer`
          class.
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
            self._loss_op = self._loss(
                self._target, self._output,
                weights=self._output_mask)

        if self._optimizer:
            self._train_op = self._optimizer.minimize(
                self._loss_op, global_step=self._global_step)

        if self._metrics:
            for _, metric in self._metrics.items():
                metric.compute(
                    self.target, self.output,
                    weights=self._output_mask)

        if self._trainable:
            self._summary_op = tf.summary.merge_all()
            if self._summary_op is None:
                self._summary_op = tf.no_op('NoSummary')

        self._built = True
        return self

    @classmethod
    @abc.abstractmethod
    def get_default_hparams(cls):
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
    def output_mask(self):
        """A tensor representing the output mask (or `None`)."""
        return self._output_mask

    # TODO(petrux): check usage and possibly REMOVE.
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
    def metrics(self):
        """A dictionary of liteflow.metrics.StreamingMetric used for the evaluation."""
        return self._metrics


class DketModel(BaseModel):
    """Base dket model."""

    __metaclass__ = abc.ABCMeta

    WORDS_KEY = data.WORDS_KEY
    SENTENCE_LENGTH_KEY = data.SENTENCE_LENGTH_KEY
    FORMULA_KEY = data.FORMULA_KEY
    FORMULA_LENGTH_KEY = data.FORMULA_LENGTH_KEY
    TARGET_KEY = FORMULA_KEY

    def __init__(self, graph=None):
        super(DketModel, self).__init__(graph=graph)
        self._words = None
        self._sentence_length = None
        self._formula_length = None
        self._formula = None

    def _feed_helper(self, tensors):
        if self.FORMULA_KEY not in tensors:
            raise ValueError("""The tensor with key `""" + self.FORMULA_KEY +
                             """` must be supplied as an input tensor.""")
        self._formula = tensors[self.FORMULA_KEY]

        self._inputs = {}
        if self.WORDS_KEY not in tensors:
            raise ValueError("""The tensor with key `""" + self.WORDS_KEY +
                             """` must be supplied as an input tensor.""")
        self._words = tensors[self.WORDS_KEY]

        self._sentence_length = tensors.get(self.SENTENCE_LENGTH_KEY, None)
        if self._sentence_length is None:
            tf.logging.info(
                self.SENTENCE_LENGTH_KEY + ' tensor not provided, creating default one.')
            batch = utils.get_dimension(self._words, 0)
            length = utils.get_dimension(self._words, 1)
            self._sentence_length = length * \
                tf.ones(dtype=tf.float32, shape=[batch])

        self._formula_length = tensors.get(self.FORMULA_LENGTH_KEY, None)
        if self._formula_length is None:
            tf.logging.info(
                self.FORMULA_KEY + ' tensor not provided, creating default one.')
            batch = utils.get_dimension(self._target, 0)
            length = utils.get_dimension(self._target, 1)
            self._formula_length = length * \
                tf.ones(dtype=tf.float32, shape=[batch])
        else:
            self._output_mask = tf.sequence_mask(
                self._formula_length, dtype=tf.float32, name='output_mask')

        self._inputs[self.WORDS_KEY] = self._words
        self._inputs[self.SENTENCE_LENGTH_KEY] = self._sentence_length
        self._inputs[self.FORMULA_LENGTH_KEY] = self._formula_length
        self._target = tensors[self.FORMULA_KEY]


    @classmethod
    @abc.abstractmethod
    def get_default_hparams(cls):
        pass

    @abc.abstractmethod
    def _build_graph(self):
        pass
