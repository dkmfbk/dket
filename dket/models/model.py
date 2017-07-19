"""Model implementation for the `dket` system."""

import abc
import logging
import six

import tensorflow as tf

from liteflow import utils

from dket import configurable
from dket import data
from dket import ops


class ModelInputs(configurable.Configurable):
    """Dket model input tensors parser."""

    WORDS_KEY = data.WORDS_KEY
    SENTENCE_LENGTH_KEY = data.SENTENCE_LENGTH_KEY
    FORMULA_KEY = data.FORMULA_KEY
    FORMULA_LENGTH_KEY = data.FORMULA_LENGTH_KEY

    FILES_PK = 'files'
    EPOCHS_PK = 'epochs'
    BATCH_SIZE_PK = 'batch_size'
    SHUFFLE_PK = 'shuffle'
    SEED_PK = 'seed'

    def __init__(self, mode, params):
        super(ModelInputs, self).__init__(mode, params)
        self._tensors = {}
        if self._params[self.FILES_PK]:
            self._tensors = self._build_from_files()
        else:
            self._tensors = self._build_placeholders()

    @classmethod
    def get_default_params(cls):
        return {
            cls.FILES_PK: '',
            cls.EPOCHS_PK: 0,
            cls.BATCH_SIZE_PK: 200,
            cls.SHUFFLE_PK: True,
            cls.SEED_PK: None,
        }

    def _validate_params(self, params):
        logging.debug('validating the paramters.')

        # `files` is intended to be a comma separated list
        # of input files to read data from. If no file is
        # provided, all other params will be ignored.
        files = params[self.FILES_PK]
        files = files.split(',') if files else []
        for file in files:
            logging.debug('input file: %s', file)
        if not files:
            logging.info('no input files: all other params will be ignored.')
            params[self.FILES_PK] = None
            return params

        # The number of epochs must be a non-negative integer
        # or `None`. If 0 is provided, `None` will be used.
        epochs_msg = 'epochs number must be an non-negative integer or `None`'
        epochs = params[self.EPOCHS_PK]
        if epochs is None:
            pass
        elif epochs == 0:
            logging.debug('setting epochs value to `None`.')
            params[self.EPOCHS_PK] = None
        elif epochs < 0:
            logging.critical(epochs_msg)
            raise ValueError(epochs_msg)
        else:
            pass
        logging.debug('epochs: %s', str(params[self.EPOCHS_PK]))

        # The batch size must be a non-neg integer.
        batch_size_msg = 'epochs number must be an positive integer.'
        if not params[self.BATCH_SIZE_PK]:
            logging.critical(batch_size_msg)
            raise ValueError(batch_size_msg)
        logging.debug('batch size: %d')

        # The shuffle flag must be `True`, `False` or `None`.
        if params[self.SHUFFLE_PK] is None:
            logging.debug('applying automating shuffling policy.')
            params[self.SHUFFLE_PK] = self._mode == tf.contrib.learn.ModeKeys.TRAIN
        logging.debug('shuffle: %s', str(params[self.SHUFFLE_PK]))

        # The random seed must be an integer or `None`.
        logging.debug('seed: %s', str(params[self.SEED_PK]))

        # return the validate params dictionary!
        return params

    def _build_from_files(self):
        tensors = data.inputs(
            file_patterns=self._params[self.FILES_PK].split(','),
            batch_size=self._params[self.BATCH_SIZE_PK],        
            shuffle=self._params[self.SHUFFLE_PK],
            num_epochs=self._params[self.EPOCHS_PK],
            seed=self._params[self.SEED_PK])
        for key, value in tensors.items():
            logging.debug('tensor %s fetched from input pipeline (%s)', key, value)

        # WORDS_KEY is mandatory in every configuration.
        if self.WORDS_KEY not in tensors:
            msg = 'tensor `' + self.WORDS_KEY + '` is mandatory.'
            logging.critical(msg)
            raise ValueError(msg)

        # if SENTENCE_LENGTH_KEY is not provided, use the full
        # sentence lengths for each sequence in the batch.
        if self.SENTENCE_LENGTH_KEY not in tensors:
            logging.warning(
                'tensor `' + self.SENTENCE_LENGTH_KEY +
                '` not provided: using default one.')
            tensors[self.SENTENCE_LENGTH_KEY] = None

        # FORMULA_KEY is mandatory if the mode is TRAIN or EVAL,
        # while can be none if the mode is INFER.
        formula_msg = 'no `' + self.FORMULA_KEY + '` tensor provided.'
        if self.FORMULA_KEY not in tensors:
            if self._mode == tf.contrib.learn.ModeKeys.INFER:
                logging.info(formula_msg)
                tensors[self.FORMULA_KEY] = None
            else:
                logging.critical(formula_msg)
                raise ValueError(formula_msg)

        # FORMULA_LENGTH_KEY is mandatory only in TRAIN mode.
        if self.FORMULA_LENGTH_KEY not in tensors:
            if self._mode == tf.contrib.learn.ModeKeys.TRAIN:
                msg = 'tensor `' + self.FORMULA_LENGTH_KEY\
                      + '` must be provided in TRAIN mode.'
                logging.critical(msg)
                raise ValueError(msg)
            tensors[self.FORMULA_LENGTH_KEY] = None
        return tensors

    def _build_sequences_and_lengths(self):
        sequences = tf.placeholder(dtype=tf.int32, shape=[None, None])
        batch = utils.get_dimension(sequences, 0)
        length = utils.get_dimension(sequences, 1)
        lengths = length * tf.ones(dtype=tf.int32, shape=[batch])
        lengths = tf.cast(lengths, tf.float32)
        return sequences, lengths

    def _build_placeholders(self):
        logging.info('feeding model with placeholders only.')
        words, wlengths = self._build_sequences_and_lengths()
        formula, flengths = self._build_sequences_and_lengths()
        return {
            self.WORDS_KEY: words,
            self.SENTENCE_LENGTH_KEY: wlengths,
            self.FORMULA_KEY: formula,
            self.FORMULA_LENGTH_KEY: flengths,
        }

    def get(self, key):
        """Get the input tensor for the given key."""
        return self._tensors[key]
    

@six.add_metaclass(abc.ABCMeta)
class Model(configurable.Configurable):
    """Base dket model class."""

    EOS_IDX = 0
    INPUT_CLASS_PK = 'input.class'
    INPUT_PARAMS_PK = 'input.params'
    INPUT_VOC_SIZE_PK = 'input.vocabulary_size'
    OUTPUT_VOC_SIZE_PK = 'output.vocabulary_size'
    LOSS_NAME_PK = 'loss.name'
    OPTIMIZER_CLASS_PK = 'optimizer.class'
    OPTIMIZER_PARAMS_PK = 'optimizer.params',

    def __init__(self, mode, params):
        super(Model, self).__init__(mode, params)
        self._graph = None
        self._global_step = None
        self._inputs = None
        self._predictions = None
        self._loss_op = None
        self._optimizer = None
        self._train_op = None
        self._summary_op = None
        self._metrics = None

    @property
    def graph(self):
        """The graph context for the current model instance."""
        return self._graph

    @property
    def global_step(self):
        """The global step of the model."""
        return self._global_step

    @property
    def inputs(self):
        return self._inputs

    @property
    def predictions(self):
        return self._predictions

    @property
    def loss_op(self):
        return self._loss_op

    @property
    def train_op(self):
        return self._train_op

    @property
    def metrics(self):
        return self._metrics

    @classmethod
    def get_default_params(cls):
        return {
            'model.class': '',
            'input.class': 'dket.models.model.ModelInputs',
            'input.params': ModelInputs.get_default_params(),
            'input.vocabulary_size': 0,
            'output.vocabulary_size': 0,
            'loss.name': 'dket.models.losses.XEntropy',
            'optimizer.class': 'SGD',
            'optimizer.params': {
                'lr': 0.1,
                'lr.decay.class': '',
                'lr.decay.params': {},
                'clip_gradients.class': '',
                'clip_gradients.params': {},
                'colocate_ops_and_grads': True,
            }
        }

    def _validate_params(self, params):
        return params

    def _build_inputs(self):
        clz = self._params[self.INPUT_CLASS_PK]
        params = self._params[self.INPUT_PARAMS_PK]
        self._inputs = configurable.factory(clz, self.mode, params)

    @abc.abstractmethod
    def _build_graph(self):
        raise NotImplementedError()

    def _build_loss(self):
        if self.mode != tf.contrib.learn.ModeKeys.TRAIN:
            logging.debug('mode is `%s`: skipping the loss calculation.', self.mode)
            return

        targets = self.inputs.get(ModelInputs.FORMULA_KEY)
        lengths = self.inputs.get(ModelInputs.FORMULA_LENGTH_KEY)
        weights = tf.sequence_mask(lengths, dtype=tf.float32)
        predictions = self._predictions
        clz = self._params[self.LOSS_NAME_PK]
        loss = configurable.factory(clz, self.mode, {})
        self._loss_op = loss.compute(targets, predictions, weights=weights)

    def _build_train_op(self):
        if self.mode != tf.contrib.learn.ModeKeys.TRAIN:
            logging.debug('mode is `%s`: skipping the loss calculation.', self.mode)
            return

        opt_class = self._params[self.OPTIMIZER_CLASS_PK]
        opt_params = self._params[self.OPTIMIZER_PARAMS_PK]
        self._optimizer = configurable.factory(opt_class, self.mode, opt_params)
        self._train_op = self._optimizer.minimize(
            self._loss_op, global_step=self._global_step)

    def _build_metrics(self):
        logging.warning('in-graph metrics are currently NOT SUPPORTED.')
        self._metrics = {}

    def _build_summary(self):
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            for var in tf.trainable_variables():
                ops.summarize(var)
            for grad in self._optimizer.grads:
                ops.summarize(grad)
            for grad in self._optimizer.cgrads:
                ops.summarize(grad)
            tf.summary.scalar('learning_rate', self._optimizer.learning_rate)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN\
            or self.mode == tf.contrib.learn.ModeKeys.EVAL:
                for _, _ in self._metrics.items():
                    # TODO(petrux): in theory, during training you should
                    # summarize the batch values, during the evaluation you
                    # should save the moving avg. values.
                    pass

        self._summary_op = tf.summary.merge_all()

    def build(self, graph=None):
        """Build the current instance of the model."""
        if self._graph:
            raise RuntimeError(
                'The model has already been built.')
        self._graph = graph or tf.Graph()
        with self._graph.as_default() as graph:
            self._global_step = ops.get_or_create_global_step(graph=graph)
            self._build_inputs()
            self._build_graph()
            self._build_loss()
            self._build_train_op()
            self._build_metrics()
            self._build_summary()
