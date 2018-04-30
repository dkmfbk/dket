"""Model implementation for the `dket` system."""

import abc
from collections import OrderedDict
import logging
import six

import tensorflow as tf

from liteflow import layers
from liteflow import utils

from dket import configurable
from dket import data
from dket import ops
from dket import train
from dket import rnn

SEED_DEFAULT_VALUE = 23

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
        return OrderedDict([
            (cls.FILES_PK, ''),
            (cls.EPOCHS_PK, 0),
            (cls.BATCH_SIZE_PK, 200),
            (cls.SHUFFLE_PK, True),
            (cls.SEED_PK, SEED_DEFAULT_VALUE),
        ])

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
        epochs_msg = 'epochs number must be a non-negative integer or `None`'
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
        batch_size_msg = 'batch size must be an positive integer.'
        if not params[self.BATCH_SIZE_PK]:
            logging.critical(batch_size_msg)
            raise ValueError(batch_size_msg)
        logging.debug('batch size: %d')

        # The shuffle flag must be `True`, `False` or `None`.
        if params[self.SHUFFLE_PK] is None:
            logging.debug('applying automating shuffling policy.')
            params[self.SHUFFLE_PK] = self._mode == tf.contrib.learn.ModeKeys.TRAIN
        if not self._mode == tf.contrib.learn.ModeKeys.TRAIN:
            logging.info('unseting dataset shuffling.')
            params[self.SHUFFLE_PK] = False
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
    OPTIMIZER_PARAMS_PK = 'optimizer.params'
    SEED_PK = 'seed'

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
        self._seed = int(params[self.SEED_PK]) if self.SEED_PK in params else None
        print(params)

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
        """A ModelInputs instance representing the model inputs."""
        return self._inputs

    @property
    def predictions(self):
        """The model predictions.

        A `3D Tensor` of type `tf.float32` and shape `[batch_size, timesteps, num_classes]`
        representing the model output predictions as probability distributions over the
        output symbols.
        """
        return self._predictions

    @property
    def loss_op(self):
        """The model loss operator."""
        return self._loss_op

    @property
    def train_op(self):
        """The model train operator."""
        return self._train_op

    @property
    def metrics(self):
        """Dictionary of `LiTeFlow` metrics.

        *NOTE* in-graph metrics are currently not implemented.
        """
        return self._metrics

    @property
    def summary_op(self):
        """The summary operator."""
        return self._summary_op

    @property
    def seed(self):
        """The random seed (None if not set)."""
        return self._seed

    @classmethod
    def get_default_params(cls):
        return OrderedDict([
            (cls.INPUT_VOC_SIZE_PK, 0),
            (cls.OUTPUT_VOC_SIZE_PK, 0),
            (cls.INPUT_CLASS_PK, 'dket.model.ModelInputs'),
            (cls.INPUT_PARAMS_PK, ModelInputs.get_default_params()),
            (cls.LOSS_NAME_PK, 'dket.train.XEntropy'),
            (cls.OPTIMIZER_CLASS_PK, 'dket.train.SGD'),
            (cls.OPTIMIZER_PARAMS_PK, train.Optimizer.get_default_params()),
            (cls.SEED_PK, SEED_DEFAULT_VALUE)
        ])

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
        loss = configurable.factory(clz, self.mode, {}, train)
        self._loss_op = loss.compute(targets, predictions, weights=weights)

    def _build_train_op(self):
        if self.mode != tf.contrib.learn.ModeKeys.TRAIN:
            logging.debug('mode is `%s`: skipping the loss calculation.', self.mode)
            return

        opt_class = self._params[self.OPTIMIZER_CLASS_PK]
        opt_params = self._params[self.OPTIMIZER_PARAMS_PK]
        self._optimizer = configurable.factory(opt_class, self.mode, opt_params, train)
        self._train_op = self._optimizer.minimize(
            self._loss_op, global_step=self._global_step)

    def _build_metrics(self):
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            logging.warning('in-graph metrics are currently NOT SUPPORTED.')
            self._metrics = {}

    def _build_summary(self):
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            for var in tf.trainable_variables():
                ops.summarize(var)
            for grad in self._optimizer.gradients:
                ops.summarize(grad)
            for grad in self._optimizer.clipped_gradients:
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
            if self._seed:
                tf.set_random_seed(self._seed)
            self._global_step = ops.get_or_create_global_step(graph=graph)
            self._build_inputs()
            self._build_graph()
            self._build_loss()
            self._build_train_op()
            self._build_metrics()
            self._build_summary()

    @classmethod
    def create(cls, mode, params):
        model = cls(mode, params)
        model.build()
        return model


class PointingSoftmaxModel(Model):
    """PointingSoftmax model implementation."""

    def __init__(self, mode, params):
        super(PointingSoftmaxModel, self).__init__(mode, params)
        self._decoder_inputs = None

    @property
    def decoder_inputs(self):
        """A `3D Tensor` representing the gold-truth decoder inputs.

        *NOTE* if the model is not trainable, this tensor will be `None`
        since the decoder will be fed back with the previous output step
        by step.
        """
        return self._decoder_inputs

    @classmethod
    def get_default_params(cls):
        base = super(PointingSoftmaxModel, cls).get_default_params()
        params = OrderedDict([
            ('embedding_size', 128),
            ('attention_size', 128),
            ('encoder', OrderedDict([
                ('cell.type', 'GRUCell'),
                ('cell.params', OrderedDict()),
            ])),
            ('decoder', OrderedDict([
                ('cell.type', 'GRUCell'),
                ('cell.params', OrderedDict()),
            ])),
            ('feedback_size', 0),
            ('parallel_iterations', 10)
        ])
        base.update(params)
        return base

    def _build_graph(self):
        trainable = self.mode == tf.contrib.learn.ModeKeys.TRAIN
        words = self.inputs.get(self.inputs.WORDS_KEY)
        slengths = self.inputs.get(self.inputs.SENTENCE_LENGTH_KEY)
        targets = self.inputs.get(self.inputs.FORMULA_KEY)
        flengths = self.inputs.get(self.inputs.FORMULA_LENGTH_KEY)
        with self._graph.as_default():  # pylint: disable=E1129
            if self._seed:
                tf.set_random_seed(self._seed)
            with tf.variable_scope('Embedding'):  # pylint: disable=E1129
                with tf.device('CPU:0'):
                    embedding_size = self._params['embedding_size']
                    vocabulary_size = self._params[self.INPUT_VOC_SIZE_PK]
                    embeddings = tf.get_variable(
                        'E', [vocabulary_size, embedding_size])
                    inputs = tf.nn.embedding_lookup(embeddings, words)

            batch_dim = utils.get_dimension(words, 0)
            with tf.variable_scope('Encoder'):  # pylint: disable=E1129
                encoder_params = self._params['encoder']
                encoder_cell_type = encoder_params['cell.type']
                encoder_cell_params = encoder_params['cell.params']
                encoder_cell = configurable.factory(encoder_cell_type, self._mode, encoder_cell_params, rnn)
                state = encoder_cell.zero_state(batch_dim, tf.float32)
                encoder_out, _ = tf.nn.dynamic_rnn(
                    cell=encoder_cell,
                    initial_state=state,
                    inputs=inputs,
                    sequence_length=slengths,
                    parallel_iterations=self._params['parallel_iterations'])

            with tf.variable_scope('Decoder'):  # pylint: disable=E1129
                decoder_params = self._params['decoder']
                decoder_cell_type = decoder_params['cell.type']
                decoder_cell_params = decoder_params['cell.params']
                decoder_cell = configurable.factory(decoder_cell_type, self._mode, decoder_cell_params, rnn)
                attention = layers.BahdanauAttention(
                    states=encoder_out,
                    inner_size=self._params['attention_size'],
                    trainable=trainable)
                location = layers.LocationSoftmax(
                    attention=attention,
                    sequence_length=slengths)
                output = layers.PointingSoftmaxOutput(
                    shortlist_size=self._params[self.OUTPUT_VOC_SIZE_PK],
                    decoder_out_size=decoder_cell.output_size,
                    state_size=encoder_out.shape[-1].value,
                    trainable=trainable)
                
                self._decoder_inputs = None
                if trainable:
                    location_size = utils.get_dimension(words, 1)
                    output_size = self._params[self.OUTPUT_VOC_SIZE_PK] + location_size
                    self._decoder_inputs = tf.one_hot(
                        targets, output_size, dtype=tf.float32,
                        name='decoder_training_input')
                
                ps_decoder = layers.PointingSoftmaxDecoder(
                    cell=decoder_cell,
                    location_softmax=location,
                    pointing_output=output,
                    input_size=self._params['feedback_size'],
                    decoder_inputs=self._decoder_inputs,
                    trainable=trainable)
                
                eos = None if trainable else self.EOS_IDX
                pad_to = None if trainable else utils.get_dimension(targets, 1)
                helper = layers.TerminationHelper(
                    lengths=flengths, EOS=eos)
                decoder = layers.DynamicDecoder(
                    decoder=ps_decoder, helper=helper, pad_to=pad_to,
                    parallel_iterations=self._params['parallel_iterations'],
                    swap_memory=False)
                
                self._predictions, _ = decoder.decode()
