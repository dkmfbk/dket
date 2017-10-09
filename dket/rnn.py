"""Recurrent Neural Networks building blocks."""

import abc
from collections import OrderedDict
import six

import tensorflow as tf

from dket import configurable

_TRAIN = tf.contrib.learn.ModeKeys.TRAIN

@six.add_metaclass(abc.ABCMeta)
class RNNCell(configurable.Configurable, tf.contrib.rnn.RNNCell):

    DROPOUT_INPUT_KEEP_PROB_PK = 'dropout_input.keep_prob'
    DROPOUT_OUTPUT_KEEP_PROB_PK = 'dropout_output.keep_prob'
    NUM_LAYERS_PK = 'num_layers'

    def __init__(self, mode, params):
        super(RNNCell, self).__init__(mode, params)
        cells = []
        num_layers = self._params[self.NUM_LAYERS_PK]
        input_keep_prob = self._params[self.DROPOUT_INPUT_KEEP_PROB_PK]
        output_keep_prob = self._params[self.DROPOUT_OUTPUT_KEEP_PROB_PK]
        for _ in range(num_layers):
            cell = self._build_inner_cell(mode=mode)
            if self.mode == _TRAIN:
                if input_keep_prob < 1.0 or output_keep_prob < 1.0:
                    cell = tf.contrib.rnn.DropoutWrapper(
                        cell=cell,
                        input_keep_prob=input_keep_prob,
                        output_keep_prob=output_keep_prob)
            cells.append(cell)
        if len(cells) > 1:
            self._cell = tf.contrib.rnn.MultiRNNCell(cells)
        else:
            self._cell = cells[0]

    @property
    def cell(self):
        """The inner cell."""
        return self._cell

    # RNNCell implementation.
    @property
    def state_size(self):
        """The cell state size."""
        return self.cell.state_size

    @property
    def output_size(self):
        """The cell output size."""
        return self.cell.output_size
        
    def zero_state(self, batch_size, dtype):
        """The cell zero state."""
        return self.cell.zero_state(batch_size, dtype)

    def __call__(self, inputs, state, scope=None):
        # pylint: disable=E1102,I0011
        return self.cell(inputs, state, scope=scope)
    
    @classmethod
    def get_default_params(cls):
        return OrderedDict([
            (cls.NUM_LAYERS_PK, 1),
            (cls.DROPOUT_INPUT_KEEP_PROB_PK, 1.0),
            (cls.DROPOUT_OUTPUT_KEEP_PROB_PK, 1.0),
        ])

    def _validate_params(self, params):
        # Input dropout probability must be in [0.0, 1.0], default to 1.0.
        dropout_input_keep_prob = params[self.DROPOUT_INPUT_KEEP_PROB_PK]
        if dropout_input_keep_prob is None:
            dropout_input_keep_prob = 1.0
            params[self.DROPOUT_INPUT_KEEP_PROB_PK] = 1.0
        if dropout_input_keep_prob < 0.0 or dropout_input_keep_prob > 1.0:
            raise ValueError(
                '{} must be an float between 0 and 1'\
                .format(self.DROPOUT_INPUT_KEEP_PROB_PK))

        # Output dropout probability must be in [0.0, 1.0], default to 1.0.
        dropout_output_keep_prob = params[self.DROPOUT_OUTPUT_KEEP_PROB_PK]
        if dropout_output_keep_prob is None:
            dropout_output_keep_prob = 1.0
            params[self.DROPOUT_OUTPUT_KEEP_PROB_PK] = 1.0
        if dropout_output_keep_prob < 0.0 or dropout_output_keep_prob > 1.0:
            raise ValueError(
                '{} must be an float between 0 and 1'\
                .format(self.DROPOUT_OUTPUT_KEEP_PROB_PK))

        # Number of layer must be >=1 , defaut to 1.
        num_layers = params[self.NUM_LAYERS_PK]
        if num_layers is None:
            num_layers = 1
            params[self.NUM_LAYERS_PK] = 1
        if num_layers < 1:
            raise ValueError(
                '{} must be greater or equal than 1.'\
                .format(self.NUM_LAYERS_PK))

        # return the param dictionary.
        return params

    @abc.abstractmethod
    def _build_inner_cell(self, mode=tf.contrib.learn.ModeKeys.TRAIN):
        """Build the inner cell."""
        raise NotImplementedError()


class GRUCell(RNNCell):
    """GRU recurrent cell."""

    HIDDEN_SIZE = 'hidden_size'

    @classmethod
    def get_default_params(cls):
        params = OrderedDict()
        params[cls.HIDDEN_SIZE] = 256
        for key, value in super(GRUCell, cls).get_default_params().items():
            params[key] = value
        return params

    def _validate_params(self, params):
        params = super(GRUCell, self)._validate_params(params)
        hidden_size = params[self.HIDDEN_SIZE]
        if hidden_size <= 0:
            raise ValueError('{} must be greater than zero.'.format(self.HIDDEN_SIZE))
        return params

    def _build_inner_cell(self, mode=tf.contrib.learn.ModeKeys.TRAIN):
        hidden_size = self._params[self.HIDDEN_SIZE]
        return tf.contrib.rnn.GRUCell(num_units=hidden_size)


class LSTMCell(RNNCell):
    """LSTM recurrent cell."""

    NUM_UNITS_PK = 'num_units'
    FORGET_BIAS_PK = 'forget_bias'
    LAYER_NORM_PK = 'layer_norm'
    LAYER_NORM_GAIN_PK = 'norm_gain'
    LAYER_NORM_SHIFT_PK = 'norm_shift'
    RECURRENT_DROPOUT_KEEP_PROB = 'dropout_keep_prob'
    RECURRENT_DROPOUT_PROB_SEED = 'dropout_prob_seed'


    @classmethod
    def get_default_params(cls):
        params = OrderedDict()
        params[cls.NUM_UNITS_PK] = 0
        params[cls.FORGET_BIAS_PK] = 1.0
        params[cls.LAYER_NORM_PK] = False
        params[cls.LAYER_NORM_GAIN_PK] = 1.0
        params[cls.LAYER_NORM_SHIFT_PK] = 0.0
        params[cls.RECURRENT_DROPOUT_KEEP_PROB] = 1.0
        params[cls.RECURRENT_DROPOUT_PROB_SEED] = None
        for key, value in super(LSTMCell, cls).get_default_params().items():
            params[key] = value
        return params

    def _validate_params(self, params):
        params = super(LSTMCell, self)._validate_params(params)
        if params[self.NUM_LAYERS_PK] <= 0:
            raise ValueError('{} must be greater than 0.'.format(self.NUM_UNITS_PK))
        prob = params[self.RECURRENT_DROPOUT_KEEP_PROB]
        if prob < 0.0 or prob > 1.0:
            raise ValueError(
                '{} must be a float between 0.0 and 1.0.'\
                    .format(self.RECURRENT_DROPOUT_KEEP_PROB))
        return params

    def _build_inner_cell(self, mode=tf.contrib.learn.ModeKeys.TRAIN):
        dropout_keep_prob = self._params[self.RECURRENT_DROPOUT_KEEP_PROB]
        if mode != tf.contrib.learn.ModeKeys.TRAIN:
            dropout_keep_prob = 1.0
        return tf.contrib.rnn.LayerNormBasicLSTMCell(
            num_units=self._params[self.NUM_UNITS_PK],
            forget_bias=self._params[self.FORGET_BIAS_PK],
            layer_norm=self._params[self.LAYER_NORM_PK],
            norm_gain=self._params[self.LAYER_NORM_GAIN_PK],
            norm_shift=self._params[self.LAYER_NORM_SHIFT_PK],
            dropout_keep_prob=dropout_keep_prob,
            dropout_prob_seed=self._params[self.RECURRENT_DROPOUT_PROB_SEED])
