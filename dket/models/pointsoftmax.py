"""Pointing Softmax model implementation."""

import tensorflow as tf
from liteflow import layers, utils

from dket.models import model


class PointingSoftmaxModel(model.Model):
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
        params = {
            'embedding_size': 128,
            'attention_size': 128,
            'recurrent_cell': 'GRU',
            'hidden_size:': 256,
            'feedback_size': 0,
            'parallel_iterations': 10
        }
        base.update(params)
        return base

    def _build_graph(self):
        trainable = self.mode == tf.contrib.learn.ModeKeys.TRAIN
        words = self.inputs.get(self.inputs.WORDS_KEY)
        slengths = self.inputs.gets(self.inputs.SENTENCE_LENGTH_KEY)
        targets = self.inputs.get(self.inputs.FORMULA_KEY)
        flengths = self.inputs.get(self.inputs.FORMULA_LENGTH_KEY)
        with self._graph.as_default():  # pylint: disable=E1129
            with tf.variable_scope('Embedding'):
                with tf.device('CPU:0'):
                    embedding_size = self._params['embedding_size']
                    vocabulary_size = self._params[self.INPUT_VOC_SIZE_PK]
                    embeddings = tf.get_variable(
                        'E', [vocabulary_size, embedding_size])
                    inputs = tf.nn.embedding_lookup(embeddings, words)

            batch_dim = utils.get_dimension(words, 0)
            with tf.variable_scope('Encoder'):  # pylint: disable=E1129
                cell = tf.contrib.rnn.GRUCell(self._params['hidden_size'])
                state = cell.zero_state(batch_dim, tf.float32)
                encoder_out, _ = tf.nn.dynamic_rnn(
                    cell=cell,
                    initial_state=state,
                    inputs=inputs,
                    sequence_length=slengths,
                    parallel_iterations=self._params['parallel_iterations'])

            with tf.variable_scope('Decoder'):  # pylint: disable=E1129
                decoder_cell = tf.contrib.rnn.GRUCell(self._params['hidden_size'])
                attention = layers.BahdanauAttention(
                    states=encoder_out,
                    inner_size=self._params['attention_size'],
                    trainable=trainable)
                location = layers.LocationSoftmax(
                    attention=attention,
                    sequence_length=slengths)
                output = layers.PointingSoftmaxOutput(
                    shortlist_size=self._params[self.OUTPUT_VOC_SIZE_PK],
                    decoder_out_size=cell.output_size,
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
