"""Pointing Softmax model implementation."""

import tensorflow as tf
from liteflow import layers, utils

from dket.models import model


class PointingSoftmaxModel(model.DketModel):
    """PointingSoftmax model implementation."""

    def __init__(self, graph=None):
        super(PointingSoftmaxModel, self).__init__(graph=graph)
        self._decoder_inputs = None

    @property
    def decoder_inputs(self):
        """A `3D Tensor` representing the gold-truth decoder inputs.
        
        *NOTE* if the model is not trainable, this tensor will be `None`
        since the decoder will be fed back with the previous output step
        by step.
        """

    @classmethod
    def get_default_hparams(cls):
        hparams = tf.contrib.training.HParams(
            vocabulary_size=0,
            embedding_size=128,
            attention_size=128,
            recurrent_cell='GRU',
            hidden_size=256,
            shortlist_size=0,
            feedback_size=0,
            parallel_iterations=10)
        return hparams

    def _build_graph(self):
        with self._graph.as_default():
            with tf.variable_scope('Embedding'):
                with tf.device('CPU:0'):
                    embedding_size = self.hparams.embedding_size
                    vocabulary_size = self.hparams.vocabulary_size
                    embeddings = tf.get_variable(
                        'E', [vocabulary_size, embedding_size])
                    inputs = tf.nn.embedding_lookup(embeddings, self._words)

            batch_dim = utils.get_dimension(self._words, 0)
            with tf.variable_scope('Encoder'):
                cell = tf.contrib.rnn.GRUCell(self.hparams.hidden_size)
                state = cell.zero_state(batch_dim, tf.float32)
                encoder_out, _ = tf.nn.dynamic_rnn(
                    cell=cell,
                    initial_state=state,
                    inputs=inputs,
                    sequence_length=self._sentence_length,
                    parallel_iterations=self.hparams.parallel_iterations)

            with tf.variable_scope('Decoder'):
                decoder_cell = tf.contrib.rnn.GRUCell(self.hparams.hidden_size)
                attention = layers.BahdanauAttention(
                    states=encoder_out,
                    inner_size=self.hparams.attention_size,
                    trainable=self.trainable)
                location = layers.LocationSoftmax(
                    attention=attention,
                    sequence_length=self._sentence_length)
                output = layers.PointingSoftmaxOutput(
                    shortlist_size=self.hparams.shortlist_size,
                    decoder_out_size=cell.output_size,
                    state_size=encoder_out.shape[-1].value,
                    trainable=self._trainable)
                
                self._decoder_inputs = None
                if self._trainable:
                    location_size = utils.get_dimension(self._words, 1)
                    output_size = self.hparams.shortlist_size + location_size
                    self._decoder_inputs = tf.one_hot(self._target, output_size, dtype=tf.float32)

                ps_decoder = layers.PointingSoftmaxDecoder(
                    cell=decoder_cell,
                    location_softmax=location,
                    pointing_output=output,
                    input_size=self.hparams.feedback_size,
                    decoder_inputs=self._decoder_inputs,
                    trainable=self._trainable)
                
                eos = None if self._trainable else self.EOS_IDX
                pad_to = None if self._trainable else utils.get_dimension(self._target, 1)
                helper = layers.TerminationHelper(lengths=self._formula_length, EOS=eos)
                decoder = layers.DynamicDecoder(
                    decoder=ps_decoder, helper=helper, pad_to=pad_to,
                    parallel_iterations=self.hparams.parallel_iterations,
                    swap_memory=False)
                
                self._predictions, _ = decoder.decode()
