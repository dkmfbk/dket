"""Pointing Softmax model implementation."""
# TODO(petrux): dump tf.logging and switch everything to regular python logging.

import tensorflow as tf
from liteflow import layers, utils

from dket.models import model


class PointingSoftmaxModel(model.DketModel):
    """PointingSoftmax model implementation."""

    def __init__(self, graph=None):
        super(PointingSoftmaxModel, self).__init__(graph=graph)
        self._words = None
        self._sentence_length = None
        self._formula_length = None
        self._formula = None

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
                decoder = layers.pointing_decoder(
                    attention_states=encoder_out,
                    attention_inner_size=self.hparams.attention_size,
                    decoder_cell=decoder_cell,
                    shortlist_size=self.hparams.shortlist_size,
                    attention_sequence_length=self._sentence_length,
                    output_sequence_length=self._formula_length,
                    emit_out_feedback_size=self.hparams.feedback_size,
                    parallel_iterations=self.hparams.parallel_iterations,
                    swap_memory=False,
                    trainable=self.trainable)
                self._output = decoder()
