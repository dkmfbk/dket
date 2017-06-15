"""Test module for the `PointingSoftmaxModel`."""
# TODO(petrux): make this test case a proper experiment. Not a priority, though.

import logging
import os
import random
import shutil
import tempfile

import tensorflow as tf
from liteflow import input as lin

from dket import data, losses, metrics, optimizers
from dket.models.pointsoftmax import PointingSoftmaxModel as PSM
from dket.runtime import logutils

class _MovingAvgRecord(object):

    def __init__(self, max_items_no=100, max_stats_no=100):
        self._max_items_no = max_items_no
        self._max_stats_no = max_stats_no
        self._min_item = None
        self._max_item = None
        self._items = []
        self._min_stat = None
        self._max_stat = None
        self._stats = []

    def _add_item(self, item):
        if len(self._items) == self._max_items_no:
            self._items = self._items[1:]
        self._items.append(item)
        if self._min_item is None or item < self._min_item:
            self._min_item = item
        if self._max_item is None or item > self._max_item:
            self._max_item = item

    def _add_stats(self, stat):
        if len(self._stats) == self._max_stats_no:
            self._stats = self._stats[1:]
        self._stats.append(stat)
        if self._min_stat is None or stat < self._min_stat:
            self._min_stat = stat
        if self._max_stat is None or stat > self._max_stat:
            self._max_stat = stat

    def add_item(self, item):
        """Add the item and compute the average."""
        self._add_item(item)
        average = ((sum(self._items) * 1.0) / (len(self._items) * 1.0))
        self._add_stats(average)
        return average

    def latest(self):
        """Get the latest average."""
        return self._stats[-1]


class _ToyTask(object):

    _EOS_IDX = 0
    _MIN_SHORTLIST = 1
    _MAX_SHORTLIST = 20
    _SHORTLIST_SIZE = _MAX_SHORTLIST - _MIN_SHORTLIST + 1 + 1  # consider EOS.
    _MIN_DISCARD = 21
    _MAX_DISCARD = 40
    _MIN_IDX = 1
    _MAX_IDX = 99
    _VOCABULARY_SIZE = _MAX_IDX - _MIN_IDX + 1 + 1  # consider EOS
    _MIN_LEN = 20
    _MAX_LEN = 30
    _FEEBACK_SIZE = _SHORTLIST_SIZE + 25  # half-way
    _NUM_EXAMPLES = 100000
    _LOG_EVRY = 1  # 100
    _NUM_EPOCHS = 100
    _LEARNING_RATE = 0.2
    _BATCH_SIZE = 100

    def __init__(self):
        self._data_dir = None
        self._data_file = None
        self._model = None
        self._losses = _MovingAvgRecord()
        self._accs = _MovingAvgRecord()

    def _generate_words_and_formula(self):
        length = random.randint(self._MIN_LEN, self._MAX_LEN)
        words = [random.randint(self._MIN_IDX, self._MAX_IDX) for _ in range(length)]
        formula = []
        for pos, idx in enumerate(words):
            if self._MIN_SHORTLIST <= idx <= self._MAX_SHORTLIST:
                formula.append(idx)
            elif idx > self._MAX_DISCARD:
                formula.append(self._SHORTLIST_SIZE + pos)
            else:
                pass  # discard zone.
        words.append(self._EOS_IDX)
        formula.append(self._EOS_IDX)
        return words, formula

    def _generate_data(self):
        self._data_dir = tempfile.mkdtemp()
        self._data_file = os.path.join(self._data_dir, 'dataset.rio')
        writer = tf.python_io.TFRecordWriter(self._data_file)
        for _ in range(self._NUM_EXAMPLES):
            words, formula = self._generate_words_and_formula()
            example = data.encode(words, formula)
            writer.write(example.SerializeToString())
        writer.close()

    def _build_model(self):
        # HParams
        hparams = tf.contrib.training.HParams(
            batch_size=self._BATCH_SIZE,
            vocabulary_size=self._VOCABULARY_SIZE,
            embedding_size=32,
            attention_size=50,
            recurrent_cell='GRU',
            hidden_size=128,
            shortlist_size=self._SHORTLIST_SIZE,
            feedback_size=self._FEEBACK_SIZE,
            parallel_iterations=None)

        # feeding
        tensors = data.read_from_files(
            [self._data_file], shuffle=True, num_epochs=self._NUM_EPOCHS)
        tensors = lin.shuffle_batch(tensors, batch_size=self._BATCH_SIZE)
        feeding = {
            PSM.WORDS_KEY: tf.cast(tensors[0], tf.int32),
            PSM.SENTENCE_LENGTH_KEY: tf.cast(tensors[1], tf.int32),
            PSM.FORMULA_KEY: tf.cast(tensors[2], tf.int32),
            PSM.FORMULA_LENGTH_KEY: tf.cast(tensors[3], tf.int32)
        }

        cce = losses.Loss.categorical_crossentropy()
        sgd = optimizers.Optimizer.sgd(0.1)
        acc = metrics.Metrics.mean_categorical_accuracy()

        # building
        self._model = PSM().feed(feeding).build(hparams, cce, sgd, acc)

    def _train(self):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                while True:
                    _, loss, acc, step = sess.run([
                        self._model.train_op,
                        self._model.loss_op,
                        self._model.metrics_ops[0],
                        self._model.global_step])
                    self._losses.add_item(loss)
                    self._accs.add_item(acc)
                    if step % self._LOG_EVRY == 0:
                        logging.info(
                            'step: %d - avg. accuracy: %f - avg. loss: %f',
                            step, self._accs.latest(), self._losses.latest())
            except tf.errors.OutOfRangeError as ex:
                coord.request_stop(ex=ex)
            finally:
                coord.request_stop()
                coord.join(threads)

    def _cleanup(self):
        shutil.rmtree(self._data_dir)

    def run(self):
        """Run the toy task."""
        self._generate_data()
        self._build_model()
        self._train()
        self._cleanup()

if __name__ == '__main__':
    logutils.config(level=logging.DEBUG, fpath='/tmp/test-pointingsoftmax.log', stderr=True)
    _ToyTask().run()
