"""Test suite for the `dket.data` module."""

import itertools
import os
import random
import tempfile
import unittest

import numpy as np
import tensorflow as tf

from liteflow import input as lin

from dket import data


class TestEncodeDecode(unittest.TestCase):
    """Test case for the `dket.data.encode` and `dket.data.decode` functions."""

    def _assertions(self, input_, output, example):
        fmap = example.features.feature

        sentence_length = fmap[data.SENTENCE_LENGTH_KEY].int64_list.value
        self.assertEqual(1, len(sentence_length))
        self.assertEqual(len(input_), sentence_length[0])
        sentence_length = sentence_length[0]

        formula_length = fmap[data.FORMULA_LENGTH_KEY].int64_list.value
        self.assertEqual(1, len(formula_length))
        self.assertEqual(len(output), formula_length[0])
        formula_length = formula_length[0]

        words = fmap[data.WORDS_KEY].int64_list.value
        self.assertEqual(len(input_), len(words))
        for idx, word in zip(input_, words):
            self.assertEqual(idx, word)

        formula = fmap[data.FORMULA_KEY].int64_list.value
        self.assertEqual(len(output), len(formula))
        for idx, term in zip(output, formula):
            self.assertEqual(idx, term)

    def test_encode_decode(self):
        """Base test for the `dket.data.encode/.decode` functions."""

        input_ = [1, 2, 3, 0]
        output = [12, 23, 34, 45, 0]
        example = data.encode(input_, output)
        self._assertions(input_, output, example)

        input_, output = data.decode(example)
        self._assertions(input_, output, example)

    def test_encode_decode_numpy(self):
        """Base test for the `dket.data.encode` function."""

        input_ = [1, 2, 3, 0]
        output = [12, 23, 34, 45, 0]
        example = data.encode(
            np.asarray(input_, dtype=np.int64),
            np.asarray(output, dtype=np.int64))
        self._assertions(input_, output, example)

        input_, output = data.decode(example)
        self._assertions(input_, output, example)


class TestParse(unittest.TestCase):
    """Test case for the `dket.data.decode` function."""

    def test_parse(self):
        """Base test for the `dket.data.decode` function."""

        words = [1, 2, 3, 0]
        formula = [12, 23, 34, 45, 0]
        example = data.encode(words, formula)
        serialized = example.SerializeToString()
        words_t, sent_len_t, formula_t, form_len_t = data.parse(serialized)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual = sess.run([words_t, sent_len_t, formula_t, form_len_t])
        self.assertEqual(words, actual[0].tolist())
        self.assertEqual(len(words), np.asscalar(actual[1]))
        self.assertEqual(formula, actual[2].tolist())
        self.assertEqual(len(formula), np.asscalar(actual[3]))


class TestReadFromFiles(unittest.TestCase):
    """Test case for the reading of data from files."""

    _WORDS_MIN_LEN = 5
    _WORDS_MAX_LEN = 12
    _FORMULA_MIN_LEN = 3
    _FORMULA_MAX_LEN = 6

    def _random_io_data(self):
        wlen = range(random.randint(self._WORDS_MIN_LEN, self._WORDS_MAX_LEN))
        words = [random.randint(1, 9) for _ in wlen] + [0]
        flen = range(random.randint(self._FORMULA_MIN_LEN, self._FORMULA_MAX_LEN))
        formula = [random.randint(1, 9) for _ in flen] + [0]
        return words, formula

    def _write_examples(self, fpath, examples):
        writer = tf.python_io.TFRecordWriter(fpath)
        for words, formula in examples:
            example = data.encode(words, formula)
            writer.write(example.SerializeToString())
        writer.close()

    def _as_str(self, words, formula):
        return '-'.join([
            ''.join([str(item) for item in words]),
            ''.join([str(item) for item in formula])])

    def test_read_from_files(self):
        """Rading from files."""
        tdir = tempfile.TemporaryDirectory()
        files = {
            'afile-00.txt': [self._random_io_data() for _ in range(5)],
            'afile-10.txt': [self._random_io_data() for _ in range(5)],
            'bfile-09.txt': [self._random_io_data() for _ in range(5)],
            'bfile-23.txt': [self._random_io_data() for _ in range(5)]
        }
        file_patterns = [
            os.path.join(tdir.name, 'afile-*.txt'),
            os.path.join(tdir.name, 'bfile-*.*')]
        expecteds = list(itertools.chain(*files.values()))
        for fpath, examples in files.items():
            self._write_examples(os.path.join(tdir.name, fpath), examples)

        tensors = data.read_from_files(file_patterns, shuffle=False, num_epochs=1)
        self.assertEqual(4, len(tensors))

        actuals = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                while True:
                    actual = sess.run(tensors)
                    actuals.append(actual)
            except tf.errors.OutOfRangeError as ex:
                coord.request_stop(ex=ex)
            finally:
                coord.request_stop()
                coord.join(threads)
        tdir.cleanup()

        self.assertEqual(len(actuals), len(expecteds))
        for actual in actuals:
            words, wlen, formula, flen = tuple(actual)
            self.assertEqual(len(words), wlen)
            self.assertEqual(len(formula), flen)
        expecteds = sorted([self._as_str(w, f) for w, f in expecteds])
        actuals = sorted([self._as_str(w, f) for w, _, f, _ in actuals])
        for exp, act in zip(expecteds, actuals):
            self.assertEqual(exp, act)

    def test_smoke(self):
        """Smoke test for a full pipeline."""
        _, tname = tempfile.mkstemp()
        num = 100
        num_epochs = 2
        self._write_examples(tname, [self._random_io_data() for _ in range(num)])
        tensors = data.read_from_files([tname], shuffle=True, num_epochs=num_epochs)
        batches = lin.shuffle_batch(tensors=tensors, batch_size=5)

        count = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                while True:
                    actual = sess.run(batches)
                    count += len(actual[0])
            except tf.errors.OutOfRangeError as ex:
                coord.request_stop(ex=ex)
            finally:
                coord.request_stop()
                coord.join(threads)
        self.assertEqual(num * num_epochs, count)
        os.remove(tname)

if __name__ == '__main__':
    unittest.main()
