"""Test suite for the `dket.data` module."""

import unittest

import numpy as np
import tensorflow as tf

from dket import data


class TestEncodeDecode(unittest.TestCase):
    """Test case for the `dket.data.encode` and `dket.data.decode` functions."""

    def _assertions(self, input_, output, example):
        fmap = example.features.feature

        sentence_length = fmap[data.SENTENECE_LENGTH_KEY].int64_list.value
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
        twords, tformula = data.parse(serialized)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            awords, aformula = sess.run([twords, tformula])
        self.assertEqual(words, awords.tolist())
        self.assertEqual(formula, aformula.tolist())


if __name__ == '__main__':
    unittest.main()
