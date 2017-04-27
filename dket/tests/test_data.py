"""Test suite for the `dket.data` module."""

import numpy as np
import tensorflow as tf

from dket import data


class TestEncode(tf.test.TestCase):
    """Test case for the `dket.data.encode` function."""

    def _test_encode(self, input_, output, example):
        fmap = example.features.feature

        sentence_length = fmap[data.SENTENECE_LENGTH_KEY].int64_list.value
        self.assertEquals(1, len(sentence_length))
        self.assertEquals(len(input_), sentence_length[0])
        sentence_length = sentence_length[0]

        formula_length = fmap[data.FORMULA_LENGTH_KEY].int64_list.value
        self.assertEquals(1, len(formula_length))
        self.assertEquals(len(output), formula_length[0])
        formula_length = formula_length[0]

        words = fmap[data.WORDS_KEY].int64_list.value
        self.assertEquals(len(input_), len(words))
        for idx, word in zip(input_, words):
            self.assertEquals(idx, word)

        formula = fmap[data.FORMULA_KEY].int64_list.value
        self.assertEquals(len(output), len(formula))
        for idx, term in zip(output, formula):
            self.assertEquals(idx, term)

    def test_encode(self):
        """Base test for the `dket.data.encode` function."""

        input_ = [1, 2, 3, 0]
        output = [12, 23, 34, 45, 0]
        example = data.encode(input_, output)
        self._test_encode(input_, output, example)


    def test_encode_numpy(self):
        """Base test for the `dket.data.encode` function."""
        input_ = [1, 2, 3, 0]
        output = [12, 23, 34, 45, 0]
        example = data.encode(
            np.asarray(input_, dtype=np.int64),
            np.asarray(output, dtype=np.int64))
        self._test_encode(input_, output, example)


class TestDecode(tf.test.TestCase):
    """Test case for the `dket.data.decode` function."""

    def test_decode(self):
        """Base test for the `dket.data.decode` function."""

        words = [1, 2, 3, 0]
        formula = [12, 23, 34, 45, 0]
        example = data.encode(words, formula)
        twords, tformula = data.decode(example)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            awords, aformula = sess.run([twords, tformula])
        self.assertEquals(words, awords.tolist())
        self.assertEquals(formula, aformula.tolist())


if __name__ == '__main__':
    tf.test.main()
