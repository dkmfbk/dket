"""Test utilities."""

import os
import shutil
import random
import tempfile

import tensorflow as tf

from dket import data


class TestExampleGenerator(object):
    """Generate a training example."""

    def __init__(self, words_min_len=5, words_max_len=5, words_range=(1, 9),
                 formula_min_len=3, formula_max_len=6, formula_range=(1, 9)):

        self._wmin_len = words_min_len
        self._wmax_len = words_max_len
        self._wrange = words_range

        self._fmin_len = formula_min_len
        self._fmax_len = formula_max_len
        self._frange = formula_range

    def next(self):
        """Genrates a new example."""
        wlen = range(random.randint(self._wmin_len, self._wmax_len))
        words = [random.randint(self._wrange[0], self._wrange[1]) for _ in wlen] + [0]
        flen = range(random.randint(self._fmin_len, self._fmax_len))
        formula = [random.randint(self._frange[0], self._frange[1]) for _ in flen] + [0]
        return words, formula

    def __call__(self):
        return self.next()


class TestDataFactory(object):
    """Test data factory."""

    WORDS_MIN_LEN = 5
    WORDS_MAX_LEN = 12
    FORMULA_MIN_LEN = 3
    FORMULA_MAX_LEN = 6
    FILE_PREFIX = 'file-'
    FILE_EXT = '.rio'

    def __init__(self, generator=None, dir_=None):
        if dir_ and not os.path.exists(dir_):
            os.makedirs(dir_)
        self._dir = dir_ or tempfile.mkdtemp()
        self._gen = generator or TestExampleGenerator()

    def _write(self, fpath, num_examples):
        with tf.python_io.TFRecordWriter(fpath) as writer:
            for _ in range(num_examples):
                example = data.encode(*self._gen.next())
                writer.write(example.SerializeToString())

    def generate(self, num_files=1, num_examples=10):
        """Generate `num_files` data files, each with `num_examples` examples."""
        files = []
        for i in range(num_files):
            fname = self.FILE_PREFIX + str(i) + self.FILE_EXT
            fpath = os.path.join(self._dir, fname)
            self._write(fpath, num_examples)
            files.append(fpath)
        return files

    def cleanup(self):
        """Cleanup."""
        shutil.rmtree(self._dir)