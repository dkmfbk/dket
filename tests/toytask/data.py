"""Data generation for the toy task."""

import os
import random
import tempfile

import tensorflow

import dket.data

# pylint: disable=C0301
tensorflow.app.flags.DEFINE_integer('size', 10000, 'The number of examples to be generated.')
tensorflow.app.flags.DEFINE_string('output', None, 'The output file. If not set, a temporary one will be used')
tensorflow.app.flags.DEFINE_integer('seed', 23, 'The random seed to be used.')
FLAGS = tensorflow.app.flags.FLAGS
# pylint: enable=C0301


def example(min_len=20, max_len=30, eos=0, min_shortlist=1,
            min_discard=21, min_point=41, max_idx=99):
    """Generate a new example."""
    shortlist_size = min_discard - min_shortlist + 1  # consider eos
    words = [random.randint(min_shortlist, max_idx)
             for _ in range(random.randint(min_len, max_len))]
    formula = []
    for pos, idx in enumerate(words):
        if min_shortlist <= idx < min_discard:
            formula.append(idx)
        elif min_discard <= idx < min_point:
            pass  # discard the symbol.
        else:
            formula.append(shortlist_size + pos)
    # append eos
    words.append(eos)
    formula.append(eos)
    return words, formula


def generate_dataset(size, fpath=None, factory=example):
    """Generate a dataset as TFRecords consumable by a DketModel.

    Arguments:
      size: `int` representing the number of exampels.
      fpath: the file to write the data to; if None, a temporary directory with a file
        named `dataset.rio` will be created an returned.
      factory: a function generating a words, formula pair of `int` list representing
        the indexes of words and formula terms of a new example. If not provided, the
        `example` function with default parameters will be used.
    """
    if not fpath:
        tmpdir = tempfile.mkdtemp()
        fpath = os.path.join(tmpdir, 'dataset.rio')
    with tensorflow.python_io.TFRecordWriter(fpath) as writer:
        for _ in range(size):
            words, formula = factory()
            writer.write(
                dket.data.encode(words, formula)
                .SerializeToString())
    return fpath


if __name__ == '__main__':
    random.seed(FLAGS.seed)
    generate_dataset(FLAGS.size, FLAGS.output)
