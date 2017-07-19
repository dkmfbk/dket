"""Data generation for the toy task."""

import os
import random
import tempfile

import tensorflow

import dket.data
import tests.utils 

# pylint: disable=C0301
tensorflow.app.flags.DEFINE_integer('size', 10000, 'The number of examples to be generated.')
tensorflow.app.flags.DEFINE_string('output', None, 'The output file. If not set, a temporary one will be used')
tensorflow.app.flags.DEFINE_integer('seed', 23, 'The random seed to be used.')
FLAGS = tensorflow.app.flags.FLAGS
# pylint: enable=C0301


class _ExGen(tests.utils.TestExampleGenerator):
    
    def __init__(self):
        self.min_len=20
        self.max_len=30
        self.eos=0
        self.min_shortlist=1
        self.min_discard=21
        self.min_point=41
        self.max_idx=99

    def next(self):
        """Generate a new example."""
        shortlist_size = self.min_discard - self.min_shortlist + 1  # consider eos
        words = [random.randint(self.min_shortlist, self.max_idx)
                for _ in range(random.randint(self.min_len, self.max_len))]
        formula = []
        for pos, idx in enumerate(words):
            if self.min_shortlist <= idx < self.min_discard:
                formula.append(idx)
            elif self.min_discard <= idx < self.min_point:
                pass  # discard the symbol.
            else:
                formula.append(shortlist_size + pos)
        # append eos
        words.append(self.eos)
        formula.append(self.eos)
        return words, formula

def generate_dataset(size, fpath=None, factory=None):
    """Generate a dataset as TFRecords consumable by a DketModel.

    Arguments:
      size: `int` representing the number of exampels.
      fpath: the file to write the data to; if None, a temporary directory with a file
        named `dataset.rio` will be created an returned.
      factory: a function generating a words, formula pair of `int` list representing
        the indexes of words and formula terms of a new example.
    """
    if not factory:
        generator = _ExGen()
        factory = generator.next
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
