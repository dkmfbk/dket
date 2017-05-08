"""Data management for the `dket` package.

A `dket` example is made of essentially two sequence of symbols:
  - a sentence, i.e. a sequence of words, as input;
  - a formula, i.e. a sequence of terms, as output.
Such examples should be persisted as `tf.train.Example` protobuf
message having two int64 scalar features:
  - `sentence_length`: the length of the input sentence;
  - `formula_length`: the length of the output formula;
and has two int64 list features:
  - `words`: a list of single valued `int64_list` with the index
    values for the words of the input sentence;
  - `formula`: a list of single valued `int64_list` with the index
     values for the terms of the output formula.
"""

import tensorflow as tf


SENTENECE_LENGTH_KEY = 'sentence_length'
FORMULA_LENGTH_KEY = 'formula_length'
WORDS_KEY = 'words'
FORMULA_KEY = 'formula'


def encode(words_idxs, formula_idxs):
    """Encode a list of word and formula terms into a tf.train.SequenceExample.

    Arguments:
      words_idx: `list` of `int` or 1D numpy array representing the index values
        for the words in the input sentence of an example.
      formula_idx: `list` of `int` or 1D numpy array representing the index values
        for all the terms of the output formula of an example.

    Returns:
      a `tf.train.Example` to be consumed by the dket architecture.
        It has two int64 scalar features:
          - `sentence_length`: the length of the input sentence;
          - `formula_length`: the length of the output formula;
        and has two int64 list features:
          - `words`: a list with the index values for the words of the input sentence;
          - `formula`: a list with the index values for the terms of the output formula.

    Example:
    >>> import tensorflow as tf
    >>> from dket import data
    >>> words_idxs = [1, 2, 3, 0]
    >>> formula_idxs = [12, 23, 34, 45, 0]
    >>> print data.encode(input_, output)

    features {
        feature {
            key: "formula"
            value {
                int64_list {
                    value: 12
                    value: 23
                    value: 34
                    value: 45
                    value: 0
                    }
                }
            }
        }
        feature {
            key: "formula_length"
            value {
                int64_list {
                    value: 5
                }
            }
        }
        feature {
            key: "sentence_length"
            value {
                int64_list {
                    value: 4
                }
            }
        }
        feature {
            key: "words"
            value {
                int64_list {
                    value: 1
                    value: 2
                    value: 3
                    value: 0
                }
            }
        }
    }
    """
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                SENTENECE_LENGTH_KEY: tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[len(words_idxs)])),
                FORMULA_LENGTH_KEY: tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=[len(formula_idxs)])),
                WORDS_KEY: tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=words_idxs)),
                FORMULA_KEY: tf.train.Feature(
                    int64_list=tf.train.Int64List(
                        value=formula_idxs)),
            }
        )
    )
    return example


def parse(serialized):
    """Parse a string into tensors.

    Arguments:
      example: a serialized `tf.train.SequenceExample` (like the one returned
        from the `encode()` method).

    Returns:
      a pair of 1D ternsors, `words` of shape [sentence_length] and
        `formula` of shape [formula_length].
    """
    features = {
        SENTENECE_LENGTH_KEY: tf.FixedLenFeature([], tf.int64),
        FORMULA_LENGTH_KEY: tf.FixedLenFeature([], tf.int64),
        WORDS_KEY: tf.VarLenFeature(tf.int64),
        FORMULA_KEY: tf.VarLenFeature(tf.int64),
    }
    parsed = tf.parse_single_example(
        serialized=serialized,
        features=features)
    words = tf.sparse_tensor_to_dense(parsed[WORDS_KEY])
    formula = tf.sparse_tensor_to_dense(parsed[FORMULA_KEY])
    return words, formula
