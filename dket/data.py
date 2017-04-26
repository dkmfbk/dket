"""Data management for the `dket` package.

A `dket` example is made of essentially two sequence of symbols:
  - a sentence, i.e. a sequence of words, as input;
  - a formula, i.e. a sequence of terms, as output.
Such examples should be persisted as `tf.train.SequenceExample` protobuf
messages having two int64 scalar (context) fields:
  - `sentence_length`: the length of the input sentence;
  - `formula_length`: the length of the output formula;
and has two feature lists:
  - `words`: a list of single valued `int64_list` with the index
    values for the words of the input sentence;
  - `formula`: a list of single valued `int64_list` with the index
     values for the terms of the output formula.

NOTA BENE: the persistence schema could rapidly evolve.
"""

import tensorflow as tf


SENTENECE_LENGTH_KEY = 'sentence_length'
FORMULA_LENGTH_KEY = 'formula_length'
WORDS_KEY = 'words'
FORMULA_KEY = 'formula'

def encode(words_idxs, formula_idxs):
    """Encode a list of word and formula terms into a tf.train.SequenceExample.

    Arguments:
      words_idx: `list` of `int` representing the index values for the words
        in the input sentence of an example.
      formula_idx: `list` of `int` representing the index values for all the
        terms of the output formula of an example.

    Returns:
      a `tf.train.SequenceExample` to be consumed by the dket architecture.
        It has two int64 scalar (context) fields:
          - `sentence_length`: the length of the input sentence;
          - `formula_length`: the length of the output formula;
        and has two feature lists:
          - `words`: a list of single valued `int64_list` with the index
            values for the words of the input sentence;
          - `formula`: a list of single valued `int64_list` with the index
            values for the terms of the output formula.

    Example:
    >>> import tensorflow as tf
    >>> from dket import data
    >>> input_ = [1, 2, 3, 0]
    >>> output = [23, 34, 45, 0]
    >>> print encode(input_, output)

    context {
        feature {
            key: "formula_length"
            value {
                int64_list {
                    value: 4
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
    }
    feature_lists {
        feature_list {
            key: "formula"
            value {
                feature {
                    int64_list {
                        value: 23
                    }
                }
                feature {
                    int64_list {
                        value: 34
                    }
                }
                feature {
                    int64_list {
                        value: 45
                    }
                }
                feature {
                    int64_list {
                        value: 0
                }
            }
        }
    }
    """

    example = tf.train.SequenceExample(
        context=tf.train.Features(
            feature={
                'sentence_length': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[len(words_idxs)])),
                'formula_length': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[len(formula_idxs)]))
            }),
        feature_lists=tf.train.FeatureLists(
            feature_list={
                'words': tf.train.FeatureList(
                    feature=[
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[item]))
                        for item in words_idxs]),
                'formula': tf.train.FeatureList(
                    feature=[
                        tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[item]))
                        for item in formula_idxs])
            }
        )
    )
    return example


def decode(example):
    """Decode a `tf.train.SequenceExample` protobuf message.

    Arguments:
      example: a `tf.train.SequenceExample` like the one returned
        from the `encode()` method.

    Returns:
      a pair of 2-D ternsors, `words` of shape [sentence_length, 1] and
        `formula` of shape [formula_length, 1].
    """

    context_features = {
        SENTENECE_LENGTH_KEY: tf.FixedLenFeature([], tf.int64),
        FORMULA_LENGTH_KEY: tf.FixedLenFeature([], tf.int64)
    }
    sequence_features = {
        WORDS_KEY: tf.FixedLenSequenceFeature([], tf.int64),
        FORMULA_KEY: tf.FixedLenSequenceFeature([], tf.int64)
    }
    _, sequence = tf.parse_single_sequence_example(
        serialized=example.SerializeToString(),
        context_features=context_features,
        sequence_features=sequence_features)
    words = sequence[WORDS_KEY]
    formula = sequence[FORMULA_KEY]
    return words, formula
