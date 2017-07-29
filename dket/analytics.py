"""Analytics tool for evaluation dump of dket experiments."""

import collections
import logging
import json
import os

import edit_distance as editdist

from liteflow import vocabulary as lvoc

from dket.logutils import HDEBUG
from dket import metrics


def load_vocabulary(fpath):
    """Load a LiTeFlow vocabulary from a file."""
    logging.debug('creating an in-memory vocabulary from file: %s', fpath)
    voc = lvoc.InMemoryVocabulary()
    for line in open(fpath):
        word = line.replace('\n', '')
        if word:
            logging.log(HDEBUG, 'adding word: %s', word)
            voc.add(word)
    logging.info('%d symbols loaded from %s', voc.size(), fpath)
    return voc


def decode_sentence(idxs, vocabulary):
    """Decode a list of words indexes into a sentence."""
    return [vocabulary.word(idx) for idx in idxs]


def decode_formula(idxs, shortlist, sentence):
    """Decode a list of formula term indexes into a formula."""
    formula = []
    for idx in idxs:
        if idx < shortlist.size():
            formula.append(shortlist.word(idx))
        else:
            pointer = idx - shortlist.size()
            word = sentence[pointer]
            formula.append(word)
    return formula


def edit_distance(target, prediction):
    """The edit distance score and op codes."""
    distance_t = editdist.edit_distance(prediction, target)
    if isinstance(distance_t, tuple):
        logging.log(
            HDEBUG, 'edit distance: %s --> picking up the first element: %s',
            str(distance_t), distance_t[0])
        distance = distance_t[0]
    else:
        distance = distance_t

    bdistance, _, opcodes = editdist.edit_distance_backpointer(prediction, target)
    if distance != bdistance:
        logging.warning('Edit distance: ' + str(distance) + ' (' + str(distance_t) + ')')
        logging.warning('Edit distance backpointed: ' + str(bdistance))
        logging.warning('TARGET: %s', str(target))
        logging.warning('PREDICTION: %s', str(prediction))
        for opcode in opcodes:
            logging.warning(serialize_diff_op(opcode, target, prediction))
    return bdistance, opcodes


def serialize_diff_op(diff_op, target, prediction):
    """Applies a diff op for editing one sequence into another."""
    fmt = '{:7}   T[{}:{}] -> P[{}:{}] {} -> {}'
    tag, i1, i2, j1, j2 = tuple(diff_op)  # pylint: disable=C0103
    serialized = fmt.format(tag, i1, i2, j1, j2, target[i1:i2], prediction[j1:j2])
    return serialized


TAB = '\t'
BLANK = ' '
EXAMPLE = 'example'
DUMP = 'dump'
SENTENCE = 'sentence'
TARGET = 'target'
PREDICTION = 'prediction'
IDX = '_idx'
ACCURACY = 'accuracy'
EDIT_DISTANCE = 'edit_distance'
EDIT_SCORE = 'score'
EDIT_DIFF_OPS = 'diff_ops'
ITEM_SEP = '\n\n'

def parse(tsv_line):
    """Parses a TSV dump line into 3 list of int indexes."""
    sentence, target, prediction = tuple(tsv_line.split(TAB))
    sentence = [int(item) for item in sentence.split(BLANK)]
    target = [int(item) for item in target.split(BLANK)]
    prediction = [int(item) for item in prediction.split(BLANK)]
    return sentence, target, prediction


class _NoIndent(object):

    def __init__(self, value):
        self.value = value


class _NoIndentEncoder(json.JSONEncoder):

    def default(self, o):  # pylint: disable=E0202
        if isinstance(o, _NoIndent):
            return str(o.value)
        else:
            return super(_NoIndentEncoder, self).default(o)


def convert(tsv_line, vocabulary, shortlist):
    """Returns a serialized json object with the analytics for the TSV dump line."""
    sentence, target, prediction = parse(tsv_line)
    dump = collections.OrderedDict()
    dump[SENTENCE + IDX] = _NoIndent(sentence)
    dump[TARGET + IDX] = _NoIndent(target)
    dump[PREDICTION + IDX] = _NoIndent(prediction)

    accuracy = metrics.per_token_accuracy(target, prediction)

    sentence = decode_sentence(sentence, vocabulary)
    target = decode_formula(target, shortlist, sentence)
    prediction = decode_formula(prediction, shortlist, sentence)
    example = collections.OrderedDict()
    example[SENTENCE] = _NoIndent(sentence)
    example[TARGET] = _NoIndent(target)
    example[PREDICTION] = _NoIndent(prediction)

    distance, ops = edit_distance(target, prediction)

    ed_data = collections.OrderedDict()
    ed_data[EDIT_SCORE] = distance
    ed_data[EDIT_DIFF_OPS] = [serialize_diff_op(op, target, prediction) for op in ops]

    data = collections.OrderedDict()
    data[EXAMPLE] = example
    data[DUMP] = dump
    data[ACCURACY] = round(accuracy, 3)
    data[EDIT_DISTANCE] = ed_data

    return json.dumps(data, indent=2, separators=(',', ': '), cls=_NoIndentEncoder)


def process(dump_fp, vocabulary_fp, shortlist_fp, data_fp=None, force=False):
    """Process a dump file."""
    if not dump_fp:
        raise ValueError('A dump file must be provided.')

    if not os.path.exists(dump_fp):
        raise FileNotFoundError('The dump file {} does not exist.'.format(data_fp))

    if not data_fp:
        data_fp = dump_fp + '.data'

    if os.path.exists(data_fp) and not force:
        raise FileExistsError('The output file {} already exist.'.format(data_fp))

    vocabulary = load_vocabulary(vocabulary_fp)
    shortlist = load_vocabulary(shortlist_fp)

    with open(data_fp, 'w') as fout:
        fout.write('# ANALYTICS\n')
        fout.write('# SOURCE: ' + str(dump_fp) + '\n')
        fout.write('# VOCABULARY: ' + str(vocabulary_fp) + '\n')
        fout.write('# SHORTLIST: ' + str(shortlist_fp) + '\n')
        fout.write('\n')
        for tsv_line in open(dump_fp):
            tsv_line = tsv_line.replace('\n', '')
            if tsv_line:
                fout.write(convert(tsv_line, vocabulary, shortlist) + ITEM_SEP)
