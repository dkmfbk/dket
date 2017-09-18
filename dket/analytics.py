"""Analytics tool for evaluation dump of dket experiments."""

import collections
import json
import logging
import operator
import os

import edit_distance as editdist
from liteflow import vocabulary as lvoc

from dket import metrics
from dket.logutils import HDEBUG


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
    """Decode a list of words indexes into a sentence.

    If a word is <UNK> it is decoded with <UNK>@[pos] where
    `pos` is the 0-based position of the word within the sentence.
    """
    words = []
    for pos, idx in enumerate(idxs):
        word = vocabulary.word(idx)
        if word == lvoc.UNKVocabulary.UNK:
            word = word + '@' + str(pos)
        words.append(word)
    return words



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
    matcher = editdist.SequenceMatcher(a=prediction, b=target)
    distance = matcher.distance()
    ops = matcher.get_opcodes()
    return distance, ops


def serialize_diff_op(diff_op, target, prediction):
    """Applies a diff op for editing one sequence into another."""
    fmt = '{:7}   PRED[{}:{}] -> TAR[{}:{}] {} -> {}'
    tag, i1, i2, j1, j2 = tuple(diff_op)  # pylint: disable=C0103
    serialized = fmt.format(tag, i1, i2, j1, j2, prediction[i1:i2], target[j1:j2])
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
ITEM_SEP = '\n\n\n'
EOS = '<EOS>'
EOS_IDX = 0
TOKENS_TOT = 'tokens_tot'
TOKENS_OK = 'tokens_OK'


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


def unpad(sequence, padding=None):
    """Removes padding symbols after first occurrence."""
    if not sequence:
        return sequence
    try:
        end = sequence.index(padding)
        return sequence[:end + 1]
    except ValueError as _:
        logging.warning('No padding symbol %s found in %s', str(padding), str(sequence))
        return sequence


def samelength(first, second, padding=None):
    """Makes the two sequences of the same length."""
    if not first and not second == 0:
        return first, second

    if len(first) == len(second):
        return first, second

    if padding is None:
        # infer padding: must be the last item all the non-empty sequences.
        if first and second:
            if first[-1] != second[-1]:
                raise ValueError("""The two sequences must end with the same symbol """
                                 """or the `padding` should be provided.""")
            padding = first[-1]
        if not first:
            first = []
            padding = second[-1]
        if not second:
            second = []
            padding = first[-1]

    if len(first) < len(second):
        return (first + ([padding] * (len(second) - len(first)))), second
    return first, (second + ([padding] * (len(first) - len(second))))


def convert(tsv_line, vocabulary, shortlist, equals=False):
    """Returns a serialized json object with the analytics for the TSV dump line."""
    sentence, target, prediction = parse(tsv_line)

    sentence = unpad(sentence, EOS_IDX)
    target = unpad(target, EOS_IDX)
    prediction = unpad(prediction, EOS_IDX)

    target_, prediction_ = samelength(target, prediction, padding=EOS_IDX)
    tokens_tot = len(target_)
    tokens_ok = 0
    for i in range(len(target_)):
        if target_[i] == prediction_[i]:
            tokens_ok += 1

    dump = collections.OrderedDict()
    dump[SENTENCE + IDX] = _NoIndent(sentence)
    dump[TARGET + IDX] = _NoIndent(target_)
    dump[PREDICTION + IDX] = _NoIndent(prediction_)

    accuracy = metrics.per_token_accuracy(target_, prediction_)

    sentence = decode_sentence(sentence, vocabulary)
    target = decode_formula(target, shortlist, sentence)
    prediction = decode_formula(prediction, shortlist, sentence)
    example = collections.OrderedDict()
    example[SENTENCE] = ' '.join(sentence)
    example[TARGET] = ' '.join(target)
    example[PREDICTION] = ' '.join(prediction)

    distance, ops = edit_distance(target, prediction)
    if not equals:
        ops = [op for op in ops if op[0] != 'equal']

    ed_data = collections.OrderedDict()
    ed_data[EDIT_SCORE] = distance
    ed_data[EDIT_DIFF_OPS] = [serialize_diff_op(op, target, prediction) for op in ops]

    data = collections.OrderedDict()
    data[EXAMPLE] = example
    data[DUMP] = dump
    data[ACCURACY] = round(accuracy, 3)
    data[TOKENS_OK] = tokens_ok
    data[TOKENS_TOT] = tokens_tot
    data[EDIT_DISTANCE] = ed_data
    return data
#    return json.dumps(data, indent=2, separators=(',', ': '), cls=_NoIndentEncoder)


def process(dump_fp, vocabulary_fp, shortlist_fp, data_fp=None, force=False, equals=False):
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

    # Collect all the data as dictionary objects
    data = []
    for tsv_line in open(dump_fp):
        tsv_line = tsv_line.replace('\n', '')
        if tsv_line:
            data.append(convert(tsv_line, vocabulary, shortlist, equals=equals))
    data = sorted(data, key=lambda datum: datum[EDIT_DISTANCE][EDIT_SCORE])
    correct = [datum for datum in data if datum[EDIT_DISTANCE][EDIT_SCORE] == 0]
    edists = [datum[EDIT_DISTANCE][EDIT_SCORE] for datum in data]
    pfacc = len(correct) * 1.0 / len(data)
    avgedists = sum(edists) * 1.0 / len(data)

    edstats = collections.defaultdict(lambda: 0)
    for datum in data:
        edist = datum[EDIT_DISTANCE][EDIT_SCORE]
        edstats[edist] = edstats[edist] + 1
    tuples = [(k, v) for k, v in edstats.items()]
    tuples = sorted(tuples, key=lambda t: t[0])

    tokens_tot = 0
    tokens_ok = 0
    for datum in data:
        tokens_tot += datum[TOKENS_TOT]
        tokens_ok += datum[TOKENS_OK]
    accuracy = 0.0 if tokens_tot == 0 else (tokens_ok * 1.0) / tokens_tot

    with open(data_fp, 'w') as fout:
        fout.write('# ANALYTICS\n')
        fout.write('# SOURCE: ' + str(dump_fp) + '\n')
        fout.write('# VOCABULARY: ' + str(vocabulary_fp) + '\n')
        fout.write('# SHORTLIST: ' + str(shortlist_fp) + '\n')
        fout.write('#\n')
        fout.write('# AVG. PER-FORMULA ACCURACY: {:.5f}'.format(pfacc) + '\n')
        fout.write('# AVG. EDIT DISTANCE: {:.5f}'.format(avgedists) + '\n')
        fout.write('# AVG. PER-TOKEN ACCURACY: {:.5f}'.format(accuracy) + '\n')
        fout.write('#\n')
        partial = 0.0
        fout.write('# ED\t#\t%\t% INC\n')
        for ed, count in tuples:
            partial += count * 1.0
            fout.write("# {}\t{}\t{:.2f}\t{:.2f}\n".format(
                ed, count, count * 1.0 / len(data) * 100, (partial / len(data)) * 100))
        fout.write(ITEM_SEP)
        for datum in data:
            line = json.dumps(datum, indent=2, separators=(',', ': '), cls=_NoIndentEncoder)
            fout.write(line + ITEM_SEP)
