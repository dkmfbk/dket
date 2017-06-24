"""Run your dket experiment."""
# TODO(petrux): review all the log entries.

import logging
import os

import tensorflow as tf

from liteflow import input as lin
from liteflow import metrics
from liteflow import losses

from dket.runtime import logutils
from dket.runtime import runtime
from dket.models import pointsoftmax
from dket import data
from dket import ops
from dket import optimizers
from logutils import HDEBUG


BASE = os.path.dirname(os.path.realpath(__file__))

# pylint: disable=C0301
tf.app.flags.DEFINE_string('model-name', None, 'The model name.')
tf.app.flags.DEFINE_integer('batch-size', 32, 'The batch size.')
tf.app.flags.DEFINE_string('hparams', None, 'The hparams for the model instance. A comma-separated list of key=value pairs or a path to a file with one key=value pair per line.')
tf.app.flags.DEFINE_string('data-dir', None, 'The directory where the data files are. If omitted, the local directory will be used.')
tf.app.flags.DEFINE_string('data-files', None, 'A comma separated list of data file patterns (e.g. `file-*.txt`). If the --data-dir is provided, file patterns can be relative to that directory. If no value is provided, [DATA-DIR]/[MODE]*.* is the default.')

tf.app.flags.DEFINE_string('mode', 'train', 'The execution mode: can be train, eval, test.')
tf.app.flags.DEFINE_integer('epochs', None, 'The number of training epochs. If none, only the [STEPS] value will be considered.')
tf.app.flags.DEFINE_integer('steps', None, 'The number of training steps. If none, only the [EPOCHS] value will be considered.')
tf.app.flags.DEFINE_integer('checkpoint-every-steps', 100, 'Number of training steps after which the model state is saved. This flag is considered only in --mode=train.')
tf.app.flags.DEFINE_integer('eval-check-every-sec', 300, 'Time interval in seconds. If --mode=eval/test, the [LOG-DIR] is periodically checked to see if there is a new checkpoint to evaluate.')
tf.app.flags.DEFINE_integer('eval-check-until-sec', 3600, 'Time interval in seconds. If --mode=eval/test, the maximum amount of time to wait for a new model checkpoint to appear in [LOG-DIR] before ending the evaluation process.')
tf.app.flags.DEFINE_integer('eval-max-global-step', None, 'The maximum global step checkpoint to evaluate. Works only if --mode=eval/test.')

tf.app.flags.DEFINE_float('lr', 0.1, 'The initial value for the learning rate.')
tf.app.flags.DEFINE_float('lr-decay-rate', None, 'The decay rate for the learning rate.')
tf.app.flags.DEFINE_float('lr-decay-steps', None, 'The decay steps for the learning rate.')
tf.app.flags.DEFINE_boolean('lr-decay-staircase', True, 'If set, the learning rate decay is staircase-like.')
tf.app.flags.DEFINE_string('optimizer', 'sgd', 'The optimizer name. Can be one of sgd, adagrad, adadelta, adam.')
tf.app.flags.DEFINE_float('adagrad-epsilon', None, 'If --optimizer=adagrad, set the epsilon parameter.')
tf.app.flags.DEFINE_float('adagrad-rho', None, 'If --optimizer=adagrad, set the rho parameter.')

tf.app.flags.DEFINE_string('device', None, 'Force the model to be built on a certain type of device (CPU or GPU). If not set, the device will be chosen according to the mode using GPU for training and CPU for evaluation.')
tf.app.flags.DEFINE_integer('threads', 4, 'The number of threads to be used for the input pipeline.')

tf.app.flags.DEFINE_string('base-log-dir', '.', 'The base log directory where all the model dumps, summaries, ecc. will be stored in [BASE-LOG-DIR]/[MODE]. If not set, the current directory will be taken.')
tf.app.flags.DEFINE_string('log-level', 'INFO', 'The log level. Can be none ore one of HDEBUG, DEBUG, INFO, WARNING')
tf.app.flags.DEFINE_string('log-file', 'log', 'The log file as file name or as an absolute path, if not specified, it is just `log`. If not an absolute path, it will be placed in [BASE-LOG-DIR]/[MODE]/[LOG-FILE]. If the file already exists, new log entries will be appended.')
tf.app.flags.DEFINE_boolean('log-to-stderr', False, 'If set, redirect also the log entries with level less than WARNING to the standard error stream.')
# pylint: enable=C0301

FLAGS = tf.app.flags.FLAGS

def _debug_log_flags():
    logging.debug('command line flags.')
    for key, value in FLAGS.__flags.items():  # pylint: disable=I0011,W0212
        logging.debug('--%s %s', key, str(value))


_MODE_TRAIN = 'train'
_MODE_EVAL = 'eval'
_MODE_TEST = 'test'

def _validate_mode():
    mode = FLAGS.mode
    if mode == _MODE_TRAIN:
        return mode
    if mode == _MODE_EVAL:
        return mode
    if mode == _MODE_TEST:
        return mode
    raise ValueError(
        'Invalid mode. Must bee one of %s, %s, %s. Found `%s` instead'
        % (_MODE_TRAIN, _MODE_EVAL, _MODE_TEST, mode))


def _get_log_dir():
    return os.path.join(FLAGS.base_log_dir, FLAGS.mode)


def _setup_log_dir():
    log_dir = _get_log_dir()
    logging.info('logging directory: %s', log_dir)
    if not os.path.exists(log_dir):
        logging.debug('creating the directory: %s', log_dir)
        os.makedirs(log_dir)
    return log_dir


def _setup_logging():
    log_level = logutils.parse_level(FLAGS.log_level)
    log_file = os.path.join(_get_log_dir(), FLAGS.log_file)
    log_to_stderr = FLAGS.log_to_stderr
    logutils.config(level=log_level, fpath=log_file, stderr=log_to_stderr)
    logging.debug('logging infrastructure configured.')
    logging.debug('setting TF log level to 9')
    tf.logging.set_verbosity(9)


def _get_model_type():
    if not FLAGS.model_name:
        message = 'the model name MUST be specified.'
        logging.critical(message)
        raise ValueError(message)

    logging.debug('loading the model class for name: %s.', FLAGS.model_name)
    if FLAGS.model_name == 'pointsoftmax':
        return pointsoftmax.PointingSoftmaxModel

    message = 'Invalid model name {}'.format(FLAGS.model_name)
    logging.critical(message)
    raise ValueError(message)


_GPU = 'GPU'
_CPU = 'CPU'

def _get_device():
    device = FLAGS.device
    if not device:
        logging.debug('device not set: trying to set it for the mode.')
        if FLAGS.mode == _MODE_TRAIN:
            logging.debug('setting %s device for mode %s', _GPU, FLAGS.mode)
            device = _GPU
        elif FLAGS.mode == _MODE_EVAL or FLAGS.mode == _MODE_TEST:
            logging.debug('setting %s device for mode %s', _CPU, FLAGS.mode)
            device = _CPU
        else:
            message = 'cannot set a default device for mode {}'.format(device)
            logging.critical(message)
            raise ValueError(message)
    return device

def _get_hparams(dhparams):
    logging.log(HDEBUG, 'deault params:')
    for key, value in dhparams.values().items():
        logging.log(HDEBUG, 'default hparams.' + key + '=' + str(value))

    if not FLAGS.hparams:
        logging.info('no hparams set: using default.')
        return dhparams

    hparams = FLAGS.hparams
    if isinstance(hparams, str):
        if os.path.exists(hparams):
            logging.info('trying to parse hparams from file %s', hparams)
            hparams = ','.join([line.replace('\n', '') for line in open(hparams)])
    logging.debug('serialized hparams: %s', hparams)
    logging.debug('parsing hparams.')
    hparams = dhparams.parse(hparams)
    logging.debug('model will be configured with the following hparams:')
    for key, value in hparams.values().items():
        logging.debug('hparams.' + key + '=' + str(value))
    return hparams


def _get_epochs_and_steps():
    logging.debug('getting epochs and steps.')
    epochs = FLAGS.epochs
    if epochs is None:
        logging.debug('epochs: None')
    else:
        logging.debug('epochs: %d', epochs)

    steps = FLAGS.steps
    if steps is None:
        logging.debug('steps: None -- setting to 0')
        steps = 0
    else:
        logging.debug('steps: %d', steps)

    if not epochs and not steps:
        message = 'at least one of epochs or steps must be set.'
        logging.critical(message)
        raise ValueError(message)
    return epochs, steps


def _get_data_files():
    logging.debug('getting data files.')
    data_dir = FLAGS.data_dir
    data_files = FLAGS.data_files
    if not data_dir and not data_files:
        message = 'Both data directory and data files are missing.'
        logging.critical(message)
        raise ValueError(message)

    data_files = data_files.split(',')
    if not data_dir:
        return data_files

    data_abs_files = []
    for data_file in data_files:
        if os.path.isabs(data_file):
            logging.debug('absolute path data file: %s', data_file)
            data_abs_files.append(data_file)
        else:
            logging.debug('relative path data file [%s/]%s', data_dir, data_file)
            data_abs_files.append(os.path.join(data_dir, data_file))
    return data_abs_files


def _get_feed_dict():
    logging.debug('getting the feed dictionary.')
    # TODO(petrux): shuffle only in training. This can be HUGE to implement since
    # the liteflow component is made for shuffling and batching. So maybe a regular
    # TF component should be used.
    data_files = _get_data_files()
    shuffle = _validate_mode() == _MODE_TRAIN
    logging.debug('suffling.' if shuffle else 'not shuffling.')
    epochs, _ = _get_epochs_and_steps()
    if epochs is not None:
        logging.debug('epochs: %d', epochs)
    logging.debug('reading from data.')
    tensors = data.read_from_files(data_files, shuffle=shuffle, num_epochs=epochs)
    logging.debug('got %d tensors.', len(tensors))
    logging.debug('reading shuffled and batched and padded tensors.')
    tensors = lin.shuffle_batch(tensors, FLAGS.batch_size)
    logging.debug('got %d tensors.', len(tensors))
    feed_dict = {
        data.WORDS_KEY: tf.cast(tensors[0], tf.int32),
        data.SENTENCE_LENGTH_KEY: tf.cast(tensors[1], tf.int32),
        data.FORMULA_KEY: tf.cast(tensors[2], tf.int32),
        data.FORMULA_LENGTH_KEY: tf.cast(tensors[3], tf.int32)
    }
    for key, value in feed_dict.items():
        logging.debug('%s: %s', key, str(value))
    return feed_dict


def _get_loss():
    logging.debug('getting the loss function')
    loss = losses.StreamingLoss(
        func=losses.categorical_crossentropy,
        name='XEntropy')
    return loss


_SGD = 'sgd'
_ADAM = 'adam'
_ADADELTA = 'adadelta'
_ADAGRAD = 'adagrad'

def _get_optimizer():
    logging.debug('getting the optimizer')
    lr = FLAGS.lr  # pylint: disable=I0011,C0103
    lr_decay_rate = FLAGS.lr_decay_rate
    lr_decay_steps = FLAGS.lr_decay_steps
    lr_decay_staircase = FLAGS.lr_decay_staircase
    if (lr_decay_rate is None) ^ (lr_decay_steps is None):
        message = 'lr-decay-rate and lr-decay-step must be both set or both `None`.'
        logging.critical(message)
        raise ValueError(message)
    if lr_decay_steps is not None:
        logging.debug('configuring exponential lr decay.')
        logging.debug('getting the global step')
        global_step = ops.get_global_step()
        # pylint: disable=I0011,C0103
        lr = tf.train.exponential_decay(
            learning_rate=lr,
            global_step=global_step,
            decay_steps=lr_decay_steps,
            decay_rate=lr_decay_rate,
            staircase=lr_decay_staircase)

    if FLAGS.optimizer == _SGD:
        return optimizers.Optimizer.sgd(learning_rate=lr)
    else:
        message = 'invalid optimizer name: `{}`.'.format(FLAGS.optimizer)
        logging.critical(message)
        raise ValueError(message)


def _get_metrics_dict():
    logging.debug('getting evaluation metrics.')
    logging.debug('creating (per token) accuracy metric.')
    acc = metrics.StreamingMetric(
        metrics.accuracy, name='Accuracy')
    logging.debug('creating per sentence accuracy metric.')
    psa = metrics.StreamingMetric(
        metrics.per_sentence_accuracy, name='PerSentenceAccuracy')
    return {
        acc.name: acc,
        psa.name: psa
    }

def _get_loop(model):
    mode = _validate_mode()
    _, steps = _get_epochs_and_steps()
    if mode == _MODE_TRAIN:
        loop = runtime.TrainLoop(
            model=model,
            log_dir=_get_log_dir(),
            steps=steps,
            checkpoint_every=FLAGS.checkpoint_every_steps)
        return loop
    if mode == _MODE_EVAL or mode == _MODE_TEST:
        return runtime.EvalLoop(
            model=model,
            log_dir=_get_log_dir(),
            checkpoint_dir=os.path.join(FLAGS.base_log_dir, _MODE_TRAIN),
            steps=steps,
            eval_check_every_secs=FLAGS.eval_check_every_sec,
            eval_check_until_secs=FLAGS.eval_check_until_sec)
    raise ValueError('Invalid mode: ' + mode)


def _build_model():
    logging.info('getting the device type to be used.')
    device = _get_device()
    logging.info('the device type to be used is %s', device)

    logging.info('getting the model factory.')
    mtype = _get_model_type()

    with tf.Graph().as_default() as graph:
        with tf.device(device):
            logging.debug('instantiating the model')
            model = mtype(graph=graph)

            logging.debug('getting the feed dictionary.')
            feed_dict = _get_feed_dict()

            logging.debug('getting default hparams for the model.')
            dhparams = mtype.get_default_hparams()
            hparams = _get_hparams(dhparams)

            logging.debug('getting the loss function.')
            loss = _get_loss()

            logging.debug('getting the optimizer.')
            optimizer = _get_optimizer()

            logging.debug('getting the metrics')
            metrics_dict = _get_metrics_dict()

            logging.debug('feeding the model')
            model = model.feed(feed_dict)

            logging.debug('building the model.')
            model = model.build(hparams, loss, optimizer, metrics_dict)

            logging.debug('getting the runtime loop.')
            loop = _get_loop(model)
            logging.debug('starting the runtime loop.')
            loop.start()
            logging.debug('runtime loop complete.')

def main(_):
    """Main application entry point."""
    _validate_mode()
    _setup_log_dir()
    _setup_logging()
    _debug_log_flags()
    _build_model()

if __name__ == '__main__':
    tf.app.run()
