"""Run your dket experiment."""

import logging
import os
import sys

import tensorflow as tf

from liteflow import input as lin
from liteflow import metrics
from liteflow import losses

from dket.runtime import logutils
from dket.runtime import runtime
from dket.models import pointsoftmax
from dket import data
from dket import metrics as dmetrics
from dket import ops
from dket import optimizers
from logutils import HDEBUG


PATH = os.path.realpath(__file__)
BASE = os.path.dirname(PATH)


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
tf.app.flags.DEFINE_integer('eval-check-every-sec', 10, 'Time interval in seconds. If --mode=eval/test, the [LOG-DIR] is periodically checked to see if there is a new checkpoint to evaluate.')
tf.app.flags.DEFINE_integer('eval-check-until-sec', 300, 'Time interval in seconds. If --mode=eval/test, the maximum amount of time to wait for a new model checkpoint to appear in [LOG-DIR] before ending the evaluation process.')
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

FLAGS = tf.app.flags.FLAGS
# pylint: enable=C0301


def _get_log_dir():
    return os.path.join(FLAGS.base_log_dir, FLAGS.mode)


_MODE_TRAIN = 'train'
_MODE_EVAL = 'eval'
_MODE_TEST = 'test'

def _setup():
    """Set up the execution environment."""
    logging.debug('setting up the execution environment.')
    logging.debug('validating mode: %s', FLAGS.mode)
    modes = [_MODE_TRAIN, _MODE_TEST, _MODE_EVAL]
    if FLAGS.mode not in modes:
        message = "Mode must bee one of {}, found {}.".format(', '.join(modes), FLAGS.mode)
        logging.critical(message)
        raise ValueError(message)

    logging.debug('setting up the logging directory.')
    log_dir = _get_log_dir()
    if not os.path.exists(log_dir):
        logging.info('creating log directory %s', log_dir)
        os.makedirs(log_dir)

    logging.debug('setting up the log infrastructure.')
    log_level = logutils.parse_level(FLAGS.log_level)
    log_file = os.path.join(log_dir, FLAGS.log_file)
    log_to_stderr = FLAGS.log_to_stderr
    logutils.config(level=log_level, fpath=log_file, stderr=log_to_stderr)
    logging.debug('logging infrastructure configured.')
    logging.debug('setting TF log level to 9')
    tf.logging.set_verbosity(9)

    logging.info('python version: %s', sys.version.replace('\n', ' '))
    logging.info('executing: %s', PATH)
    for key, value in FLAGS.__flags.items():  # pylint: disable=I0011,W0212
        logging.info('--%s %s', key, str(value))


def _get_model_type():
    if not FLAGS.model_name:
        message = 'the model name MUST be specified.'
        logging.critical(message)
        raise ValueError(message)

    mtype = None
    logging.debug('loading the model class for name: %s.', FLAGS.model_name)
    if FLAGS.model_name == 'pointsoftmax':
        mtype = pointsoftmax.PointingSoftmaxModel
    else:
        message = 'Invalid model name {}'.format(FLAGS.model_name)
        logging.critical(message)
        raise ValueError(message)

    logging.info('model type %s loaded for name %s', mtype.__name__, FLAGS.model_name)
    return mtype


_GPU = 'GPU'
_CPU = 'CPU'

def _get_device_type():
    device = FLAGS.device
    if not device:
        logging.debug('device not set: trying to set it for the mode.')
        if FLAGS.mode == _MODE_TRAIN:
            logging.info('setting %s device type for mode %s', _GPU, FLAGS.mode)
            device = _GPU
        elif FLAGS.mode == _MODE_EVAL or FLAGS.mode == _MODE_TEST:
            logging.info('setting %s device type for mode %s', _CPU, FLAGS.mode)
            device = _CPU
        else:
            message = 'cannot set a default device for mode {}'.format(device)
            logging.critical(message)
            raise ValueError(message)
    else:
        supported = [_GPU, _CPU]
        if device not in supported:
            message = "unsupported device type: {}. Must be one of {}"\
                .format(device, ', '.join(supported))
            logging.critical(message)
            raise ValueError(message)
        logging.info('device type selected: %s', device)
    return device


def _get_hparams(dhparams):
    logging.log(HDEBUG, 'deault params:')
    for key, value in dhparams.values().items():
        logging.log(HDEBUG, 'default hparams.' + key + '=' + str(value))

    if not FLAGS.hparams:
        logging.info('no hparams set: using default.')
        hparams = dhparams
    else:
        hparams = FLAGS.hparams
        if isinstance(hparams, str):
            if os.path.exists(hparams):
                logging.debug('trying to parse hparams from file %s', hparams)
                hparams = ','.join([line.replace('\n', '') for line in open(hparams)])
        logging.debug('serialized hparams: %s', hparams)
        logging.debug('parsing hparams.')
        hparams = dhparams.parse(hparams)

    # final info logging.
    logging.info('model will be configured with the following hparams:')
    for key, value in hparams.values().items():
        logging.info('hparams.' + key + '=' + str(value))
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
    logging.info('epochs: %s, steps: %s', str(epochs), str(steps))
    return epochs, steps


def _get_data_files():
    logging.debug('getting data files.')
    if not FLAGS.data_dir and not FLAGS.data_files:
        message = 'Both data directory and data files are missing.'
        logging.critical(message)
        raise ValueError(message)

    if not FLAGS.data_dir:
        data_files = FLAGS.data_files.split(',')
    else:
        data_files = []
        for data_file in FLAGS.data_files:
            if os.path.isabs(data_file):
                logging.debug('absolute path data file: %s', data_file)
                data_files.append(data_file)
            else:
                logging.debug('relative path data file [%s/]%s', FLAGS.data_dir, data_file)
                data_files.append(os.path.join(FLAGS.data_dir, data_file))

    for data_file in data_files:
        logging.info('reading data from %s', data_file)
    return data_files


def _get_feed_dict():
    logging.debug('getting the feed dictionary.')
    # TODO(petrux): shuffle only in training. This can be HUGE to implement since
    # the liteflow component is made for shuffling and batching. So maybe a regular
    # TF component should be used.
    data_files = _get_data_files()
    shuffle = FLAGS.mode == _MODE_TRAIN
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
        logging.info('feeding with `%s`: %s', key, str(value))
    return feed_dict


def _get_loss():
    logging.debug('getting the loss function')
    loss = losses.StreamingLoss(
        func=losses.categorical_crossentropy,
        name='XEntropy')
    logging.info('loss function: categorical cross entropy (streaming average)')
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
        logging.info('learning rate: %f, decay rate: %f, decay steps: %d, staircase: %s',
                     lr, lr_decay_rate, lr_decay_steps, str(lr_decay_staircase))
    else:
        logging.info('learnign rate: %f', lr)

    optimizer = None
    if FLAGS.optimizer == _SGD:
        optimizer = optimizers.Optimizer.sgd(learning_rate=lr)
    else:
        message = 'invalid optimizer name: `{}`.'.format(FLAGS.optimizer)
        logging.critical(message)
        raise ValueError(message)

    logging.info('optimizer %s for name %s',
                 optimizer.__class__.__name__, FLAGS.optimizer)
    return optimizer


def _get_metrics_dict():
    logging.debug('getting evaluation metrics.')
    metrics_dict = {}

    logging.info('creating (per token) accuracy metric.')
    acc = metrics.StreamingMetric(metrics.accuracy, name='Accuracy')
    metrics_dict[acc.name] = acc

    logging.info('creating per sentence accuracy metric.')
    psa = metrics.StreamingMetric(metrics.per_sentence_accuracy, name='PerSentenceAccuracy')
    metrics_dict[psa.name] = psa
    return metrics_dict

def _get_post_metrics():
    logging.debug('getting the post metrics.')
    post_metrics = {}
    logging.info('creating Levenstein edit distance (LED).')
    led = dmetrics.Metric.editdistance()
    post_metrics['led'] = led
    return post_metrics


def _get_loop(model):
    mode = FLAGS.mode
    _, steps = _get_epochs_and_steps()
    post_metrics = _get_post_metrics()
    if mode == _MODE_TRAIN:
        logging.info('building the train loop.')
        loop = runtime.TrainLoop(
            model=model,
            log_dir=_get_log_dir(),
            steps=steps,
            checkpoint_every=FLAGS.checkpoint_every_steps,
            post_metrics=post_metrics)
        return loop
    if mode == _MODE_EVAL or mode == _MODE_TEST:
        logging.info('building the evaluation loop.')
        ckprov = runtime.CheckpointProvider(
            checkpoint_dir=os.path.join(FLAGS.base_log_dir, _MODE_TRAIN),
            idle_time=FLAGS.eval_check_every_sec,
            max_idle_time=FLAGS.eval_check_until_sec)
        return runtime.EvalLoop(
            model=model,
            log_dir=_get_log_dir(),
            checkpoint_provider=ckprov,
            steps=steps)
    raise ValueError('Invalid mode: ' + mode)


def _build_model():
    logging.debug('building the model.')
    mtype = _get_model_type()
    with tf.Graph().as_default() as graph:  # pylint: disable=I0011,E1129
        with tf.device(_get_device_type()):
            feed_dict = _get_feed_dict()
            hparams = _get_hparams(mtype.get_default_hparams())
            loss = _get_loss()
            optimizer = None
            if FLAGS.mode == _MODE_TRAIN:
                optimizer = _get_optimizer()
            metrics_dict = _get_metrics_dict()
            return mtype(graph=graph)\
                .feed(feed_dict)\
                .build(hparams, loss, optimizer, metrics_dict)


def main(_):
    """Main execution body."""
    _get_loop(_build_model()).start()

if __name__ == '__main__':
    _setup()
    tf.app.run()
