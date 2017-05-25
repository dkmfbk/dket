"""Run your dket experiment."""

import os

import tensorflow as tf

# pylint: disable=C0301
tf.app.flags.DEFINE_string('model', None, 'The model name.')
tf.app.flags.DEFINE_string('hparams', None, 'The hparams for the model instance. A comma-separated list of key=value pairs.')
tf.app.flags.DEFINE_boolean('check-hparams', False, 'If set, tries to check the value of some hparams with respect to the experiment data files (e.g. dimension of vocabulary, ecc).')
tf.app.flags.DEFINE_string('data-dir', None, 'The directory where the data files are. If omitted, the local directory will be used.')
tf.app.flags.DEFINE_string('files', None, 'A comma separated list of data file patterns (e.g. `file-*.txt`). If the --data-dir is provided, file patterns can be relative to that directory. If no value is provided, [DATA-DIR]/[MODE]*.* is the default.')
tf.app.flags.DEFINE_string('words', 'words.idx', 'The input vocabulary .idx file. Default value is [DATA-DIR]/words.idx.')
tf.app.flags.DEFINE_string('terms', 'terms.idx', 'The output vocabulary .idx file. efault value is [DATA-DIR]/terms.idx')

tf.app.flags.DEFINE_string('mode', 'train', 'The execution mode: can be train, eval, test.')
tf.app.flags.DEFINE_integer('epochs', None, 'The number of training epochs. If none, only the [STEPS] value will be considered.')
tf.app.flags.DEFINE_integer('steps', None, 'The number of training steps. If none, only the [EPOCHS] value will be considered.')
tf.app.flags.DEFINE_integer('train_save_every_steps', 100, 'Number of training steps after which the model state is saved. This flag is considered only in --mode=train.')
tf.app.flags.DEFINE_integer('eval_check_every_sec', 300, 'Time interval in seconds. If --mode=eval, the [LOG-DIR] is periodically checked to see if there is a new checkpoint to evaluate.')
tf.app.flags.DEFINE_integer('eval_check_untill_sec', 3600, 'Time interval in seconds. If --mode=eval, the maximum amount of time to wait for a new model checkpoint to appear in [LOG-DIR] before ending the evaluation process.')

tf.app.flags.DEFINE_float('lr', 0.1, 'The initial value for the learning rate.')
tf.app.flags.DEFINE_float('lr-decay-rate', None, 'The decay rate for the learning rate.')
tf.app.flags.DEFINE_float('lr-decay-step', None, 'The decay steps for the learning rate.')
tf.app.flags.DEFINE_boolean('lr-decay-staircase', True, 'If set, the learning rate decay is staircase-like.')
tf.app.flags.DEFINE_string('optimizer', 'sgd', 'The optimizer name. Can be one of sgd, adagrad, adadelta, adam.')
tf.app.flags.DEFINE_float('adagrad-epsilon', None, 'If --optimizer=adagrad, set the epsilon parameter.')
tf.app.flags.DEFINE_float('adagrad-rho', None, 'If --optimizer=adagrad, set the rho parameter.')

tf.app.flags.DEFINE_string('device', None, 'Force the model to be built on a certain type of device (CPU or GPU). If not set, the device will be chosen according to the mode using GPU for training and CPU for evaluation.')
tf.app.flags.DEFINE_integer('threads', 4, 'The number of threads to be used for the input pipeline.')
tf.app.flags.DEFINE_string('log-dir', None, 'The log directory where all the model dumps, summaries, ecc. will be stored.')
tf.app.flags.DEFINE_string('log-level', 'INFO', 'The log level. Can be none ore one of DEBUG, INFO, WARN')
tf.app.flags.DEFINE_string('log-file', 'logfile', 'The log file as a relative path in [LOG-DIR] or as an absolute path')
tf.app.flags.DEFINE_boolean('log-to-stdout', False, 'If set, redirect the log entries to the standard output stream.')
tf.app.flags.DEFINE_boolean('log-to-stderr', False, 'If set, redirect the log entries to the standard error stream.')
# pylint: enable=C0301

FLAGS = tf.app.flags.FLAGS
BASE = os.path.dirname(os.path.realpath(__file__))

def main(_):
    """Main application entry point."""
    raise NotImplementedError('Not implemented (yet)!')

if __name__ == '__main__':
    tf.app.run()
