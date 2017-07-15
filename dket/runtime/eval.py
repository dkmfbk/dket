"""Evaluate a dket model."""

import logging
import os
import sys
import time

import tensorflow as tf

from dket.runtime import logutils
from dket.runtime.logutils import HDEBUG

PATH = os.path.realpath(__file__)
BASE = os.path.dirname(PATH)

# <3 COMMAND LINE FLAGS. <3
# pylint: disable=C0301
tf.app.flags.DEFINE_string('model-name', None, 'The model name.')
tf.app.flags.DEFINE_string('checkpoint', None, 'The checkpoint file to be evaluated.')
tf.app.flags.DEFINE_integer('batch-size', 100, 'The batch size.')
tf.app.flags.DEFINE_string('hparams', None, 'The hparams for the model instance. A comma-separated list of key=value pairs or a path to a file with one key=value pair per line.')
tf.app.flags.DEFINE_string('data-files', None, 'A comma separated list of data file patterns (e.g. `file-*.txt`).')
tf.app.flags.DEFINE_string('log-dir', '.', 'The base log directory. All the logs and summaries will be stored in [BASE-LOG-DIR]/eval. If not set, logs will be printed to /tmp/.')
tf.app.flags.DEFINE_string('log-level', 'INFO', 'The log level. Can be none ore one of HDEBUG, DEBUG, INFO, WARNING')
tf.app.flags.DEFINE_string('log-file', 'log', 'The log file as file name or as an absolute path, if not specified, it is just `log`. If not an absolute path, it will be placed in [LOG-DIR]/eval/[LOG-FILE]. If the file already exists, new log entries will be appended.')
tf.app.flags.DEFINE_boolean('log-to-stderr', False, 'If set, redirect also the log entries with level less than WARNING to the standard error stream.')


FLAGS = tf.app.flags.FLAGS
# pylint: enable=C0301


_ALLOW_SOFT_DEV_PLACEMENT = True


class Evaluation(object):
    """Evaluation phase for a model."""

    _NOT_NONE_ARG = '{} argument cannot be `None`'

    def __init__(self, model,  checkpoint, steps=0,
                 log_dir=None, post_metrics=None, dump_dir=None):
        """Initializes a new Evaluation instance."""

        if model is None:
            message = self._NOT_NONE_ARG.format('model')
            logging.critical(message)
            raise ValueError(message)
        self._model = model

        if checkpoint is None:
            message = self._NOT_NONE_ARG.format('checkpoint')
            logging.critical(message)
            raise ValueError(message)
        self._checkpoint = checkpoint
        logging.debug('checkpoint to evaluate: %s', self._checkpoint)
        
        self._steps = steps or 0
        logging.debug('number of steps: %s', self._steps)

        self._log_dir = log_dir or ''
        logging.debug('logging directory: %s', self._log_dir)

        self._post_metrics = post_metrics or {}
        logging.debug('%d downstream metrics set for the evaluation.')
        for key, _ in self._post_metrics.items():
            logging.log('%s post metric', key)

        self._dump_dir = dump_dir or ''
        self._dump_file = ''
        logging.debug('dumping directory: %s', self._dump_dir)

        self._global_step = None
        self._eval_step = None
        self._fetches = None
        self._saver = None
        self._writer = None
        self._session = None

    def run(self):
        """Run the evaluation of the checkpoint."""

        # reset some instance variables.
        self._global_step = None
        self._eval_step = None
        self._fetches = None

        # initialize saver/writer.
        with self._model.graph.as_default() as graph:
            logging.debug('initializing session saver.')
            self._saver = tf.train.Saver()
            if self._log_dir:
                logging.debug('initializing the summary writer to path %s.', self._log_dir)
                self._writer = tf.summary.FileWriter(self._log_dir, graph=graph)
                logging.debug('flushing writer.')
                self._writer.flush()

        # initializing eval step and fetches.
        logging.info('initializing evaluation step.')
        self._eval_step = 0
        logging.debug('getting streaming average metrics update_ops.')
        metrics_update_ops = {}
        for key, metric in self._model.metrics.items():
            metrics_update_ops[key] = metric.update_op
        logging.debug('initializing step fetches.')
        self._fetches = [
            metrics_update_ops,  # it's a dictionary!
            self._model.words,
            self._model.target,
            self._model.output,
        ]
        for key, pmetric in self._post_metrics.items():
            logging.debug('resetting post metric %s', key)
            pmetric.reset()

        config = tf.ConfigProto(allow_soft_placement=_ALLOW_SOFT_DEV_PLACEMENT)
        self._session = tf.Session(config=config, graph=self._model.graph)
        with self._session as sess:
            logging.debug('initializing global and local variables')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            logging.debug('restoring session.')
            self._saver.restore(sess, self._checkpoint)

            logging.debug('initializing coordinator and starting queue runners.')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                logging.debug('getting the global step.')
                self._global_step = sess.run(self._model.global_step)
                logging.info('global step: %d', self._global_step)

                logging.debug('starting the train loop.')
                eval_loop = True
                while eval_loop:
                    eval_loop = self._step()

            except tf.errors.OutOfRangeError as ex:
                logging.info('a tf.errors.OutOfRangeError is stopping the loop.')
                coord.request_stop(ex=ex)
            finally:
                logging.debug('stopping the loop.')
                coord.request_stop()
                coord.join(threads)
                self._summarize(sess)

        logging.info('evaluation of checkpoint %s complete.', self._checkpoint)

    def _step(self):
        """Run the evaluation step."""

        metrics, words, targets, predictions = self._session.run(self._fetches)

        for key, pmetric in self._post_metrics.items():
            logging.debug('accumulating post metric: %s', key)
            curr = pmetric.compute(targets, predictions)
            metrics[key] = curr
        logging.debug(
            self._summary_msg(
                self._eval_step, metrics, self._checkpoint))

        self._dump(words, targets, predictions)

        self._eval_step += 1
        next_step = self._steps == 0 or self._eval_step < self._step
        logging.debug('next evaluation step? %s', str(next_step))
        return next_step

    def _dump(self, words, targets, predictions):
        """Dump the batch to thee dump file."""
        if not self._dump_dir:
            logging.debug('no dumping will be performed.')
            return  # exit the method body

        # initialize the dump file if not existing.
        if not self._dump_file:
            logging.debug('initializing dumping file.')
            os.makedirs(self._dump_dir)
            dump_file_name = 'dump-' + str(self._global_step) + '.tsv'
            self._dump_file = os.path.join(self._dump_dir, dump_file_name)
            logging.info('dumping to: %s', self._dump_file)

        _str = lambda items: ' '.join([str(item) for item in list(items)])

        with open(self._dump_file, mode='a') as fdump:
            for ww, tt, pp in zip(words, targets, predictions):
                fdump.write('\t'.join([_str(ww), str(tt), str(pp)]) + '\n')

    def _summarize(self, sess):
        """Summarize the session metrics."""
        values = []

        logging.debug('getting average metrics values.')
        metrics_avg_t = {}
        for key, metric in self._model.metrics.items():
            metrics_avg_t[key] = metric.value

        logging.debug('evaluating average loss and metrics.')
        fetches = metrics_avg_t
        metrics_avg = sess.run(fetches)

        for key, pmetric in self._post_metrics.items():
            logging.debug('adding post metric: %s', key)
            metrics_avg[key] = pmetric.average

        logging.info(
            self._summary_msg(
                self._global_step, metrics_avg, self._checkpoint))

        if self._writer is None:
            logging.debug('no summary will be written.')
            return  # exit from the method.

        logging.debug('creating metrics summaries.')
        for key, value in metrics_avg.items():
            values.append(tf.summary.Summary.Value(tag=key, simple_value=value))

        logging.debug('writing the summaries.')
        summary = tf.summary.Summary(value=values)
        self._writer.add_summary(summary, global_step=self._global_step)
        self._writer.flush()

    def _summary_msg(self, step, metrics, checkpoint):
        """Build the logging summarization message."""
        return ', '.join(
            ['step: {}'.format(step)] +
            ['{}: {:.2f}'.format(key, value) for key, value in metrics.items()]+
            ['checkpoint: {}'.format(checkpoint)])


def _get_log_dir():
    if not FLAGS.log_dir:
        return os.path.join('/tmp/', 'eval' + str(time.time()))
    return os.path.join(FLAGS.log_dir, 'eval')

def _setup():
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

def _main():
    pass

if __name__ == '__main__':
    _setup()
    tf.app.run()