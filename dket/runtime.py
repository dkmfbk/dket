"""Runtime infrastructure for training and evaluating dket models."""

import copy
from collections import OrderedDict
import json
import logging
import time
import shutil
import os

import numpy as np
import tensorflow as tf

import dket.model
from dket import configurable
from dket.metrics import Metric
from dket.model import Model, ModelInputs
from dket.logutils import HDEBUG


_LOSS_SUMMARY_KEY = 'Loss'
_MAX_TO_KEEP = 20
_SOFT_PLACEMENT = True


def as_summary(kvp):
    """Turns a dictionary into a `TensorFlow` summary."""
    return tf.summary.Summary(
        value=[
            tf.summary.Summary.Value(tag=key, simple_value=value)
            for key, value in kvp.items()
        ])

def get_metrics():
    """Build the metrics dictionary."""
    metrics = [
        Metric.editdistance(),
        Metric.per_token_accuracy(),
        Metric.per_sentence_accuracy()]
    return dict((m.name, m) for m in metrics)


class Experiment(object):
    """A dket experiment main class."""

    CONFIG_FILE_NAME = 'config.json'
    LOG_FILE_NAME = 'log'

    NAME_KEY = 'name'
    LOGDIR_KEY = 'logdir'
    TRAIN_FILES_KEY = 'train.files'
    TRAIN_STEPS_KEY = 'train.steps'
    TRAIN_CKPT_EVERY_KEY = 'train.checkpoint_every'
    TRAIN_DEVICE_KEY = 'train.device'
    EVAL_FILES_KEY = 'eval.files'
    EVAL_DUMP_KEY = 'eval.dump'
    EVAL_DEVICE_KEY = 'eval.device'
    MODEL_CLASS_KEY = 'model.class'
    MODEL_PARAMS_KEY = 'model.params'

    _TRAIN = tf.contrib.learn.ModeKeys.TRAIN
    _EVAL = tf.contrib.learn.ModeKeys.EVAL

    def __init__(self, config):
        """Initializes a new experiment instance."""

        self._logdir = config[self.LOGDIR_KEY]
        self._train_files = config[self.TRAIN_FILES_KEY]
        self._train_steps = config[self.TRAIN_STEPS_KEY]
        self._train_dev = config[self.TRAIN_DEVICE_KEY]
        self._train_ckpt_every = config[self.TRAIN_CKPT_EVERY_KEY]
        self._eval_files = config[self.EVAL_FILES_KEY]
        self._eval_dev = config[self.EVAL_DEVICE_KEY]
        self._eval_dump = config[self.EVAL_DUMP_KEY]
        self._params = config[self.MODEL_PARAMS_KEY]

        # build the training model.
        t_params = copy.deepcopy(self._params)
        t_params[Model.INPUT_PARAMS_PK][ModelInputs.FILES_PK] = self._train_files
        clz = config[self.MODEL_CLASS_KEY]
        with tf.device(self._train_dev):
            t_model = configurable.factory(clz, self._TRAIN, t_params, dket.model)
            t_logdir = os.path.join(self._logdir, self._TRAIN)

        # build the eval model.
        e_params = copy.deepcopy(self._params)
        e_params[Model.INPUT_PARAMS_PK][ModelInputs.FILES_PK] = self._eval_files
        e_params[Model.INPUT_PARAMS_PK][ModelInputs.EPOCHS_PK] = 1
        with tf.device(self._eval_dev):
            e_model = configurable.factory(clz, self._EVAL, e_params, dket.model)
            e_logdir = os.path.join(self._logdir, self._EVAL)
            e_dumpdir = os.path.join(e_logdir, 'dump') if self._eval_dump else ''

        self._eval = Evaluation(
            model=e_model,
            logdir=e_logdir,
            steps=0,
            metrics=get_metrics(),
            dumpdir=e_dumpdir)
        self._training = Training(
            model=t_model,
            logdir=t_logdir,
            steps=self._train_steps,
            checkpoint_every=self._train_ckpt_every,
            metrics=get_metrics(),
            evaluation=self._eval)

    def run(self):
        """Runs the experiment."""
        self._training.start()

    @classmethod
    def get_default_config(cls):
        """Gets the default configuration settings."""
        return OrderedDict([
            (cls.NAME_KEY, ''),
            (cls.LOGDIR_KEY, ''),
            (cls.TRAIN_FILES_KEY, ''),
            (cls.TRAIN_STEPS_KEY, 0),
            (cls.TRAIN_CKPT_EVERY_KEY, 1),
            (cls.TRAIN_DEVICE_KEY, 'GPU'),
            (cls.EVAL_FILES_KEY, ''),
            (cls.EVAL_DUMP_KEY, True),
            (cls.EVAL_DEVICE_KEY, 'CPU'),
            (cls.MODEL_CLASS_KEY, ''),
            (cls.MODEL_PARAMS_KEY, OrderedDict)
        ])

    @staticmethod
    def _abs_file_paths(base, patterns):
        return ','.join([
            os.path.abspath(
                os.path.join(base, p))
            for p in patterns.split(',')])

    @classmethod
    def load(cls, config, logdir=None, force=False):
        """Load an experiment from logdir containing only a json config file."""
        if not config:
            raise ValueError('Experiment configuration must be specified.')

        basedir, _ = tuple(os.path.split(config))
        fname = os.path.splitext(os.path.basename(config))[0]
        config = json.load(open(config))
        
        name = config[cls.NAME_KEY]
        if name != fname:
            logging.warning('Found name %s for file %s; unsing %s.', name, fname, fname)
            name = fname

        if logdir:
            logging.warning('overwriting logdir with %s', logdir)
            config[cls.LOGDIR_KEY] = logdir

        logdir = config[cls.LOGDIR_KEY]
        if not logdir:
            logging.warning('no logdir set, assuming %s', basedir)
            logdir = basedir

        if not os.path.isabs(logdir):
            logging.info('trying to build an absolute path for logdir %s', logdir)
            logdir = os.path.abspath(os.path.join(basedir, logdir))
        
        logdir = os.path.join(logdir, name)
        logging.info('experiment results will be loggedd to: %s', logdir)
        
        if os.path.exists(logdir):
            if force:
                logging.info('removing existing logdir %s and recreating.', logdir)
                shutil.rmtree(logdir)
                os.makedirs(logdir)
            else:
                if os.listdir(logdir):
                    raise FileExistsError(
                        'The logging directory {} already exists and is not empty.'\
                        .format(logdir))
                else:
                    logging.warning(
                        'The log directory %s already exists but is empty: will be used anyway.',
                        logdir)
        else:
            os.makedirs(logdir)
            logging.info('created logdir is: %s', logdir)
        #override the logdir setting.
        config[cls.LOGDIR_KEY] = logdir

        # Train/Eval file patterns made absolude.
        config[cls.TRAIN_FILES_KEY] = cls._abs_file_paths(basedir, config[cls.TRAIN_FILES_KEY])
        config[cls.EVAL_FILES_KEY] = cls._abs_file_paths(basedir, config[cls.EVAL_FILES_KEY])
        return Experiment(config)


class Training(object):
    """Training process."""

    _CKPT_NAME = 'CHECKPOINT'

    def __init__(self, model, logdir,
                 steps=0, checkpoint_every=0,
                 metrics=None, evaluation=None):
        logging.debug('initializing the training process instance.')
        self._model = model
        self._logdir = logdir
        self._steps = steps
        self._ckpt_every = checkpoint_every
        self._metrics = metrics
        self._eval = evaluation

        self._config = None
        self._saver = None
        self._writer = None
        self._fetches = None
        self._ckpt_name = None
        self._ckpt_ts = None
        self._sess = None
        self._initialize()

    def _initialize(self):
        logging.debug('initializing training process.')
        if not os.path.exists(self._logdir):
            logging.info('creating directory: %s', self._logdir)
            os.makedirs(self._logdir)

        with self._model.graph.as_default() as graph:
            logging.debug('max checkpoint kept: %s', _MAX_TO_KEEP)
            self._saver = tf.train.Saver(max_to_keep=_MAX_TO_KEEP)
            logging.debug('saving graph definition.')
            self._writer = tf.summary.FileWriter(self._logdir, graph=graph)
            self._writer.flush()

        self._ckpt_ts = time.time()
        self._ckpt_name = os.path.join(self._logdir, self._CKPT_NAME)
        logging.debug('checkpoint name: %s', self._ckpt_name)

        logging.debug('initializing fetches.')
        target_key = self._model.inputs.FORMULA_KEY
        target_length_key = self._model.inputs.FORMULA_LENGTH_KEY
        self._fetches = [
            self._model.global_step,
            self._model.train_op,
            self._model.loss_op,
            self._model.summary_op,
            self._model.inputs.get(target_key),
            self._model.predictions,
            self._model.inputs.get(target_length_key)]

        logging.debug('allow soft placement: %s', str(_SOFT_PLACEMENT))
        self._config = tf.ConfigProto(allow_soft_placement=_SOFT_PLACEMENT)
        logging.debug('initializing session instance.')
        self._sess = tf.Session(config=self._config, graph=self._model.graph)
        logging.debug('training process initialized.')

    def start(self):
        """Start the training process."""
        logging.info('starting training process.')
        with self._sess as sess:
            logging.debug('initializing global/local variables.')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            logging.debug('initializing coordinator and starting queue runners.')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            # START LOOP.
            step = None
            try:
                has_next = True
                while has_next:
                    step, has_next = self._step()
            except tf.errors.OutOfRangeError as oore:
                logging.info('a tf.errors.OutOfRangeError is stopping the loop.')
                coord.request_stop(ex=oore)
            finally:
                self._writer.flush()
            # END LOOP.
                logging.info('stopping the loop.')
                coord.request_stop()
                coord.join(threads)
        logging.info('training process complete.')

    def _step(self):
        step, _, loss, summary, targets, predictions, lengths = self._sess.run(self._fetches)

        logging.log(HDEBUG, 'computing donwstream metrics')
        metrics = dict((key, metric.reset().compute(targets, predictions, lengths))
                       for (key, metric) in self._metrics.items())

        save_step = self._ckpt_every == 0 or (step % self._ckpt_every == 0)
        ckpt = self._save_ckpt(step) if save_step else None
        self._summarize(step, loss, summary, metrics, ckpt=ckpt)
        if ckpt and self._eval:
            self._eval.start(ckpt)

        next_step = self._steps == 0 or step < self._steps
        logging.log(HDEBUG, 'next step: %s', str(next_step))
        return step, next_step

    def _summarize(self, step, loss, summary, metrics, ckpt=None):
        self._writer.add_summary(summary, global_step=step)
        fmt = '{}: {:.5f}'
        message = ', '.join(
            ['global step: {}'.format(step),
             fmt.format(_LOSS_SUMMARY_KEY, loss)] + 
            [fmt.format(key, value) for key, value in metrics.items()])

        metrics.update({_LOSS_SUMMARY_KEY: loss})
        summarized = as_summary(metrics)
        self._writer.add_summary(summarized, global_step=step)
        self._writer.flush()

        if ckpt is None:
            logging.debug(message)
        else:
            logging.info('checkpoint: {}'.format(ckpt) + ', ' + message)

    def _save_ckpt(self, step):
        ckpt = self._saver.save(self._sess, self._ckpt_name, step)
        delta = time.time() - self._ckpt_ts
        logging.debug('checkpoint saved at {} (time elapsed {:.2f}s)'.format(ckpt, delta))
        self._ckpt_ts = time.time()
        return ckpt


class Evaluation(object):
    """Evaluation process."""

    def __init__(self, model, logdir, steps=0, metrics=None, dumpdir=None):
        logging.debug('initializing the evaluation process instance.')
        self._model = model
        self._logdir = logdir
        self._steps = steps
        self._metrics = metrics or {}
        self._dumpdir = dumpdir

        self._config = None
        self._saver = None
        self._writer = None
        self._fetches = None
        self._sess = None
        self._global_step = None
        self._eval_step = None
        self._initialize()

    def _initialize(self):
        logging.debug('initializing the evaluation process.')
        if not os.path.exists(self._logdir):
            logging.info('creating log directory: %s', self._logdir)
            os.makedirs(self._logdir)

        if self._dumpdir and not os.path.exists(self._dumpdir):
            logging.info('creating dump directory: %s', self._dumpdir)
            os.makedirs(self._dumpdir)

        with self._model.graph.as_default() as graph:
            logging.debug('max checkpoint kept: %s', _MAX_TO_KEEP)
            self._saver = tf.train.Saver(max_to_keep=_MAX_TO_KEEP)
            logging.debug('saving graph definition.')
            self._writer = tf.summary.FileWriter(self._logdir, graph=graph)
            self._writer.flush()

        logging.debug('initializing fetches.')
        words_key = self._model.inputs.WORDS_KEY
        target_key = self._model.inputs.FORMULA_KEY
        # target_length_key = self._model.inputs.FORMULA_LENGTH_KEY
        self._fetches = [
            self._model.inputs.get(words_key),
            self._model.inputs.get(target_key),
            self._model.predictions]

        logging.debug('allow soft placement: %s', str(_SOFT_PLACEMENT))
        self._config = tf.ConfigProto(allow_soft_placement=_SOFT_PLACEMENT)
        logging.debug('evaluation process initialized.')

    def start(self, checkpoint):
        """Start the evaluation process."""
        
        logging.debug('resetting the metrics.')
        for key, metric in self._metrics.items():
            logging.debug('reseting metric %s.', key)
            metric.reset()

        logging.debug('initializing session instance.')
        self._sess = tf.Session(config=self._config, graph=self._model.graph)
        with self._sess as sess:
            logging.debug('initializing global and local variables')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            logging.debug('restoring session.')
            self._saver.restore(sess, checkpoint)

            logging.debug('initializing coordinator and starting queue runners.')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            self._eval_step = 0
            logging.debug('getting the global step.')
            self._global_step = sess.run(self._model.global_step)
            logging.info(
                """starting the evaluation process of checkpoint: %s """
                """at global step %d (and local step %d).""",
                checkpoint, self._global_step, self._eval_step)
            try:
                loop = True
                while loop:
                    loop = self._step()
            except tf.errors.OutOfRangeError as ex:
                logging.debug('a tf.errors.OutOfRangeError is stopping the loop.')
                coord.request_stop(ex=ex)
            finally:
                logging.info('stopping the loop.')
                coord.request_stop()
                coord.join(threads)
                self._summarize()

    def _step(self):
        words, targets, predictions = self._sess.run(self._fetches)
        self._eval_step += 1

        tmetrics = {}
        for key, metric in self._metrics.items():
            logging.log(HDEBUG, 'accumulating metric: %s.', key)
            tmetrics[key] = metric.compute(targets, predictions)
        tmsg = ', '.join(['{}:{:.2f}'.format(k, v) for k, v in tmetrics.items()])
        logging.debug('evaluation step  %d@%d: %s', self._global_step, self._eval_step, tmsg)

        predictions = np.argmax(predictions, axis=-1)
        self._dump(words, targets, predictions)

        return self._steps == 0 or self._eval_step <= self._steps

    def _dump(self, words, targets, predictions):
        """Dump the batch to thee dump file."""
        if not self._dumpdir:
            logging.debug('no dumping will be performed.')
            return  # exit the method body

        # initialize the dump file if not existing.
        dumpfile_name = 'dump-' + str(self._global_step) + '.tsv'
        dumpfile = os.path.join(self._dumpdir, dumpfile_name)
        logging.log(HDEBUG, 'dumping to: %s', dumpfile)

        _str = lambda items: ' '.join([str(item) for item in list(items)])
        with open(dumpfile, mode='a') as fdump:
            for ww, tt, pp in zip(words, targets, predictions):
                fdump.write('\t'.join([_str(ww), _str(tt), _str(pp)]) + '\n')

    def _summarize(self):
        gmetrics = {}
        for key, metric in self._metrics.items():
            gmetrics[key] = metric.average()
        gmsg = ', '.join(['{}:{:.5f}'.format(k, v) for k, v in gmetrics.items()])
        logging.info('evaluation at global step %d: %s', self._global_step, gmsg)
        logging.debug('saving tf summaries.')
        self._writer.add_summary(as_summary(gmetrics), global_step=self._global_step)
        self._writer.flush()
