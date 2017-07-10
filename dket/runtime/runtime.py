"""Runtime utilities."""

import logging
import time
import os
from datetime import datetime

import tensorflow as tf

from dket.runtime.logutils import HDEBUG


def _validate_not_none(arg, name):
    if arg is None:
        message = '{} argument cannot be `None`'.format(name)
        logging.critical(message)
        raise ValueError(message)
    message = '{}: {}'.format(name, arg)
    logging.debug(message)
    return arg


_TIMESTAMP_FMT = '%Y-%m-%d %H:%M:%S.%3d'

def _timestamp_fmt(timestamp):
    return datetime.fromtimestamp(timestamp).strftime(_TIMESTAMP_FMT)


def _as_summary(kvp):
    return tf.summary.Summary(
        value=[
            tf.summary.Summary.Value(tag=key, simple_value=value)
            for key, value in kvp.items()
        ])


_MAX_CHECKPOINTS_TO_KEEP = 20
_ALLOW_SOFT_DEV_PLACEMENT = True


class TrainLoop(object):
    """Train loop."""

    def __init__(self, model, log_dir,
                 steps=0, checkpoint_every=0,
                 post_metrics=None):
        logging.debug('initializing TrainLoop instance.')
        self._model = _validate_not_none(model, 'model')
        self._log_dir = _validate_not_none(log_dir, 'log_dir')
        self._steps = _validate_not_none(steps, 'steps')
        self._checkpoint_every = _validate_not_none(checkpoint_every, 'checkpoint_every')
        self._post_metrics = post_metrics or {}
        if not self._post_metrics:
            logging.debug('no post metrics set.')
        for key, _ in self._post_metrics.items():
            logging.log(HDEBUG, 'post metric: %s', key)

        self._checkpoint_name = os.path.join(self._log_dir, 'CHECKPOINT')
        logging.info('TrainLoop initialized. Checkpoint at %s.', self._checkpoint_name)

        self._next_step = False
        self._saver = None
        self._writer = None
        self._fetches = None
        self._last_checkpoint_ts = None

    def start(self):
        """Start the train loop."""
        logging.debug('started running the train loop.')
        with self._model.graph.as_default() as graph:
            logging.debug('initializing session saver.')
            logging.debug('max number of checkpoint to keep: %d.', _MAX_CHECKPOINTS_TO_KEEP)
            self._saver = tf.train.Saver(max_to_keep=_MAX_CHECKPOINTS_TO_KEEP)
            logging.debug('initializing summary writer to path %s.', self._log_dir)
            self._writer = tf.summary.FileWriter(self._log_dir, graph=graph)
            logging.debug('flushing writer.')
            self._writer.flush()

        logging.debug('setting up session configuration')
        logging.debug('allow softt device placement: %s', str(_ALLOW_SOFT_DEV_PLACEMENT))
        config = tf.ConfigProto(allow_soft_placement=_ALLOW_SOFT_DEV_PLACEMENT)

        # Here starts the actual train loop.
        logging.debug('starting session.')
        with tf.Session(config=config, graph=self._model.graph) as sess:
            logging.debug('initializing local and global variables.')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            logging.debug('initializing coordinator and starting queue runners.')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                logging.info('starting the train loop.')
                self._last_checkpoint_ts = time.time()
                logging.debug('checkpoint start at %s', _timestamp_fmt(self._last_checkpoint_ts))
                self._next_step = True
                while self._next_step:
                    step, self._next_step = self._step(sess)
            except tf.errors.OutOfRangeError as ex:
                logging.info('a tf.errors.OutOfRangeError is stopping the loop.')
                coord.request_stop(ex=ex)
                logging.debug('saving latest checkpoint')
                checkpoint, delta = self._save_checkpoint(sess, self._checkpoint_name, step)
                logging.info('saved last checkpoint %s, checkpoint time: %f.', checkpoint, delta)
            finally:
                logging.debug('stopping the loop.')
                coord.request_stop()
                coord.join(threads)

        logging.info('cleaning up.')
        logging.debug('deallocatiing saver.')
        self._saver = None
        logging.debug('flushing and deallocating file writer.')
        self._writer.flush()
        self._writer = None
        logging.debug('deallocating the checkpoint timestep.')
        self._last_checkpoint_ts = None
        logging.debug('deallocating fetches.')
        self._fetches = None
        self._next_step = False
        logging.debug('has next step? %s.', str(self._next_step))
        logging.info('training loop complete.')


    def _step(self, sess):
        """The next training step.

        Returns:
          a tuple `global_step, continue` where:
            `global_step` is an int indicating the current value of the global step,
            `continue` is True if the number current step has reached the number of
              steps set for the training.

        Raises:
          tf.errors.OutOfRangeError: if the input queues has finished their job.
        """
        if not self._fetches:
            logging.info('initializing fetches.')
            logging.debug('getting the per-batch metrics.')
            metrics_t = {}
            for key, value in self._model.metrics.items():
                metrics_t[key] = value.batch_value
            self._fetches = [
                self._model.global_step,
                self._model.train_op,
                self._model.loss.batch_value,
                self._model.summary_op,
                metrics_t,  # it's a dictionary.,
                self._model.target,
                self._model.output,
                self._model.inputs[self._model.FORMULA_LENGTH_KEY],
            ]

        logging.debug('running the session step.')
        step, _, loss, summary, metrics, targets, predictions, lengths = sess.run(self._fetches)

        if self._post_metrics:
            logging.debug('adding post metrics.')
            for key, pmetric in self._post_metrics.items():
                metrics[key] = pmetric.reset().compute(targets, predictions, lengths)

        logging.debug('creating loss summary.')
        self._writer.add_summary(_as_summary({'loss': loss}), global_step=step)
        logging.debug('creating metrics summaries.')
        self._writer.add_summary(_as_summary(metrics), global_step=step)

        logging.debug('writing session summaries.')
        self._writer.add_summary(summary, global_step=step)
        logging.log(HDEBUG, 'flushing summaries.')
        self._writer.flush()

        save_step = self._checkpoint_every == 0 or (step % self._checkpoint_every == 0)
        logging.log(HDEBUG, 'is a checkpoint step? %s.', 'yes' if save_step else 'no')
        if save_step:
            checkpoint, delta = self._save_checkpoint(sess, self._checkpoint_name, step)
            message = ', '.join(
                [self._summary_msg(step, loss, metrics),
                 self._checkpoint_msg(checkpoint, delta)])
            logging.info(message)
        else:
            logging.log(HDEBUG, self._summary_msg(step, loss, metrics))

        next_step = self._steps == 0 or step < self._steps
        logging.log(HDEBUG, 'next step: %s', str(next_step))
        return step, next_step

    def _save_checkpoint(self, sess, name, global_step):
        logging.debug('checkpoint at step %d', global_step)
        delta = time.time() - self._last_checkpoint_ts
        logging.debug('%f seconds elapsed from last checkpoint.', delta)
        checkpoint = self._saver.save(sess, name, global_step=global_step)
        logging.debug('checkpoint saved at %s', checkpoint)
        self._last_checkpoint_ts = time.time()
        logging.log(HDEBUG, 'new latest checkpoint timestep: %s',
                    _timestamp_fmt(self._last_checkpoint_ts))
        return checkpoint, delta

    def _summary_msg(self, step, loss, metrics):
        return ', '.join(
            ['global step: {}'.format(step), 'loss: {:.2f}'.format(loss)] +
            ['{}: {:.2f}'.format(key, value) for key, value in metrics.items()])

    def _checkpoint_msg(self, checkpoint, delta):
        return ', '.join(
            ['checkpoint at {}'.format(checkpoint),
             'time elapsed {:.2f} sec'.format(delta)])


class EvalLoop(object):
    """Evaluation loop."""

    def __init__(self, model, log_dir, checkpoint_provider,
                 steps=0, post_metrics=None):
        self._model = _validate_not_none(model, 'model')
        self._log_dir = _validate_not_none(log_dir, 'log_dir')
        self._provider = _validate_not_none(checkpoint_provider, 'checkpoint_provider')
        self._steps = _validate_not_none(steps, 'steps')
        self._post_metrics = post_metrics or {}
        if not self._post_metrics:
            logging.debug('no post metrics set.')
        for key, _ in self._post_metrics.items():
            logging.log(HDEBUG, 'post metric: %s', key)

        self._global_step = None
        self._eval_step = None
        self._step_fetches = None
        self._latest_checkpoint = None
        self._saver = None
        self._writer = None

    def _eval(self):
        logging.info('evaluating the checkpoint: %s', self._latest_checkpoint)
        self._global_step = None
        self._eval_step = None
        self._step_fetches = None

        logging.debug('setting up session configuration')
        logging.debug('allow softt device placement: %s', str(_ALLOW_SOFT_DEV_PLACEMENT))
        config = tf.ConfigProto(allow_soft_placement=_ALLOW_SOFT_DEV_PLACEMENT)

        with tf.Session(config=config, graph=self._model.graph) as sess:
            logging.debug('initializing global and local variables')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            logging.debug('restoring session.')
            self._saver.restore(sess, self._latest_checkpoint)

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
                    eval_loop = self._step(sess)
            except tf.errors.OutOfRangeError as ex:
                logging.info('a tf.errors.OutOfRangeError is stopping the loop.')
                coord.request_stop(ex=ex)
            finally:
                logging.debug('stopping the loop.')
                coord.request_stop()
                coord.join(threads)
                self._summarize(sess)
        logging.info('evaluation of checkpoint %s complete.', self._latest_checkpoint)

    def _summarize(self, sess):
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

        logging.debug('creating metrics summaries.')
        for key, value in metrics_avg.items():
            values.append(tf.summary.Summary.Value(tag=key, simple_value=value))

        logging.debug('writing the summaries.')
        summary = tf.summary.Summary(value=values)
        self._writer.add_summary(summary, global_step=self._global_step)
        self._writer.flush()

        logging.info(
            self._summary_msg(
                self._global_step, metrics_avg, self._latest_checkpoint))


    def _step(self, sess):
        if self._eval_step is None:
            logging.info('initializing evaluation step.')
            self._eval_step = 0
            logging.debug('getting streaming average metrics update_ops.')
            metrics_update_ops = {}
            for key, metric in self._model.metrics.items():
                metrics_update_ops[key] = metric.update_op
            logging.debug('initializing step fetches.')
            self._step_fetches = [
                metrics_update_ops,  # it's a dictionary!
                self._model.target,
                self._model.output,
                self._model.inputs[self._model.FORMULA_LENGTH_KEY],
            ]
            for key, pmetric in self._post_metrics.items():
                logging.debug('resetting post metric %s', key)
                pmetric.reset()

        metrics, targets, predictions, lengths = sess.run(self._step_fetches)
        for key, pmetric in self._post_metrics.items():
            logging.debug('accumulating post metric: %s', key)
            curr = pmetric.compute(targets, predictions, lengths)
            metrics[key] = curr

        logging.debug(
            self._summary_msg(
                self._eval_step, metrics, self._latest_checkpoint))

        self._eval_step += 1
        next_step = self._steps == 0 or self._eval_step < self._step
        logging.debug('next evaluation step? %s', str(next_step))
        return next_step

    def start(self):
        """Run the evaluation loop."""

        with self._model.graph.as_default() as graph:
            logging.debug('initializing session saver.')
            self._saver = tf.train.Saver()
            logging.debug('initializing the summary writer to path %s.', self._log_dir)
            self._writer = tf.summary.FileWriter(self._log_dir, graph=graph)
            logging.debug('flushing writer.')
            self._writer.flush()

        logging.info('starting running the eval loop.')
        while True:
            logging.debug('polling the provider for the latest checkpoint.')
            self._latest_checkpoint = self._provider.next()
            if self._latest_checkpoint is None:
                logging.info('no more checkpoints available.')
                break
            self._eval()
        logging.info('eval loop complete.')

    def _summary_msg(self, step, metrics, checkpoint):
        return ', '.join(
            ['step: {}'.format(step)] +
            ['{}: {:.2f}'.format(key, value) for key, value in metrics.items()]+
            ['checkpoint: {}'.format(checkpoint)])


class CheckpointProvider(object):
    """Provides the latest checkpoint for evaluation.

    This class provides access to the latest checkpoint from a certain
    checkpoint directory, through the `next()` method. If no checkpoint
    is found (other than the latest one, from a previous check) in a certain
    amount of time, `None` is returned.
    """

    def __init__(self, checkpoint_dir, idle_time=10, max_idle_time=300):
        self._checkpoint_dir = _validate_not_none(checkpoint_dir, 'checkpoint_dir')
        self._idle_time = _validate_not_none(idle_time, 'idle_time')
        self._max_idle_time = _validate_not_none(max_idle_time, 'max_idle_time')
        self._tot_idle_time = 0.0
        self._latest_checkpoint = None
        self._latest_timestamp = None

    @property
    def checkpoint_dir(self):
        """The checkpoint directory."""
        return self._checkpoint_dir

    @property
    def idle_time(self):
        """The idle time (in sec.) when polling the checkpoint directory."""
        return self._idle_time

    @property
    def max_idle_time(self):
        """The maximum idle time (in sec.) when polling the checkpoint directory."""
        return self._max_idle_time

    def _get_latest_checkpoint(self):
        logging.debug('loding latest checkpoint from %s', self._checkpoint_dir)
        checkpoint = tf.train.latest_checkpoint(self._checkpoint_dir)
        if checkpoint is None:
            logging.warning('No checkpoint.')
            return None
        if checkpoint == self._latest_checkpoint:
            logging.warning('Not evaluating checkpoint %s', checkpoint)
            return None
        logging.info('valid checkpoint: %s', checkpoint)
        return checkpoint

    def _sleep(self):
        logging.debug('sleeping for %d seconds', self._idle_time)
        time.sleep(self._idle_time)
        logging.log(HDEBUG, 'waking up')
        now = time.time()
        self._tot_idle_time = now - self._latest_timestamp
        logging.debug(
            'now it is %s, last timestamp was %s',
            _timestamp_fmt(now), _timestamp_fmt(self._latest_timestamp))
        logging.info(
            'total idle time: %.2f (on maximum %d) sec.',
            self._tot_idle_time, self._max_idle_time)
        if self._tot_idle_time < self._max_idle_time:
            return True
        logging.warning('total idle timeout: %.2f sec.; stop looping.', self._tot_idle_time)
        return False

    def next(self):
        """Returns the next checkpoint.

        provides access to the latest checkpoint from a certain checkpoint directory.
        If no checkpoint is found (other than the latest one, from a previous check)
        in a certain amount of time, `None` is returned.
        """

        checkpoint = None
        logging.debug('polling %s for the latest checkpoint.', self._checkpoint_dir)
        self._latest_timestamp = time.time()
        logging.debug('setting timestamp at %s', _timestamp_fmt(self._latest_timestamp))
        logging.debug('setting total idle time at 0.0 sec.')
        self._tot_idle_time = 0.0

        while True:
            checkpoint = self._get_latest_checkpoint()
            if checkpoint or not self._sleep():
                break

        # save and return the latest checkpoint
        logging.debug('setting and returning the latest checkpoint.')
        self._latest_checkpoint = checkpoint
        return checkpoint
