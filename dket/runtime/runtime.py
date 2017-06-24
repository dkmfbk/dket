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


_MAX_CHECKPOINTS_TO_KEEP = 20
_ALLOW_SOFT_DEV_PLACEMENT = True


class TrainLoop(object):
    """Train loop."""

    def __init__(self, model, log_dir, steps=0, checkpoint_every=0):
        logging.debug('initializing TrainLoop instance.')
        self._model = _validate_not_none(model, 'model')
        self._log_dir = _validate_not_none(log_dir, 'log_dir')
        self._steps = _validate_not_none(steps, 'steps')
        self._checkpoint_every = _validate_not_none(checkpoint_every, 'checkpoint_every')
        self._checkpoint_name = os.path.join(self._log_dir, 'CHECKPOINT')
        logging.info('TrainLoop initialized. Checkpoint at %s.', self._checkpoint_name)

        self._next_step = False
        self._saver = None
        self._writer = None
        self._fetches = None
        self._last_checkpoint_ts = None

    def start(self):
        """Start the train loop."""
        logging.deug('started running the train loop.')
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
                metrics_t,  # it's a dictionary.
            ]

        logging.log(HDEBUG, 'running the session step.')
        step, _, loss, summary, metrics = sess.run(self._fetches)
        logging.log(HDEBUG, 'writing summaries.')
        self._writer.add_summary(summary, global_step=step)
        self._writer.flush()

        save_step = self._checkpoint_every == 0 or (step % self._checkpoint_every == 0)
        logging.log(HDEBUG, 'is a checkpoint step? %s.', 'yes' if save_step else 'no')
        if save_step:
            checkpoint, delta = self._save_checkpoint(sess, self._checkpoint_name, step)
            message = ', '.join(
                [self._summary_msg(step, loss, metrics),
                 self._checkpoint_msg(checkpoint, delta)])
            logging.info(message)
        elif logging.getLogger().getEffectiveLevel() <= HDEBUG:
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

    _TS_FMT = '%Y-%m-%d %H:%M:%S.%3d'

    def __init__(self, model, log_dir, checkpoint_dir, steps=0,
                 eval_check_every_secs=300, eval_check_until_secs=3600):
        self._model = _validate_not_none(model, 'model')
        self._log_dir = _validate_not_none(log_dir, 'log_dir')
        self._checkpoint_dir = _validate_not_none(checkpoint_dir, 'checkpoint_dir')
        self._steps = _validate_not_none(steps, 'steps')
        self._eval_check_every_secs = _validate_not_none(
            eval_check_every_secs, 'eval_check_every_secs')
        self._eval_check_until_secs = _validate_not_none(
            eval_check_until_secs, 'eval_check_until_secs')
        self._latest_gstep = -1
        self._latest_checkpoint = None
        self._latest_timestamp = time.time()
        self._total_idle_time = 0
        self._main_loop_flag = False
        self._eval_loop_flag = False
        self._eval_step = 0
        with self._model.graph.as_default():
            logging.debug('initializing the saver.')
            self._saver = tf.train.Saver(max_to_keep=_MAX_CHECKPOINTS_TO_KEEP)
        logging.debug('initializing the file writer and flushing the graph definition.')
        self._writer = tf.summary.FileWriter(self._log_dir, graph=self._model.graph)
        self._writer.flush()
        logging.debug('finished initialization.')

    def _tsfmt(self, timestamp):
        return datetime.fromtimestamp(timestamp).strftime(self._TS_FMT)

    def _sleep(self):
        logging.debug('sleeping for %d seconds', self._eval_check_every_secs)
        time.sleep(self._eval_check_every_secs)
        logging.log(HDEBUG, 'waking up.')
        now = time.time()
        self._total_idle_time = now - self._latest_timestamp
        logging.debug('now it is %s, last timestamp was %s',
                      self._tsfmt(now), self._tsfmt(self._latest_timestamp))
        logging.info('total idle time: %f (on maximum %d) sec.',
                     self._total_idle_time, self._eval_check_until_secs)
        if self._total_idle_time < self._eval_check_until_secs:
            return True
        logging.warning('total idle timeout: %f sec.; stop looping.', self._total_idle_time)
        return False

    def _reset(self):
        self._total_idle_time = 0.0
        self._latest_timestamp = time.time()
        logging.debug('resetting idle time to 0.0 and timestamp to %s',
                      self._tsfmt(self._latest_timestamp))
        logging.debug('resetting eval step.')
        self._eval_step = 0

    def _eval(self, checkpoint):
        logging.info('evaluating the checkpoint: %s', checkpoint)
        self._latest_checkpoint = checkpoint
        self._eval_loop_flag = True
        logging.debug('initializing evaluation session.')
        config = tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True)
        with tf.Session(config=config, graph=self._model.graph) as sess:
            logging.debug('initializing global and local variables')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            logging.debug('restoring session.')
            self._saver.restore(sess, checkpoint)

            logging.debug('initializing coordinator and starting queue runners.')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                logging.debug('starting the train loop.')
                global_step = -1
                while self._eval_loop_flag:
                    global_step, self._eval_loop_flag = self._step(sess)
            except tf.errors.OutOfRangeError as ex:
                logging.info('a tf.errors.OutOfRangeError is stopping the loop.')
                coord.request_stop(ex=ex)
            finally:
                logging.debug('stopping the loop.')
                coord.request_stop()
                coord.join(threads)
                self._summarize(sess)
        logging.debug('evaluation loop complete.')

    def _summarize(self, sess):
        values = []

        logging.debug('getting average metrics values.')
        metrics_avg_t = {}
        for key, metric in self._model.metrics.items():
            metrics_avg_t[key] = metric.value

        logging.debug('evaluating average loss and metrics.')
        fetches = [self._model.global_step, self._model.loss.value, metrics_avg_t]
        global_step, loss, metrics_avg = sess.run(fetches)

        logging.debug('creating loss summary.')
        values.append(tf.summary.Summary.Value(tag='loss', simple_value=loss))
        logging.debug('creating metrics summaries.')
        for key, value in metrics_avg.items():
            values.append(tf.summary.Summary.Value(tag=key, simple_value=value))

        logging.debug('writing the summaries.')
        summary = tf.summary.Summary(value=values)
        self._writer.add_summary(summary, global_step=global_step)
        self._writer.flush()

        logging.info(
            self._log_line(
                'evaluation:', self._eval_step, global_step, loss, metrics_avg, ''))

    def _step(self, sess):
        logging.debug('getting streaming average metrics update_ops.')
        metrics_update_ops = {}
        for key, metric in self._model.metrics.items():
            metrics_update_ops[key] = metric.update_op
        fetches = [
            self._model.global_step,
            self._model.loss.value,
            metrics_update_ops  # it's a dictionary!
        ]
        global_step, loss, metrics = sess.run(fetches)
        self._eval_step += 1
        if logging.getLogger().getEffectiveLevel() <= HDEBUG:
            logging.log(HDEBUG, self._log_line(
                '', self._eval_step, global_step, loss, metrics, ''))
        cont = self._steps == 0 or self._eval_step < self._step
        return global_step, cont

    def start(self):
        """Run the evaluation loop."""
        logging.debug('starting the eval loop.')
        self._main_loop_flag = True
        while self._main_loop_flag:
            checkpoint = tf.train.latest_checkpoint(self._checkpoint_dir)
            if checkpoint and checkpoint != self._latest_checkpoint:
                self._eval(checkpoint)
                self._reset()
            else:
                if checkpoint is None:
                    logging.warning('No checkpoint yet.')
                else:
                    logging.warning('Not evaluating: checkpoint %s', checkpoint)
                self._main_loop_flag = self._sleep()
                logging.debug('still looping? %s', str(self._main_loop_flag))
        logging.debug('eval loop complete.')

    def _log_line(self, pre, step, global_step, loss, metrics, post):
        components = []
        if pre:
            components.append(pre)
        components.append('step: {}@{}'.format(step, global_step))
        components.append('loss: {}'.format(loss))
        for key, value in metrics.items():
            components.append('{}: {}'.format(key, value))
        if post:
            components.append(post)
        return ', '.join(components)
        