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


class TrainLoop(object):
    """Train loop"""

    def __init__(self, model, log_dir, steps=0, checkpoint_every=0):
        logging.debug('initializing TrainLoop instance.')
        self._model = _validate_not_none(model, 'model')
        self._log_dir = _validate_not_none(log_dir, 'log_dir')
        self._steps = _validate_not_none(steps, 'steps')
        self._checkpoint_every = _validate_not_none(checkpoint_every, 'checkpoint_every')
        logging.log(HDEBUG, 'setting loop flag to `False`')
        self._loop_flag = False
        self._checkpoint_name = os.path.join(self._log_dir, 'CHECKPOINT')
        logging.debug('the checkpoint name will be: %s', self._checkpoint_name)
        logging.debug('initializing saver.')
        self._saver = tf.train.Saver()  # TODO(petrx): check options
        logging.debug('initializing the file writer and flushing the graph definition.')
        self._writer = tf.summary.FileWriter(self._log_dir, graph=self._model.graph)
        self._writer.flush()
        logging.debug('finished initialization.')


    def start(self):
        """Start the train loop."""
        logging.info('starting train loop.')
        config = tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            logging.debug('initializing local and global variables.')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            logging.debug('initializing coordinator and starting queue runners.')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                logging.info('starting the train loop.')
                self._loop_flag = True
                while self._loop_flag:
                    step, self._loop_flag = self.step(sess)
            except tf.errors.OutOfRangeError as ex:
                logging.debug('a tf.errors.OutOfRangeError is stopping the loop.')
                coord.request_stop(ex=ex)
                logging.debug('saving latest checkpoint')
                self._saver.save(sess, self._checkpoint_name, global_step=step)
            finally:
                logging.info('stopping the loop.')
                coord.request_stop()
                coord.join(threads)
        logging.info('training loop complete.')

    def step(self, sess):
        """The next training step.

        Returns:
          a tuple `global_step, continue` where:
            `global_step` is an int indicating the current value of the global step,
            `continue` is True if the number current step has reached the number of
              steps set for the training.

        Raises:
          tf.errors.OutOfRangeError: if the input queues has finished their job.
        """
        fetches = [
            self._model.global_step,
            self._model.train_op,
            self._model.loss_op,
            self._model.summary_op,
            self._model.metrics_ops,  # it's a dictionary!
        ]
        step, _, loss, summary, metrics = sess.run(fetches)
        self._writer.add_summary(summary, global_step=step)
        save_step = self._checkpoint_every == 0 or (step % self._checkpoint_every == 0)
        if save_step:
            logging.debug('saving checkpoint at step %d', step)
            checkpoint = self._saver.save(sess, self._checkpoint_name, global_step=step)
            message = self._log_line(
                '', step, loss, metrics,
                'saved checkpoint: {}'.format(checkpoint))
            logging.info(message)
        elif logging.getLogger().getEffectiveLevel() <= HDEBUG:
            logging.log(HDEBUG, self._log_line('', step, loss, metrics, ''))
        cont = self._steps == 0 or step < self._steps
        return step, cont

    def _log_line(self, pre, step, loss, metrics, post):
        components = []
        if pre:
            components.append(pre)
        components.append('global step: {}'.format(step))
        components.append('loss: {}'.format(loss))
        for key, value in metrics.items():
            components.append('{}: {}'.format(key, value))
        if post:
            components.append(post)
        return ', '.join(components)


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
        self._accumulate = {}
        logging.debug('building metrics accumulators.')
        for key in self._model.metrics_ops.keys():
            logging.log(HDEBUG, 'adding accumulator for %s', key)
            self._accumulate[key] = []
        logging.debug('building loss accumulator.')
        self._losses = []
        self._eval_step = 0
        logging.debug('initializing the saver.')
        self._saver = tf.train.Saver()
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
        logging.debug('total idle time: %f (on maximum %d)',
                      self._total_idle_time, self._eval_check_until_secs)
        if self._total_idle_time < self._eval_check_until_secs:
            return True
        logging.warning('total idle timeout: %f', self._total_idle_time)
        return False

    def _reset(self):
        self._total_idle_time = 0.0
        self._latest_timestamp = time.time()
        logging.debug('resetting idle time to 0.0 and timestamp to %s',
                      self._tsfmt(self._latest_timestamp))
        logging.debug('resetting eval step.')
        self._eval_step = 0
        logging.debug('resetting loss accumulator.')
        self._losses.clear()
        logging.debug('resetting metrics accumulator.')
        for _, value in self._accumulate.items():
            value.clear()

    def _eval(self, checkpoint):
        logging.info('evaluating the checkpoint: %s', checkpoint)
        self._latest_checkpoint = checkpoint
        self._eval_loop_flag = True
        global_step = -1
        config = tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True)
        with tf.Session(config=config) as sess:
            logging.debug('initializing global and local variables')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            logging.info('restoring session.')
            self._saver.restore(sess, checkpoint)

            logging.debug('initializing coordinator and starting queue runners.')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                logging.info('starting the train loop.')
                while self._eval_loop_flag:
                    global_step, self._eval_loop_flag = self._step(sess)
            except tf.errors.OutOfRangeError as ex:
                logging.debug('a tf.errors.OutOfRangeError is stopping the loop.')
                coord.request_stop(ex=ex)
            finally:
                logging.info('stopping the loop.')
                coord.request_stop()
                coord.join(threads)
                self._summarize(global_step)
        logging.info('evaluation loop complete.')

    def _avg(self, items):
        sum = 0.0
        for item in items:
            sum += item * 1.0
        return sum / len(items)

    def _summarize(self, global_step):
        values = []
        logging.debug('saving average loss.')
        loss = self._avg(self._losses)
        values.append(tf.summary.Summary.Value(tag='loss', simple_value=loss))
        for key, value in self._accumulate.items():
            values.append(tf.summary.Summary.Value(tag=key, simple_value=self._avg(value)))
        summary = tf.summary.Summary(value=values)
        self._writer.add_summary(summary, global_step=global_step)
        self._writer.flush()

    def _step(self, sess):
        fetches = [
            self._model.global_step,
            self._model.loss_op,
            self._model.metrics_ops,
        ]
        global_step, loss, metrics = sess.run(fetches)
        self._eval_step += 1
        if logging.getLogger().getEffectiveLevel() <= HDEBUG:
            logging.log(HDEBUG, self._log_line(
                '', self._eval_step, global_step, loss, metrics, ''))
        logging.debug('updating accumulators.')
        self._losses.append(loss)
        for key, value in metrics.items():
            self._accumulate[key].append(value)
        cont = self._steps == 0 or self._eval_step < self._step
        return global_step, cont

    def start(self):
        """Run the evaluation loop."""
        logging.info('starting the eval loop.')
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
        logging.info('eval loop complete.')

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
        