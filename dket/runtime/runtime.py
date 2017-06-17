"""Runtime utilities."""

import logging
import os

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
        save_step = self._checkpoint_every == 0 or ((step + 1) % self._checkpoint_every == 0)
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

    def __init__(self, model, log_dir, checkpoint_dir,
                 eval_check_every_secs=300, eval_check_until_secs=3600):
        self._model = _validate_not_none(model, 'model')
        self._log_dir = _validate_not_none(log_dir, 'log_dir')
        self._checkpoint_dir = _validate_not_none(checkpoint_dir, 'checkpoint_dir')
        self._eval_check_every_secs = _validate_not_none(
            eval_check_every_secs, 'eval_check_every_secs')
        self._eval_check_until_secs = _validate_not_none(
            eval_check_until_secs, 'eval_check_until_secs')
        self._latest_gstep = -1

    def start(self):
        """Run the evaluation loop."""
        logging.info('eval loop!')
        print(tf.train.latest_checkpoint(self._checkpoint_dir))
