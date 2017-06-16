"""Runtime utilities."""

import logging

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

    def __init__(self, model, log_dir, steps=0, save_every=0):
        logging.debug('initializing TrainLoop instance.')
        self._model = _validate_not_none(model, 'model')
        self._log_dir = _validate_not_none(log_dir, 'log_dir')
        self._steps = _validate_not_none(steps, 'steps')
        self._save_every = _validate_not_none(save_every, 'save_every')
        logging.log(HDEBUG, 'setting loop flag to `False`')
        self._loop_flag = False

    def start(self):
        """Start the train loop."""
        logging.info('starting train loop.')
        with tf.Session() as sess:
            logging.debug('initializing local and global variables.')
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            logging.debug('initializing coordinator and starting queue runners.')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                self._loop_flag = True
                while self._loop_flag:
                    pass
            except tf.errors.OutOfRangeError as ex:
                logging.debug('a tf.errors.OutOfRangeError is stopping the loop.')
                coord.request_stop(ex=ex)
            finally:
                coord.request_stop()
                coord.join(threads)
        logging.info('training loop complete.')

# class EvalLoop(object):

#     def __init__(self):
#         pass
