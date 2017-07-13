"""Test module for the `dket.models.model` module."""

# pylint: disable=E1129

import copy
import mock

import tensorflow as tf

from dket.models import model

class _BaseModel(model._Model):  # pylint: disable=W0212

    _TARGET_KEY = 'TARGET'

    def __init__(self, summary=True):
        super(_BaseModel, self).__init__()
        self._summary = summary
        self._tensors = None
        self._logits = None

    def get_default_hparams(self):
        return tf.contrib.training.HParams(dim_0=10, dim_1=3, dim_2=7)

    def _feed_helper(self, tensors):
        self._tensors = copy.copy(tensors)
        self._inputs = copy.copy(tensors)
        self._target = self._inputs.pop(self._TARGET_KEY)
        self._output_mask = None

    def _build_graph(self):
        assert not self.built
        shape = [self.hparams.dim_0, self.hparams.dim_1, self.hparams.dim_2]
        self._logits = tf.random_normal(shape, name='Logits')
        self._predictions = tf.identity(
            tf.nn.softmax(self._logits),
            name='Probabilities')
        if self._summary:
            tf.summary.scalar('SimpleSummary', tf.constant(23))


class TestBaseModel(tf.test.TestCase):
    """Test the functionality of the `dket.models.model.BaseModel` class."""

    def test_global_step_initialization(self):
        """Global step is set right after the model creation."""
        instance = _BaseModel()
        self.assertIsNotNone(instance.global_step)
        self.assertFalse(instance.fed)
        self.assertFalse(instance.built)
        self.assertIsNone(instance.hparams)
        self.assertIsNone(instance.loss)
        self.assertIsNone(instance.optimizer)
        self.assertIsNone(instance.metrics)
        self.assertIsNone(instance.inputs)
        self.assertIsNone(instance.target)
        self.assertIsNone(instance.predictions)
        self.assertIsNone(instance.train_op)
        self.assertIsNone(instance.summary_op)

    def test_get_default_hparams(self):
        """The method `get_default_hparams` should be invocable as the model is created."""
        instance = _BaseModel()
        self.assertIsNotNone(instance.get_default_hparams())
        self.assertIsNotNone(instance.global_step)
        self.assertFalse(instance.fed)
        self.assertFalse(instance.built)
        self.assertIsNone(instance.hparams)
        self.assertIsNone(instance.loss)
        self.assertIsNone(instance.optimizer)
        self.assertIsNone(instance.metrics)
        self.assertIsNone(instance.inputs)
        self.assertIsNone(instance.target)
        self.assertIsNone(instance.predictions)
        self.assertIsNone(instance.train_op)
        self.assertIsNone(instance.summary_op)

    def test_feed(self):
        """Feed the model with tensors."""
        instance = _BaseModel()
        with tf.variable_scope('Inputs'):
            inputs = {
                'A': tf.constant(23, dtype=tf.int32),
                'B': tf.constant(47, dtype=tf.int32),
            }
            target = tf.constant(90, dtype=tf.int32)

            tensors = copy.copy(inputs)
            tensors['TARGET'] = target

        instance.feed(tensors)
        self.assertTrue(instance.fed)
        self.assertEqual(inputs, instance.inputs)
        self.assertEqual(target, instance.target)
        self.assertEqual(tensors, instance.feeding)

        self.assertFalse(instance.built)
        self.assertIsNone(instance.hparams)
        self.assertIsNone(instance.loss)
        self.assertIsNone(instance.optimizer)
        self.assertIsNone(instance.metrics)
        self.assertIsNone(instance.predictions)
        self.assertIsNone(instance.train_op)
        self.assertIsNone(instance.summary_op)

        # If you try feeding the model twice, you
        # will have a RuntimeError.
        self.assertRaises(RuntimeError, instance.feed, tensors)

    def test_feed_with_none_args(self):
        """Test feeding the model with `None` inputs or target."""
        instance = _BaseModel()
        self.assertRaises(ValueError, instance.feed, tensors=None)

    def test_build_trainable(self):
        """Test the building of a trainable model."""

        instance = _BaseModel()
        with tf.variable_scope('Inputs'):
            tensors = {
                'A': tf.constant(23, dtype=tf.int32),
                'B': tf.constant(47, dtype=tf.int32),
                'TARGET': tf.constant(90, dtype=tf.int32)
            }
        instance.feed(tensors)

        hparams = tf.contrib.training.HParams(dim_0=2, dim_1=4, extra='Ciaone')

        loss_batch_value = tf.constant(0.0, dtype=tf.float32)
        loss = mock.Mock()
        type(loss).batch_value = mock.PropertyMock(return_value=loss_batch_value)

        optimizer = mock.Mock()
        train_op = tf.no_op('train_op')
        optimizer.minimize.side_effect = [train_op]

        metrics_01 = mock.Mock()
        metrics_02 = mock.Mock()

        metrics = {
            'metrics_01': metrics_01,
            'metrics_02': metrics_02
        }

        instance.build(hparams, loss, optimizer, metrics)

        self.assertTrue(instance.built)
        self.assertTrue(instance.trainable)
        self.assertEqual(hparams.dim_0, instance.hparams.dim_0)
        self.assertEqual(hparams.dim_1, instance.hparams.dim_1)
        self.assertEqual(instance.get_default_hparams().dim_2,
                         instance.hparams.dim_2)
        self.assertFalse('extra' in instance.hparams.values())
        self.assertIsNotNone(instance.predictions)

        loss.compute.assert_called_once_with(
            instance.target, instance.predictions, weights=None)

        optimizer.minimize.assert_called_once_with(
            loss_batch_value, global_step=instance.global_step)
        self.assertEqual(train_op, instance.train_op)

        metrics_01.compute.assert_called_once_with(
            instance.target, instance.predictions, weights=None)
        metrics_02.compute.assert_called_once_with(
            instance.target, instance.predictions, weights=None)

        self.assertIsNotNone(instance.summary_op)

        self.assertRaises(RuntimeError, instance.build,
                          hparams, loss, optimizer, metrics)

    def test_build_not_trainable_loss(self):
        """Test the building of a non-trainable model with loss."""

        instance = _BaseModel()
        with tf.variable_scope('Inputs'):
            tensors = {
                'A': tf.constant(23, dtype=tf.int32),
                'B': tf.constant(47, dtype=tf.int32),
                'TARGET': tf.constant(90, dtype=tf.int32)
            }
        instance.feed(tensors)

        hparams = tf.contrib.training.HParams(dim_0=2, dim_1=4, extra='Ciaone')

        loss = mock.Mock()

        metrics_01 = mock.Mock()
        metrics_02 = mock.Mock()

        metrics = {
            'metrics_01': metrics_01,
            'metrics_02': metrics_02
        }

        self.assertRaises(ValueError, instance.build, 
                          hparams, loss=loss, optimizer=None)

    def test_build_not_trainable(self):
        """Test the building of a non-trainable model without loss."""
        instance = _BaseModel()
        with tf.variable_scope('Inputs'):
            tensors = {
                'A': tf.constant(23, dtype=tf.int32),
                'B': tf.constant(47, dtype=tf.int32),
                'TARGET': tf.constant(90, dtype=tf.int32),
            }
        instance.feed(tensors)

        hparams = tf.contrib.training.HParams(dim_0=2, dim_1=4, extra='Ciaone')
        instance.build(hparams, loss=None, optimizer=None, metrics=None)

        self.assertFalse(instance.trainable)
        self.assertIsNone(instance.loss)
        self.assertIsNone(instance.optimizer)
        self.assertIsNone(instance.train_op)
        self.assertIsNone(instance.summary_op)

    def test_build_trainable_without_loss(self):  # pylint: disable=I0011,C0103
        """Built a model with an optimizer but without a loss function."""

        instance = _BaseModel()
        with tf.variable_scope('Inputs'):
            tensors = {
                'A': tf.constant(23, dtype=tf.int32),
                'B': tf.constant(47, dtype=tf.int32),
                'TARGET': tf.constant(90, dtype=tf.int32)
            }
        instance.feed(tensors)

        hparams = tf.contrib.training.HParams(dim_0=2, dim_1=4, extra='Ciaone')

        optimizer = mock.Mock()
        train_op = tf.no_op('train_op')
        optimizer.minimize.side_effect = [train_op]

        self.assertRaises(ValueError, instance.build,
                          hparams, loss=None, optimizer=optimizer)

    def test_build_not_fed(self):
        """Build a model which has not been fed."""
        instance = _BaseModel()
        hparams = instance.get_default_hparams()
        self.assertFalse(instance.fed)
        self.assertRaises(RuntimeError, instance.build, hparams)

    def test_build_trainable_without_summaries(self):  # pylint: disable=I0011,C0103
        """Test that a trainable model always has a summary_op."""
        instance = _BaseModel(summary=False)
        with tf.variable_scope('Inputs'):
            tensors = {
                'A': tf.constant(23, dtype=tf.int32),
                'B': tf.constant(47, dtype=tf.int32),
                'TARGET': tf.constant(90, dtype=tf.int32)
            }
        instance.feed(tensors)

        hparams = tf.contrib.training.HParams(dim_0=2, dim_1=4, extra='Ciaone')

        loss = mock.Mock()

        optimizer = mock.Mock()
        train_op = tf.no_op('train_op')
        optimizer.minimize.side_effect = [train_op]

        instance.build(hparams, loss, optimizer)
        self.assertIsNone(tf.summary.merge_all())
        self.assertIsNotNone(instance.summary_op)

    def test_build_without_hparams(self):
        """Test the building of a model without hparams."""
        instance = _BaseModel(summary=False)
        with tf.variable_scope('Inputs'):
            tensors = {
                'A': tf.constant(23, dtype=tf.int32),
                'B': tf.constant(47, dtype=tf.int32),
                'TARGET': tf.constant(90, dtype=tf.int32)
            }
        instance.feed(tensors)

        loss = mock.Mock()

        optimizer = mock.Mock()
        train_op = tf.no_op('train_op')
        optimizer.minimize.side_effect = [train_op]

        self.assertRaises(ValueError, instance.build, None,
                          loss=loss, optimizer=optimizer)

    def test_build_without_metrics(self):
        """Test the building without metrics."""
        instance = _BaseModel(summary=False)
        with tf.variable_scope('Inputs'):
            tensors = {
                'A': tf.constant(23, dtype=tf.int32),
                'B': tf.constant(47, dtype=tf.int32),
                'TARGET': tf.constant(90, dtype=tf.int32)
            }
        instance.feed(tensors)

        loss = mock.Mock()

        optimizer = mock.Mock()
        train_op = tf.no_op('train_op')
        optimizer.minimize.side_effect = [train_op]

        instance.build(instance.get_default_hparams(), loss, optimizer)
        self.assertIsNone(instance.metrics)


if __name__ == '__main__':
    tf.test.main()
