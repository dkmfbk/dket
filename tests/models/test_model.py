"""Test module for the `dket.models.model` module."""

# pylint: disable=E1129

import argparse
import copy
import logging
import mock
import os
import tempfile
import random

import tensorflow as tf

from dket import data
from dket.models.model import Model, ModelInputs

TRAIN = tf.contrib.learn.ModeKeys.TRAIN
EVAL = tf.contrib.learn.ModeKeys.EVAL
INFER = tf.contrib.learn.ModeKeys.INFER


class TestDataFactory(object):
    """Test data factory."""

    WORDS_MIN_LEN = 5
    WORDS_MAX_LEN = 12
    FORMULA_MIN_LEN = 3
    FORMULA_MAX_LEN = 6
    FILE_PREFIX = 'file-'
    FILE_EXT = '.rio'

    def __init__(self):
        self._tdir = tempfile.TemporaryDirectory()

    def _example(self):
        wlen = range(random.randint(self.WORDS_MIN_LEN, self.WORDS_MAX_LEN))
        words = [random.randint(1, 9) for _ in wlen] + [0]
        flen = range(random.randint(self.FORMULA_MIN_LEN, self.FORMULA_MAX_LEN))
        formula = [random.randint(1, 9) for _ in flen] + [0]
        return words, formula

    def _write(self, fpath, num_examples):
        with tf.python_io.TFRecordWriter(fpath) as writer:
            for _ in range(num_examples):
                example = data.encode(*self._example())
                writer.write(example.SerializeToString())

    def generate(self, num_files=1, num_examples=10):
        """Generate `num_files` data files, each with `num_examples` examples."""
        files = []
        for i in range(num_files):
            fname = self.FILE_PREFIX + str(i) + self.FILE_EXT
            fpath = os.path.join(self._tdir.name, fname)
            self._write(fpath, num_examples)
            files.append(fpath)
        return files


class TestModelInputs(tf.test.TestCase):
    """ModelInputs test case."""

    def test_no_params(self):
        """Build a ModelInputs instance without params."""
        modin = ModelInputs(TRAIN, {})
        self.assertIsNotNone(modin.get(modin.WORDS_KEY))
        self.assertIsNotNone(modin.get(modin.SENTENCE_LENGTH_KEY))
        self.assertIsNotNone(modin.get(modin.FORMULA_KEY))
        self.assertIsNotNone(modin.get(modin.FORMULA_LENGTH_KEY))

    def test_no_files(self):
        """No input files."""
        params = {
            ModelInputs.FILES_PK: '',
            ModelInputs.EPOCHS_PK: -1,
            ModelInputs.BATCH_SIZE_PK: 1,
            ModelInputs.SHUFFLE_PK: True,
            ModelInputs.SEED_PK: None
        }
        modin = ModelInputs(TRAIN, params)
        self.assertIsNotNone(modin.get(modin.WORDS_KEY))
        self.assertIsNotNone(modin.get(modin.SENTENCE_LENGTH_KEY))
        self.assertIsNotNone(modin.get(modin.FORMULA_KEY))
        self.assertIsNotNone(modin.get(modin.FORMULA_LENGTH_KEY))

    def test_epochs_lessthan_zero(self):
        """If epochs < 0 an exception is raised."""
        params = {
            ModelInputs.FILES_PK: 'ciao.txt',
            ModelInputs.EPOCHS_PK: -1
        }
        self.assertRaises(ValueError, ModelInputs, TRAIN, params)

    def test_batch_lesseqthan_zero(self):
        """If batch_size <= 0 an exception is raised."""
        params = {
            ModelInputs.FILES_PK: 'ciao.txt',
            ModelInputs.BATCH_SIZE_PK: 0
        }
        self.assertRaises(ValueError, ModelInputs, TRAIN, params)
        
        params = {
            ModelInputs.FILES_PK: 'ciao.txt',
            ModelInputs.BATCH_SIZE_PK: -1
        }
        self.assertRaises(ValueError, ModelInputs, TRAIN, params)

    def test_read_from_files(self):
        """Read from data files."""
        tdf = TestDataFactory()
        files = tdf.generate(num_files=2, num_examples=10)

        params = ModelInputs.get_default_params()
        params[ModelInputs.FILES_PK] = ','.join(files)
        params[ModelInputs.EPOCHS_PK] = None
        params[ModelInputs.BATCH_SIZE_PK] = 5

        modin = ModelInputs(TRAIN, params)
        self.assertIsNotNone(modin.get(modin.WORDS_KEY))
        self.assertIsNotNone(modin.get(modin.SENTENCE_LENGTH_KEY))
        self.assertIsNotNone(modin.get(modin.FORMULA_KEY))
        self.assertIsNotNone(modin.get(modin.FORMULA_LENGTH_KEY))


class TModel(Model):
    """Test model class."""

    DEFAULT_LR_KEY = 'optimizer.lr'
    DEFAULT_LR_VALUE = 0.01

    def __init__(self, mode, params):
        super(TModel, self).__init__(mode, params)
        self._data = None

    @classmethod
    def get_default_params(cls):
        dparams = super(TModel, cls).get_default_params()
        dparams.update({
            'name': 'TModel',
            'ciaone.proprio': True,
            'x': 2,
            'y': 3,
            'num_classes': 10,
            cls.DEFAULT_LR_KEY : cls.DEFAULT_LR_VALUE,
        })
        return dparams

    def _build_graph(self):
        shape = [self._params['x'], self._params['y']]
        self._data = tf.get_variable('DATA', shape=shape, dtype=tf.float32)
        coeff = tf.reduce_prod(self._data)
        self._predictions = coeff * tf.one_hot(
            self.inputs.get(ModelInputs.FORMULA_KEY), 
            self._params['num_classes'])

    @property
    def data(self):
        """Data variable."""
        return self._data

class TestModel(tf.test.TestCase):
    """Test case for the base model infrastructure."""

    def test_default_train(self): 
        """Default test."""

        tmodel = TModel.create(TRAIN, {})
        self.assertIsNone(tmodel.graph)
        self.assertIsNone(tmodel.global_step)
        self.assertIsNone(tmodel.inputs)
        self.assertIsNone(tmodel.loss_op)
        self.assertIsNone(tmodel.train_op)
        self.assertIsNone(tmodel.metrics)

        tmodel.build()
        self.assertIsNotNone(tmodel.graph)
        self.assertIsNotNone(tmodel.global_step)
        self.assertIsNotNone(tmodel.inputs)
        self.assertIsNotNone(tmodel.loss_op)
        self.assertIsNotNone(tmodel.train_op)
        self.assertIsNotNone(tmodel.metrics)
        

#     def test_factory(self):
#         """Test the factory method."""
#         params = {
#             'model.class': 'tests.models.test_model.TModel'
#         }
#         model = Model.factory(None, ModeKeys.EVAL, params)
#         self.assertEqual(type(model), TModel)

#     def test_invalid_mode(self):
#         """An invalid `mode` raises a ValueError."""
#         self.assertRaises(ValueError, TModel, None, 'InvalidMode', {})

#     def test_no_params(self):
#         """If no params are passed, default ones will be used."""
#         model = TModel(None, ModeKeys.INFER, {})
#         params = model.get_params()
#         dparams = TModel.get_default_params()
#         self.assertEqual(params, dparams)
#         self.assertEqual(params[TModel.DEFAULT_LR_KEY], TModel.DEFAULT_LR_VALUE)

#     def test_wrong_params(self):
#         """Adding a param with a unknown key, raises a value error."""
#         params = {
#             TModel.DEFAULT_LR_KEY: 23.0,
#             'this.key.is.totally.not.there': 'tricipiti.'
#         }
#         self.assertRaises(ValueError, TModel, None, ModeKeys.INFER, params)

#     def test_overload_params(self):
#         """Overload a parameter value."""
#         params = {
#             TModel.DEFAULT_LR_KEY: 23.0,
#         }
#         model = TModel(None, ModeKeys.INFER, params)
#         params = model.get_params()
#         self.assertEqual(params[TModel.DEFAULT_LR_KEY], 23.0)

#     def test_overload_wrong_type(self):
#         """Overload a parmater but with a wrong type, raises a ValueError."""
#         params = {
#             TModel.DEFAULT_LR_KEY: 'hellone.',
#         }
#         self.assertRaises(ValueError, TModel, None, ModeKeys.INFER, params)

#     def test_feed(self):
#         """Test the feeding from input data files."""
#         tdf = TestDataFactory()
#         files = tdf.generate(num_files=2, num_examples=10)
#         params = {
#             'model.class': 'tests.models.test_model.TModel',
#             'input.files': ','.join(files),
#             'input.batch_size': 5,
#         }
#         model = Model.factory(None, ModeKeys.TRAIN, params)
#         model._feed()
        
# class _BaseModel(model._Model):  # pylint: disable=W0212

#     _TARGET_KEY = 'TARGET'

#     def __init__(self, summary=True, output_mask=None):
#         super(_BaseModel, self).__init__()
#         self._summary = summary
#         self._tensors = None
#         self._logits = None
#         self._output_mask = output_mask

#     def get_default_hparams(self):
#         return tf.contrib.training.HParams(dim_0=10, dim_1=3, dim_2=7)

#     def _feed_helper(self, tensors):
#         self._tensors = copy.copy(tensors)
#         self._inputs = copy.copy(tensors)
#         self._target = self._inputs.pop(self._TARGET_KEY)

#     def _build_graph(self):
#         assert not self.built
#         shape = [self.hparams.dim_0, self.hparams.dim_1, self.hparams.dim_2]
#         self._logits = tf.random_normal(shape, name='Logits')
#         self._predictions = tf.identity(
#             tf.nn.softmax(self._logits),
#             name='Probabilities')
#         if self._summary:
#             tf.summary.scalar('SimpleSummary', tf.constant(23))


# class TestBaseModel(tf.test.TestCase):
#     """Test the functionality of the `dket.models.model.BaseModel` class."""

#     def test_global_step_initialization(self):
#         """Global step is set right after the model creation."""
#         instance = _BaseModel(output_mask=tf.ones([10, 20]))
#         self.assertIsNotNone(instance.global_step)
#         self.assertFalse(instance.fed)
#         self.assertFalse(instance.built)
#         self.assertIsNone(instance.hparams)
#         self.assertIsNone(instance.loss)
#         self.assertIsNone(instance.optimizer)
#         self.assertIsNone(instance.metrics)
#         self.assertIsNone(instance.inputs)
#         self.assertIsNone(instance.target)
#         self.assertIsNone(instance.predictions)
#         self.assertIsNone(instance.train_op)
#         self.assertIsNone(instance.summary_op)

#     def test_get_default_hparams(self):
#         """The method `get_default_hparams` should be invocable as the model is created."""
#         instance = _BaseModel(output_mask=tf.ones([10, 20]))
#         self.assertIsNotNone(instance.get_default_hparams())
#         self.assertIsNotNone(instance.global_step)
#         self.assertFalse(instance.fed)
#         self.assertFalse(instance.built)
#         self.assertIsNone(instance.hparams)
#         self.assertIsNone(instance.loss)
#         self.assertIsNone(instance.optimizer)
#         self.assertIsNone(instance.metrics)
#         self.assertIsNone(instance.inputs)
#         self.assertIsNone(instance.target)
#         self.assertIsNone(instance.predictions)
#         self.assertIsNone(instance.train_op)
#         self.assertIsNone(instance.summary_op)

#     def test_feed(self):
#         """Feed the model with tensors."""
#         instance = _BaseModel(output_mask=tf.ones([10, 20]))
#         with tf.variable_scope('Inputs'):
#             inputs = {
#                 'A': tf.constant(23, dtype=tf.int32),
#                 'B': tf.constant(47, dtype=tf.int32),
#             }
#             target = tf.constant(90, dtype=tf.int32)

#             tensors = copy.copy(inputs)
#             tensors['TARGET'] = target

#         instance.feed(tensors)
#         self.assertTrue(instance.fed)
#         self.assertEqual(inputs, instance.inputs)
#         self.assertEqual(target, instance.target)
#         self.assertEqual(tensors, instance.feeding)

#         self.assertFalse(instance.built)
#         self.assertIsNone(instance.hparams)
#         self.assertIsNone(instance.loss)
#         self.assertIsNone(instance.optimizer)
#         self.assertIsNone(instance.metrics)
#         self.assertIsNone(instance.predictions)
#         self.assertIsNone(instance.train_op)
#         self.assertIsNone(instance.summary_op)

#         # If you try feeding the model twice, you
#         # will have a RuntimeError.
#         self.assertRaises(RuntimeError, instance.feed, tensors)

#     def test_feed_with_none_args(self):
#         """Test feeding the model with `None` inputs or target."""
#         instance = _BaseModel(output_mask=tf.ones([10, 20]))
#         self.assertRaises(ValueError, instance.feed, tensors=None)

#     def test_build_trainable(self):
#         """Test the building of a trainable model."""

#         mask = tf.ones([10, 20])
#         instance = _BaseModel(output_mask=mask)
#         with tf.variable_scope('Inputs'):
#             tensors = {
#                 'A': tf.constant(23, dtype=tf.int32),
#                 'B': tf.constant(47, dtype=tf.int32),
#                 'TARGET': tf.constant(90, dtype=tf.int32)
#             }
#         instance.feed(tensors)

#         hparams = tf.contrib.training.HParams(dim_0=2, dim_1=4, extra='Ciaone')

#         loss_batch_value = tf.constant(0.0, dtype=tf.float32)
#         loss = mock.Mock()
#         type(loss).batch_value = mock.PropertyMock(
#             return_value=loss_batch_value)

#         optimizer = mock.Mock()
#         train_op = tf.no_op('train_op')
#         optimizer.minimize.side_effect = [train_op]

#         metrics_01 = mock.Mock()
#         metrics_02 = mock.Mock()

#         metrics = {
#             'metrics_01': metrics_01,
#             'metrics_02': metrics_02
#         }

#         instance.build(hparams, loss, optimizer, metrics)

#         self.assertTrue(instance.built)
#         self.assertTrue(instance.trainable)
#         self.assertEqual(hparams.dim_0, instance.hparams.dim_0)
#         self.assertEqual(hparams.dim_1, instance.hparams.dim_1)
#         self.assertEqual(instance.get_default_hparams().dim_2,
#                          instance.hparams.dim_2)
#         self.assertFalse('extra' in instance.hparams.values())
#         self.assertIsNotNone(instance.predictions)

#         loss.compute.assert_called_once_with(
#             instance.target, instance.predictions, weights=mask)

#         optimizer.minimize.assert_called_once_with(
#             loss_batch_value, global_step=instance.global_step)
#         self.assertEqual(train_op, instance.train_op)

#         metrics_01.compute.assert_called_once_with(
#             instance.target, instance.predictions, weights=mask)
#         metrics_02.compute.assert_called_once_with(
#             instance.target, instance.predictions, weights=mask)

#         self.assertIsNotNone(instance.summary_op)

#         self.assertRaises(RuntimeError, instance.build,
#                           hparams, loss, optimizer, metrics)

#     def test_build_not_trainable_loss(self):
#         """Test the building of a non-trainable model with loss."""

#         instance = _BaseModel()
#         with tf.variable_scope('Inputs'):
#             tensors = {
#                 'A': tf.constant(23, dtype=tf.int32),
#                 'B': tf.constant(47, dtype=tf.int32),
#                 'TARGET': tf.constant(90, dtype=tf.int32)
#             }
#         instance.feed(tensors)

#         hparams = tf.contrib.training.HParams(dim_0=2, dim_1=4, extra='Ciaone')

#         loss = mock.Mock()

#         metrics_01 = mock.Mock()
#         metrics_02 = mock.Mock()

#         metrics = {
#             'metrics_01': metrics_01,
#             'metrics_02': metrics_02
#         }

#         self.assertRaises(ValueError, instance.build,
#                           hparams, loss=loss, optimizer=None)

#     def test_build_not_trainable(self):
#         """Test the building of a non-trainable model without loss."""
#         mask = tf.ones([10, 20])
#         instance = _BaseModel(output_mask=mask)
#         with tf.variable_scope('Inputs'):
#             tensors = {
#                 'A': tf.constant(23, dtype=tf.int32),
#                 'B': tf.constant(47, dtype=tf.int32),
#                 'TARGET': tf.constant(90, dtype=tf.int32),
#             }
#         instance.feed(tensors)

#         metric = mock.Mock()
#         metrics = {'metric': metric}

#         hparams = tf.contrib.training.HParams(dim_0=2, dim_1=4, extra='Ciaone')
#         instance.build(hparams, loss=None, optimizer=None, metrics=metrics)

#         self.assertFalse(instance.trainable)
#         self.assertIsNone(instance.loss)
#         self.assertIsNone(instance.optimizer)
#         self.assertIsNone(instance.train_op)
#         self.assertIsNone(instance.summary_op)

#         # assert that, in inference mode, the metrics are not
#         # invoked with the output mask as `weights` but with `None`.
#         metric.compute.assert_called_once_with(
#             instance.target, instance.predictions, weights=None)

#     def test_build_trainable_without_loss(self):  # pylint: disable=I0011,C0103
#         """Built a model with an optimizer but without a loss function."""

#         instance = _BaseModel()
#         with tf.variable_scope('Inputs'):
#             tensors = {
#                 'A': tf.constant(23, dtype=tf.int32),
#                 'B': tf.constant(47, dtype=tf.int32),
#                 'TARGET': tf.constant(90, dtype=tf.int32)
#             }
#         instance.feed(tensors)

#         hparams = tf.contrib.training.HParams(dim_0=2, dim_1=4, extra='Ciaone')

#         optimizer = mock.Mock()
#         train_op = tf.no_op('train_op')
#         optimizer.minimize.side_effect = [train_op]

#         self.assertRaises(ValueError, instance.build,
#                           hparams, loss=None, optimizer=optimizer)

#     def test_build_not_fed(self):
#         """Build a model which has not been fed."""
#         instance = _BaseModel()
#         hparams = instance.get_default_hparams()
#         self.assertFalse(instance.fed)
#         self.assertRaises(RuntimeError, instance.build, hparams)

#     def test_build_trainable_without_summaries(self):  # pylint: disable=I0011,C0103
#         """Test that a trainable model always has a summary_op."""
#         instance = _BaseModel(summary=False)
#         with tf.variable_scope('Inputs'):
#             tensors = {
#                 'A': tf.constant(23, dtype=tf.int32),
#                 'B': tf.constant(47, dtype=tf.int32),
#                 'TARGET': tf.constant(90, dtype=tf.int32)
#             }
#         instance.feed(tensors)

#         hparams = tf.contrib.training.HParams(dim_0=2, dim_1=4, extra='Ciaone')

#         loss = mock.Mock()

#         optimizer = mock.Mock()
#         train_op = tf.no_op('train_op')
#         optimizer.minimize.side_effect = [train_op]

#         instance.build(hparams, loss, optimizer)
#         self.assertIsNone(tf.summary.merge_all())
#         self.assertIsNotNone(instance.summary_op)

#     def test_build_without_hparams(self):
#         """Test the building of a model without hparams."""
#         instance = _BaseModel(summary=False)
#         with tf.variable_scope('Inputs'):
#             tensors = {
#                 'A': tf.constant(23, dtype=tf.int32),
#                 'B': tf.constant(47, dtype=tf.int32),
#                 'TARGET': tf.constant(90, dtype=tf.int32)
#             }
#         instance.feed(tensors)

#         loss = mock.Mock()

#         optimizer = mock.Mock()
#         train_op = tf.no_op('train_op')
#         optimizer.minimize.side_effect = [train_op]

#         self.assertRaises(ValueError, instance.build, None,
#                           loss=loss, optimizer=optimizer)

#     def test_build_without_metrics(self):
#         """Test the building without metrics."""
#         instance = _BaseModel(summary=False)
#         with tf.variable_scope('Inputs'):
#             tensors = {
#                 'A': tf.constant(23, dtype=tf.int32),
#                 'B': tf.constant(47, dtype=tf.int32),
#                 'TARGET': tf.constant(90, dtype=tf.int32)
#             }
#         instance.feed(tensors)

#         loss = mock.Mock()

#         optimizer = mock.Mock()
#         train_op = tf.no_op('train_op')
#         optimizer.minimize.side_effect = [train_op]

#         instance.build(instance.get_default_hparams(), loss, optimizer)
#         self.assertIsNone(instance.metrics)


if __name__ == '__main__':
    tf.test.main()
