"""Test module for the `dket.models.model` module."""

# pylint: disable=E1129

import tensorflow as tf

from dket import model
from tests import utils

TRAIN = tf.contrib.learn.ModeKeys.TRAIN
EVAL = tf.contrib.learn.ModeKeys.EVAL
INFER = tf.contrib.learn.ModeKeys.INFER


class TestModelInputs(tf.test.TestCase):
    """ModelInputs test case."""

    def test_no_params(self):
        """Build a ModelInputs instance without params."""
        modin = model.ModelInputs(TRAIN, {})
        self.assertIsNotNone(modin.get(modin.WORDS_KEY))
        self.assertIsNotNone(modin.get(modin.SENTENCE_LENGTH_KEY))
        self.assertIsNotNone(modin.get(modin.FORMULA_KEY))
        self.assertIsNotNone(modin.get(modin.FORMULA_LENGTH_KEY))

    def test_no_files(self):
        """No input files."""
        params = {
            model.ModelInputs.FILES_PK: '',
            model.ModelInputs.EPOCHS_PK: -1,
            model.ModelInputs.BATCH_SIZE_PK: 1,
            model.ModelInputs.SHUFFLE_PK: True,
            model.ModelInputs.SEED_PK: None
        }
        modin = model.ModelInputs(TRAIN, params)
        self.assertIsNotNone(modin.get(modin.WORDS_KEY))
        self.assertIsNotNone(modin.get(modin.SENTENCE_LENGTH_KEY))
        self.assertIsNotNone(modin.get(modin.FORMULA_KEY))
        self.assertIsNotNone(modin.get(modin.FORMULA_LENGTH_KEY))

    def test_epochs_lessthan_zero(self):
        """If epochs < 0 an exception is raised."""
        params = {
            model.ModelInputs.FILES_PK: 'ciao.txt',
            model.ModelInputs.EPOCHS_PK: -1
        }
        self.assertRaises(ValueError, model.ModelInputs, TRAIN, params)

    def test_batch_lesseqthan_zero(self):
        """If batch_size <= 0 an exception is raised."""
        params = {
            model.ModelInputs.FILES_PK: 'ciao.txt',
            model.ModelInputs.BATCH_SIZE_PK: 0
        }
        self.assertRaises(ValueError, model.ModelInputs, TRAIN, params)
        
        params = {
            model.ModelInputs.FILES_PK: 'ciao.txt',
            model.ModelInputs.BATCH_SIZE_PK: -1
        }
        self.assertRaises(ValueError, model.ModelInputs, TRAIN, params)

    def test_read_from_files(self):
        """Read from data files."""
        tdf = utils.TestDataFactory()
        files = tdf.generate(num_files=2, num_examples=10)

        params = model.ModelInputs.get_default_params()
        params[model.ModelInputs.FILES_PK] = ','.join(files)
        params[model.ModelInputs.EPOCHS_PK] = None
        params[model.ModelInputs.BATCH_SIZE_PK] = 5

        modin = model.ModelInputs(TRAIN, params)
        self.assertIsNotNone(modin.get(modin.WORDS_KEY))
        self.assertIsNotNone(modin.get(modin.SENTENCE_LENGTH_KEY))
        self.assertIsNotNone(modin.get(modin.FORMULA_KEY))
        self.assertIsNotNone(modin.get(modin.FORMULA_LENGTH_KEY))
        tdf.cleanup()

class TModel(model.Model):
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
            self.inputs.get(model.ModelInputs.FORMULA_KEY), 
            self._params['num_classes'])

    @property
    def data(self):
        """Data variable."""
        return self._data

class TestModel(tf.test.TestCase):
    """Test case for the base model infrastructure."""

    def _test_default_train(self, mtype): 
        """Default test."""

        tmodel = mtype(TRAIN, {})
        self.assertIsNone(tmodel.graph)
        self.assertIsNone(tmodel.global_step)
        self.assertIsNone(tmodel.inputs)
        self.assertIsNone(tmodel.loss_op)
        self.assertIsNone(tmodel.train_op)
        self.assertIsNone(tmodel.summary_op)
        self.assertIsNone(tmodel.metrics)

        tmodel.build()
        self.assertIsNotNone(tmodel.graph)
        self.assertIsNotNone(tmodel.global_step)
        self.assertIsNotNone(tmodel.inputs)
        self.assertIsNotNone(tmodel.loss_op)
        self.assertIsNotNone(tmodel.train_op)
        self.assertIsNotNone(tmodel.summary_op)
        self.assertIsNotNone(tmodel.metrics)

    def _test_default_eval(self, mtype):
        """Test the model building in EVAL mode."""

        tmodel = mtype(EVAL, {})
        self.assertIsNone(tmodel.graph)
        self.assertIsNone(tmodel.global_step)
        self.assertIsNone(tmodel.inputs)
        self.assertIsNone(tmodel.loss_op)
        self.assertIsNone(tmodel.train_op)
        self.assertIsNone(tmodel.summary_op)
        self.assertIsNone(tmodel.metrics)

        tmodel.build()
        self.assertIsNotNone(tmodel.graph)
        self.assertIsNotNone(tmodel.global_step)
        self.assertIsNotNone(tmodel.inputs)
        self.assertIsNone(tmodel.loss_op)
        self.assertIsNone(tmodel.train_op)
        self.assertIsNone(tmodel.summary_op)
        self.assertIsNotNone(tmodel.metrics)

    def _test_default_infer(self, mtype):
        """Test the model building in EVAL mode."""
        tmodel = mtype(INFER, {})
        self.assertIsNone(tmodel.graph)
        self.assertIsNone(tmodel.global_step)
        self.assertIsNone(tmodel.inputs)
        self.assertIsNone(tmodel.loss_op)
        self.assertIsNone(tmodel.train_op)
        self.assertIsNone(tmodel.summary_op)
        self.assertIsNone(tmodel.metrics)

        tmodel.build()
        self.assertIsNotNone(tmodel.graph)
        self.assertIsNotNone(tmodel.global_step)
        self.assertIsNotNone(tmodel.inputs)
        self.assertIsNone(tmodel.loss_op)
        self.assertIsNone(tmodel.train_op)
        self.assertIsNone(tmodel.summary_op)
        self.assertIsNone(tmodel.metrics)

    def test_complete(self):
        """Test all the model types building in each mode."""
        tests = [
            self._test_default_train,
            self._test_default_eval,
            self._test_default_infer]
        mtypes = [
            TModel,
            model.PointingSoftmaxModel
        ]
        for mtype in mtypes:
            for test in tests:
                test(mtype)

if __name__ == '__main__':
    tf.test.main()
