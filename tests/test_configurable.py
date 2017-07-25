"""Test module for dket.configurable."""

import sys
import unittest

import tensorflow as tf

from dket import configurable


MKEYS = tf.contrib.learn.ModeKeys


class TestMerge(unittest.TestCase):
    """Test the merge() function."""

    def test_no_params(self):
        """No actual params given."""
        dparams = {
            'seed': 23,
            'foo': 'ciaone proprio!'
        }

        actual = configurable.merge(dparams, None)
        self.assertEqual(dparams, actual)
        self.assertFalse(actual is dparams)

        actual = configurable.merge(dparams, {})
        self.assertEqual(dparams, actual)
        self.assertFalse(actual is dparams)

    def test_key_not_in_default(self):
        """A param key not in default raises a Value error."""
        dparams = {'seed': 23, 'foo': 'bar'}
        params = {'seed': 23, 'newkey': 'raises a ValueError'}
        self.assertRaises(ValueError, configurable.merge, dparams, params)

    def test_none_as_default(self):
        """If `None` is a default param value, it is overwritten."""
        dparams = {'seed': 23, 'foo': None, 'bar': 'baz'}
        params = {'seed': 23, 'foo': 'raise a Warning and carry on!'}
        merged = configurable.merge(dparams, params)

        self.assertEqual(len(dparams), len(merged))
        for key in dparams:
            self.assertIn(key, merged)
        for key in merged:
            act = merged[key]
            if key in params:
                exp = params[key]
            else:
                exp = dparams[key]
            self.assertEqual(exp, act)

    def test_empty_dict_as_default(self):
        """If a default param is an empty string, it gets overriden."""
        dparams = {'seed': 23, 'foo': {}}
        params = {'foo': {'seven': 7, 'eleven': 11}}
        merged = configurable.merge(dparams, params)
        self.assertEqual(merged['foo'], params['foo'])

    def test_dict_as_default_wrong_type(self):
        """If the default param type is a dict, the actual must be as well."""
        dparams = {'seed': 23, 'foo': {}}
        params = {'foo': 'will raise an error.'}
        self.assertRaises(ValueError, configurable.merge, dparams, params)

    def test_dict_param_merged(self):
        """If the param is a dict, merge it with the default one."""
        dparams = {'seed': 23, 'foo': {'bar': 7, 'baz': 11}}
        params = {'foo': {'bar': 9}}
        merged = configurable.merge(dparams, params)
        dexp = dparams['foo']
        exp = params['foo']
        act = merged['foo']
        for key, value in act.items():
            if key in exp:
                self.assertEqual(value, exp[key])
            elif key in dexp:
                self.assertEqual(value, dexp[key])
            else:
                raise RuntimeError('Un-mapped key %s', key)

    def test_override_with_none(self):
        """`None` can override all the values."""
        dparams = {'seed': 23, 'foo': {'bar': 7, 'baz': 11}, 'text': 'fates warning.'}
        params = {'seed': None, 'foo': None, 'text': None}
        merged = configurable.merge(dparams, params)
        for key, value in merged.items():
            self.assertIn(key, params)
            self.assertIsNone(value)

    def test_override_wrong_type(self):
        """The actual param value is not castable into the default type."""
        dparams = {'seed': 23}
        params = {'seed': 'cannot parse into a number.'}
        self.assertRaises(ValueError, configurable.merge, dparams, params)


class _Configurable(configurable.Configurable):

    @classmethod
    def get_default_params(cls):
        return {
            'seed': 23.0,
            'label': 'LABEL.',
            'foo': {
                'x': 1.0,
                'y': 1.0,
            }
        }

    def _validate_params(self, params):
        seed = params['seed']
        x = params['foo']['x']
        y = params['foo']['y']
        if x * y > seed:
            raise RuntimeError('x * y = ' + str(x * y) + ' > ' + str(seed))
        return params


class TestConfigurable(unittest.TestCase):
    """Main test case for the abstract Configurable class."""

    def test_no_params(self):
        """If no params are passed, default ones are used."""
        dparams = _Configurable.get_default_params()
        self.assertEqual(dparams, _Configurable(MKEYS.TRAIN, {}).get_params())
        self.assertEqual(dparams, _Configurable(MKEYS.INFER, {}).get_params())
        self.assertEqual(dparams, _Configurable(MKEYS.EVAL, {}).get_params())
        self.assertEqual(dparams, _Configurable(MKEYS.TRAIN, None).get_params())
        self.assertEqual(dparams, _Configurable(MKEYS.INFER, None).get_params())
        self.assertEqual(dparams, _Configurable(MKEYS.EVAL, None).get_params())

    def test_mode(self):
        """Use all the valid mode values and invalid ones."""
        self.assertEqual(_Configurable(MKEYS.TRAIN, {}).mode, MKEYS.TRAIN)
        self.assertEqual(_Configurable(MKEYS.INFER, {}).mode, MKEYS.INFER)
        self.assertEqual(_Configurable(MKEYS.EVAL, {}).mode, MKEYS.EVAL)
        self.assertRaises(ValueError, _Configurable, 'invalid', {})

    def test_default(self):
        """Check that params are actually merged."""
        params = _Configurable.get_default_params()
        params['seed'] = 100.0
        params['label'] = 'IronMaiden'
        params['foo']['x'] = 7.0
        params['foo']['y'] = 9.0
        self.assertEqual(_Configurable(MKEYS.TRAIN, params).get_params(), params)
        
    def test_invalid_params(self):
        """Test that invalid parameters actually rise an exception."""
        params = _Configurable.get_default_params()
        params['seed'] = 56.0
        params['label'] = 'IronMaiden'
        params['foo']['x'] = 7.0
        params['foo']['y'] = 9.0
        self.assertRaises(RuntimeError, _Configurable, MKEYS.TRAIN, params)


# pylint: disable=C0111,E0011
class DummyConf(configurable.Configurable):

    _call_args = []

    @classmethod
    def get_default_params(cls):
        return {
            'seed': 0,
            'label': 'Hello World!'
        }

    def _validate_params(self, params):
        return params

    @classmethod
    def get_create_call_args(cls):
        return cls._call_args

    @classmethod
    def call_args(cls):
        if cls._call_args:
            return cls._call_args[-1]
        return ()

    @classmethod
    def create(cls, mode, params):
        cls._call_args.append((mode, params))
        return cls(mode, params)
# pylint: enable=C0111,E0011


class TestResolve(unittest.TestCase):
    """Test the configurable.resolve method."""

    _MODULE = sys.modules[__name__]
    _MODULE_NAME = __name__
    _CLZ = DummyConf.__name__
    _FULL_CLZ = __name__ + '.' + _CLZ
    _MODE_KEY = tf.contrib.learn.ModeKeys.TRAIN

    def test_resolve(self):
        """Test the configurable.resolve method."""
        self.assertEqual(DummyConf, configurable.resolve(self._CLZ, self._MODULE))
        self.assertEqual(DummyConf, configurable.resolve(self._CLZ, self._MODULE_NAME))
        self.assertEqual(DummyConf, configurable.resolve(self._FULL_CLZ))
        self.assertRaises(RuntimeError, configurable.resolve, self._CLZ)
        self.assertRaises(RuntimeError, configurable.resolve, type('NonConf').__name__, self._MODULE)


class TestFactory(unittest.TestCase):
    """Test the configurable.factory method."""

    def test_factory(self):
        """Test the configurable.factory method."""
        clz = __name__ + '.' + DummyConf.__name__
        module = sys.modules[__name__]
        mode = tf.contrib.learn.ModeKeys.TRAIN
        params = DummyConf.get_default_params()
        params['seed'] = 23
        params['label'] = 'ciaone proprio!'
        instance = configurable.factory(clz, mode, params, module)

        self.assertIsInstance(instance, DummyConf)
        self.assertEqual(mode, instance.mode)
        self.assertEqual(params, instance.get_params())
        self.assertEqual(1, len(DummyConf.get_create_call_args()))
        self.assertEqual((mode, params), DummyConf.call_args())


if __name__ == '__main__':
    unittest.main()
