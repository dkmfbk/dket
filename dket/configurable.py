"""Configurable object main infrastructure."""

import abc
import copy
import pydoc
import warnings


import six

import tensorflow as tf


def merge(default, params):
    """Merges the `params` and `default` into a new parameters dictionary."""

    merged = copy.deepcopy(default)
    if not params:
        return merged

    for key, value in params.items():

        # Check that the given param is actually
        # in the default parameter dictionary.
        if key not in merged:
            raise ValueError('%s is not a valid parameter.', key)
        dvalue = default[key]

        # If the default param value is None, issue a
        # warning and overwrite it with the actual value.
        if dvalue is None:
            warnings.warn('default value for key `{}` is `None`'.format(key))
            merged[key] = value
            continue

        if value is None:
            warnings.warn('default value for key `{}` is `None`'.format(key))
            merged[key] = value
            continue

        # If the default value is a dictionary:
        # 1. check that the actual param value is a dictionary
        #    as well otherwise raise an exception.
        # 2. otherwise, recursively merge them.
        # 3. if the default dictionary is empty, just copy the
        #    actual one as the param value.
        if isinstance(dvalue, dict):
            if not isinstance(value, dict):
                raise ValueError(
                    'expected {} for parameter `{}`, found {} instead.'\
                    .format(key, type({}), type(value)))
            if dvalue:
                value = merge(dvalue, value)

        # Cast the actual value into the default value type
        # and assign it to the current parameter key.
        merged[key] = type(dvalue)(value)
    return merged


def _validate_mode(mode):
    """If the mode is valid, retuns it, otherwise raises a ValueError."""
    tf.contrib.learn.ModeKeys.validate(mode)
    return mode


@six.add_metaclass(abc.ABCMeta)
class Configurable(object):
    """Configurable component."""

    def __init__(self, mode, params):
        self._mode = _validate_mode(mode)
        params = merge(self.get_default_params(), params)
        self._params = self._validate_params(params)

    @property
    def mode(self):
        """The mode for the class, one of train, infer, eval."""
        return self._mode

    def get_params(self):
        """A copy of the parameter dictionary for the current instance."""
        return copy.deepcopy(self._params)

    @abc.abstractclassmethod
    def get_default_params(cls):
        """Return the default parameter dictionary."""
        raise NotImplementedError()

    @abc.abstractmethod
    def _validate_params(self, params):
        """Validates the parameters and return the final params dictionary."""
        raise NotImplementedError()

    @classmethod
    def create(cls, mode, params):
        """Factory method for Configurable object."""
        return cls(mode, params)


def resolve(clz, module=None):
    """Resolve the configurable type."""

    unresolve = 'could not resolve type `{}`'
    notsubclass = 'resolved type {} is not subclass of {}'

    ctype = pydoc.locate(clz)
    if not ctype and module:
        if isinstance(module, six.string_types):
            clz = module + '.' + clz
        else:  # use it as a module
            clz = module.__name__ + '.' + clz
        ctype = resolve(clz)

    if not ctype:
        raise RuntimeError(unresolve.format(clz))
    if not issubclass(ctype, Configurable):
        raise RuntimeError(notsubclass.format(str(ctype), str(Configurable)))
    return ctype

def factory(clz, mode, params, module=None):
    """Factory method for generic configurable object."""
    return resolve(clz, module).create(mode, params)
