"""Optimizers for `dket` models."""

import tensorflow as tf

from dket import ops


class Optimizer(object):
    """Base implementation of the optimizer function."""

    @staticmethod
    def _no_summary(var):
        pass

    def __init__(self, optimizer, clip, colocate=False, summarize=ops.summarize):
        """Initialize the new optimizer instance.

        Arguments:
          clip: a function getting a `Tensor` in input representing a gradient and returning
            a `Tensor` of the same shape of the input and compatible `DType`, representing a
            clipped version of the gradient.
          colocate: if `True`, try to colocate gradients and ops.
          summarize: a function accepting a `Tensor` representing a gradient as input; this
            function is just intended to generate summaries for the input `Tensor`. The
            default one is `dket.ops.summarize`.
        """
        self._optimizer = optimizer
        self._clip = clip
        self._colocate = colocate
        self._summarize = summarize or Optimizer._no_summary

    @property
    def optimizer(self):
        """The wrapped `tf.train.Optimizer`."""
        return self._optimizer

    @property
    def clip(self):
        """The wrapped gradient clipping function (if any)."""
        return self._clip

    @property
    def colocate(self):
        """If `True`, try to colocate gradients and ops."""
        return self._colocate

    @property
    def summarize(self):
        """The summarization function invoked on gradients (and clipped)."""
        return self._summarize

    def minimize(self, loss, variables=None, global_step=None):
        """Minimize the loss w.r.t. the variables.

        Arguments:
          loss: an `Op` representing the loss function to be minimized.
          variables: a list of `tf.Variable` w.r.t. which to differentiate the loss;
            if `None`, the variables in the `tf.GraphKeys.TRAINABLE_VARIABLES` will
            be automatically used.
          global_step: a `0-D` (i.e. scalar) `Tensor` that represents the global step of
            the model; if provided, will be automatically incremented at each training step.

        Returns:
          an `Op` representing the training step; if the `global_step` argument has been
          provided, this will be incremented at each training step.
        """

        variables = variables or tf.trainable_variables()

        grads_and_vars = self._optimizer.compute_gradients(
            loss, variables, colocate_gradients_with_ops=self._colocate)

        # Remove the None gradiend which might have come from
        # variables actually not influencing the loss value.
        grads_and_vars = [(g, v) for g, v in grads_and_vars if g is not None]

        for grad, _ in grads_and_vars:
            self._summarize(grad)

        if self._clip is not None:
            for i, grad_and_var in enumerate(grads_and_vars):
                grad, var = grad_and_var
                clipped_grad = self._clip(grad)
                self._summarize(clipped_grad)
                grads_and_vars[i] = (clipped_grad, var)

        return self.optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

    @staticmethod
    def sgd(learning_rate, clip=None, colocate=True, summarize=ops.summarize):
        """Create a Stochastic Gradient Descent optimizer.

        Arguments:
          learning_rate: a `float` or `0-D` (i.e. scalar) `Tensor`.
          clip: a function getting a `Tensor` in input representing a gradient and returning
            a `Tensor` of the same shape of the input and compatible `DType`, representing a
            clipped version of the gradient.
          colocate: if `True`, try to colocate gradients and ops.
          summarize: a function accepting a `Tensor` representing a gradient as input; this
            function is just intended to generate summaries for the input `Tensor`.

        Returns:
          a `dket.model.Optimizer` instance implementing the Gradient Descent algorithm.
        """
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        return Optimizer(optimizer, clip=clip, colocate=colocate, summarize=summarize)
