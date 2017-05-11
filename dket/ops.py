"""Additional ops."""

import tensorflow as tf


EPSILON = 1e-6


def softmax_xent_with_logits(truth, logits, weights=1.0,
                             scope='CrossEntropy',
                             loss_collection=tf.GraphKeys.LOSSES):
    """Computes the softmax cross entropy between `truth` and `logits`.

    Arguments:
      truth: the truth label `Tensor` of `DType` `tf.int32` or `tf.int64`.
        If is of the same rank of `logits`, i.e. `[d_0, d_1, ..., d_{r-1}, num_classes]
        will be interpreted as a one-hot version of the labels. If it is one dimension less,
        i.e. `[d_0, d_1, ..., d_{r-1}` will be interpreted as a sparse labels tensor.
      logits:  a `Tensor` with unscaled log probabilities of shape
        `[d_0, d_1, ..., d_{r-1}, num_classes]` and dtype `float32` or `float64`.
      weights: coefficients for the loss. This must be scalar or of same rank as `labels`.
      scope: `str` or `tr.VariableScope`, is the scope for the operations
        performed in computing the loss.
      loss_collection: `str`, key for the collection to which the loss will be added.

    Returns:
      A `Tensor` representing the mean loss value.

    Raises:
      ValueError: if the rank of the `truth` tensor is greater than the rank of the
        `logits` tensor or minor of more than one.
    """

    trank = truth.get_shape().ndims
    lrank = logits.get_shape().ndims

    if trank > lrank or lrank - trank > 1:
        raise ValueError("""Rank of `truth` Tensor is %d while rank of the
                            `logits` tensor is %d. Rank of the `truth` tensor must
                            be equals or one less the rank of the `logits` tensor."""
                         % (trank, lrank))

    if lrank - trank == 1:
        return tf.losses.sparse_softmax_cross_entropy(
            labels=truth, logits=logits, weights=weights,
            scope=scope, loss_collection=loss_collection)

    if lrank == trank:
        return tf.losses.softmax_cross_entropy(
            onehot_labels=truth, logits=logits, weights=weights,
            scope=scope, loss_collection=loss_collection)
