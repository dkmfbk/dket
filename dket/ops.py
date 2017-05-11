"""Additional ops."""

import tensorflow as tf



def get_or_create_global_step(graph=None):
    """Get or create the global step for the given (or default) graph.

    Reads the `tf.GraphKeys.GLOBAL_STEP` collection  of the graph and
    returs the first element, assuming it is the actual global step. If the
    collection is empty, a`tf.Variable` with name `global_step` and initial
    value set to `0` is created and added to the collection.

    Arguments:
      graph: a `tf.Graph` instance, if `None` the defaul graph will be used.

    Return:
      the a `0-D` (scalar) `Tensor` representing the global step for graph.
    """
    graph = graph or tf.get_default_graph()
    global_step = get_global_step(graph=graph)
    if global_step is None:
        global_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='global_step')
        graph.add_to_collection(tf.GraphKeys.GLOBAL_STEP, global_step)
    return global_step


def get_global_step(graph=None):
    """Get the global step or `None` for the given (or default) graph.

    Reads the `tf.GraphKeys.GLOBAL_STEP` collection  of the graph and
    returs the first element, assuming it is the actual global step. If the
    collection is empty, `None` is returned.

    Arguments:
      graph: a `tf.Graph` instance, if `None` the defaul graph will be used.

    Return:
      the a `0-D` (scalar) `Tensor` representing the global step for graph, or `None`.
    """
    graph = graph or tf.get_default_graph()
    collection = graph.get_collection(tf.GraphKeys.GLOBAL_STEP)
    for global_step in collection:
        return global_step
    return None


def _mean_and_stddev(var):
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    return mean, stddev


def summarize(var, scope=None):
    """Attaches many summaries to a `Tensor`.

    Attaches many summaries to a `Tensor`. Such summaries are:
    * a scalar summary with the mean
    * a scalar summary with the standard deviation
    * a scalar summary with the maximum
    * a scalar summary with the minimum
    * an histogram summary

    Arguments:
      var: an arbitrary `Tensor`.
    """
    scope = scope or var.op.name
    with tf.name_scope(scope):
        mean, stddev = _mean_and_stddev(var)
        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


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
