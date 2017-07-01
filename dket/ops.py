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
        with graph.as_default():
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
