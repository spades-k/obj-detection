import tensorflow as tf

from tensorflow.python.ops import math_ops
from  tensorflow.python.framework import ops

def safe_divide(numerator, denominator, name):
    """Divides two values, return 0 if the denominator is <= 0
    Args:
        numerator: A real `Tensor`
        denominator: A real `Tensor`, with dtype matching `numerator`
    Returns:
        0 if `denominator` <= 0, else `numerator` / `denominator`
    """
    return tf.where(
        math_ops.greater(denominator, 0),
        math_ops.divide(numerator, denominator),
        tf.zeros_like(numerator),
        name=name)

def cummax(x, reverse=False, name=None):
    """
    Compute the cumulative maximum of the tensor `x` along `axis`. This
    operation is similar to the more classic `cumsum`.Only support 1D tensor
    for now

    :param x:
    :param reverse:
    :param name:
    :return:
    """


    with ops.name_scope(name, "Cummax", [x]) as name:
        x = ops.convert_to_tensor(x, name='x')
        if reverse:
            x = tf.reverse(x, axis=[0])
        cmax = tf.scan(lambda a, y: tf.maximum(a, y), x,
                       initializer=None, parallel_iterations=1,
                       back_prop=False, swap_memory=False)
        if reverse:
            cmax = tf.reverse(cmax, axis=[0])
        return cmax


