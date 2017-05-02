import tensorflow as tf

def get_shape(x, rank=None):
    """Return the dimentions of a Tensor as list of integers or scale tensors.
    Args:
        x: N-d Tensor;
        rank: Rank of the Tensor. If None,will try to guess it
    Returns:
        A list of `[d1,d2,...,dN]` corresponding to the demensions of the
        input tensor.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape()
        if rank is None:
            static_shape = static_shape.as_list()
            rank = len(static_shape)
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

def pad_axis(x, offset, size, axis=0, name =None):
    """

    :param x: Tensor to pad;
    :param offset: offset to add on the dimension chosen;
    :param size: Final size of the dimension;
    :return:Padded tensor whose dimension on 'axis' is 'size', or greater if
            the input vector was lager.
    """
    with tf.name_scope(name, 'pad_axus'):
        shape = get_shape(x)
        rank = len(shape)
        new_size = tf.maximum(size - offset - shape[axis], 0)
        pad1 = tf.stack([0]*axis + [offset] + [0]*(rank - axis - 1))
        pad2 = tf.stack([0]*axis + [new_size] + [0]*(rank - axis - 1))
        paddings = tf.stack([pad1, pad2], axis=1)
        x = tf.pad(x, paddings, mode='CONSTANT')
        shape[axis] = size
        x = tf.reshape(x, tf.stack(shape))
        return x

