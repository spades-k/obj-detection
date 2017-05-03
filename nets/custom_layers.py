import tensorflow as tf

def pad2d(inputs, pad=(0, 0), mode = 'CONSTANT', data_format='NHWC', trainable= True, scope=None):
    """2D Padding layer, adding a symmetric padding to H and W dimensions.

    Aims to mimic padding in Caffe and MXnet, helping the port models to Tensorflow.
    Tries to follow the naming convention of `tf.contrib.layers`.

    Args:
        inputs: 4D input Tensor;
        pad: 2-Tuple with padding values for H and W dimensions;
        mode: Padding mode. C.f. `tf.pad`
        data_format: NHWC or HCHW data format.
    """
    with tf.name_scope(scope, 'pad2d', [inputs]):
        #padding shape
        if data_format == 'NHWC':
            paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
        elif data_format == 'NCHW':
            paddings = [[0, 0], [0, 0], [pad[0], pad[0]], [pad[1], pad[1]]]
        net = tf.pad(inputs, paddings, mode=mode)
        return net

