import math
from collections import namedtuple

import numpy as np
import tensorflow as tf

import tf_extended as tfe
from nets import custom_layers
from nets import ssd_common
from nets import ssd_vgg_300

slim = tf.contrib.slim

SSDParams = namedtuple(
    'SSDParameters',
    [
        'img_shape',
        'num_classes',
        'no_annotation_label',
        'feat_layers',
        'feat_shapes',
        'anchor_size_bounda',
        'anchor_sizes',
        'anchor_raios',
        'anchor_steps',
        'anchor_offset',
        'normalizations',
        'prior_scaling'
    ]
)


class SSDNet(object):

    default_params = SSDParams(
        img_shape=(512, 512),
        num_classes=21,
        no_annotation_label=21,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11', 'block12'],
        feat_shapes=[(64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)],
        anchor_size_bounda=[0.10, 0.90],
        anchor_sizes=[(20.48, 51.2),
                      (51.2, 133.12),
                      (133.12, 215.04),
                      (215.04, 296.96),
                      (296.96, 378.88),
                      (378.88, 460.8),
                      (460.8, 542.72)],
        anchor_raios=[[2, .5],
                      [2, .5, 3, 1. / 3],
                      [2, .5, 3, 1. / 3],
                      [2, .5, 3, 1. / 3],
                      [2, .5, 3, 1. / 3],
                      [2, .5],
                      [2, .5]],
        anchor_steps=[8, 16, 32, 64, 128, 256, 512],
        anchor_offset=0.5,
        normalizations=[20, -1, -1, -1, -1, -1, -1],
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):

        if isinstance(params, SSDParams):
            self.params = params
        else:
            self.params = SSDNet.default_params

    def net(self, inputs,
            is_training = True,
            update_feat_shape = True,
            dropout_keep_prob = 0.5,
            prediction_fn = slim.softmax,
            reues = None,
            scope = 'ssd_512_vgg'):

        r = ssd_net(inputs)














def ssd_net(inputs,
            num_classes=SSDNet.default_params.num_classes,
            feat_layer=SSDNet.default_params.feat_layers,
            anchor_sizes=SSDNet.default_params.anchor_sizes,
            anchor_ratios=SSDNet.default_params.anchor_raios,
            normalizations=SSDNet.default_params.normalizations,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reues=None,
            scope='ssd_512_vgg'):
    """SSD net definition"""
    end_points = {}
    with tf.variable_scope(scope, 'ssd_512_vgg', [inputs], reuse=reues):
        # Original VGG-16 blocks
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        # Block 2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        # Block 3
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        # Block 4
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        # Block 5
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # Additional SSD blocks
        # Block 6
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6') #TODO: find the meaning of this rate
        end_points['block6'] = net
        # Block 7
