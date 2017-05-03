import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

from datasets import dataset_factoty
from deployment import model_deploy
from nets import nets_factory
import tf_utils

slim = tf.contrib.slim

DATA_FORMAT = 'NCHW'