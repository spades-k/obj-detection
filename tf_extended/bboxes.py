import numpy as np
import tensorflow as tf

from tf_extended import tensors as tfe_tensors
from tf_extended import math as tfe_math

def bboxes_all_classes(classes, scores, bboxes, top_k=400, scope=None):
    """Sort bounding boxes by decreasing order and keep only the top_k.
    Assume the input Tensor mix-up objects with different classes.\
    Assume a batch-type input.

    :param classes:Batch x N Tensor containing integer classes.
    :param scores:Batch x N Tensor containing float scores.
    :param bboxes:Batch x N x 4 Tensor containing boxes coordinates.
    :param top_k:Top_k boxes to keep
    :return:
        classes, scores, bboxes: Sorted tensors of shape Batch x Top_k
    """
    with tf.name_scope(scope, 'bboxes_sort', [classes, scores, bboxes]):
        scores, idxes = tf.nn.top_k(scores, k=top_k, sorted=True)

    # Trick to be able to use tf.gather: map for each element in the batch
    def fn_gather(classes, bboxes, idxes):
        cl = tf.gather(classes, idxes)
        bb = tf.gather(bboxes, idxes)
        return [cl, bb]
    r = tf.map_fn(lambda x: fn_gather(x[0], x[1], x[2]),  # TODO: I don't konw what this mean
                  [classes, bboxes, idxes],
                  dtype=[classes.dytpe, bboxes.dtype],
                  parallel_iterations=10,
                  back_prop=False,
                  swap_memory=False,
                  infer_shape=True)

    classes = r[0]
    bboxes = r[1]
    return classes, scores, bboxes


def bboxes_sort(scores, bboxes, top_k=400, scope=None):
    """
    Sort bounding boxes by decreasing order and keep only the top_k.
    If input are dectionnaries,assume every key is a different class.
    Assume a batch-type input.

    :param scores: Batch x N Tensor/Dictionary containing float scores.
    :param bboxes: Batch x N x 4 Tensor/Dictionary containing boxes coordinates.
    :param top_k:  Top_k boxes to keep
    :param scope:
    :return:
    """
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        # isinstance is to judge the dtype of a param
        with tf.name_scope(scope, 'bboxes_sort_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_sort(scores[c], bboxes[c], top_k=top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensor inputs
    with tf.name_scope(scope, 'bboxes_sort', [scores, bboxes]):

        scores, idxes = tf.nn.top_k(scores, k=top_k, sorted=True)

        def fn_gather(bboxes, idxes):
            bb = tf.gather(bboxes, idxes)
            return [bb]
        r = tf.map_fn(lambda x: fn_gather(x[0], x[1]),
                      [bboxes, idxes],
                      dtype=[bboxes.dtype],
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        bboxes = r[0]
        return scores, bboxes


def bboxes_clip(bbox_ref, bboxes, scope=None):
    """
    Clip bounding boxes to a reference box.
    Batch-compatible if the first demension of `bbox_ref` and `bboxes` can be
    broadcasted.

    :param bbox_ref:Reference bounding box. Nx4 or 4 shaped-Tensor;
    :param bboxes: Bounding boxes to clip. Nx4 or 4 shaped-Tensor or dictionary
    :return:clipped bboxes.
    """
    if isinstance(bboxes,dict):
        with tf.name_scope(scope, 'bboxes_clip_dict'):
            d_bboxes = {}
            for c in bboxes.keys():
                d_bboxes[c] = bboxes_clip(bbox_ref, bboxes[c])
            return d_bboxes

    # Tensor inputs
    with tf.name_scope(scope, 'bboxes_clip'):
        # Easier with transposed bboxes, Especially for broadcasting
        bbox_ref = tf.transpose(bbox_ref)
        bboxes = tf.transpose(bboxes)
        # Intersection bboxes an reference_bbox
        ymin = tf.maximum(bboxes[0], bbox_ref[0])
        xmin = tf.maximum(bboxes[1], bbox_ref[1])
        ymax = tf.minimum(bboxes[2], bbox_ref[2])
        xmax = tf.minimum(bboxes[3], bbox_ref[3])
        # Double check! Empty boxes when no-intersection
        ymin = tf.minimum(ymin, ymax)
        xmin = tf.minimum(xmin, xmax)
        bboxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax], axis=0))
        return bboxes

def bboxes_resize(bbox_ref, bboxes, name=None):
    """Resize bounding boxes based on a reference bounding box,
    assum that the latter is [0,0,1,1] after transform. Useful for
    updating a collection of boxes after cropping an image.
    """
    if isinstance(bboxes, dict):
        with tf.name_scope(name, 'bboxes_resize_dict'):
            d_bboxes = {}
            for c in bboxes.keys():
                d_bboxes[c] = bboxes_resize(bbox_ref, bboxes[c])
            return d_bboxes

    with tf.name_scope(name, 'bboxes_resize'):
        # Tanslate
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        # Scale
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes


























