import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

import re
import math
import random

from functools import partial
from termcolor import cprint

from ..Neural_Networks import *
from ..utils import tf_print



def blk(
        tensor,
        features,
        squeeze=(2, 2),
        activation=mish,
        pad_values=0.0,
        mask=None,
        trainable=True,
        scope='blk'
        ):
    with tf.variable_scope(scope):

        #sh, sw = tf.minimum(squeeze[0], shape(tensor)[1]), tf.minimum(squeeze[1], shape(tensor)[2])
        #kh, kw = tf.minimum(2 * sh, shape(tensor)[1]), tf.minimum(2 * sw, shape(tensor)[2])
        sh, sw = squeeze[0], squeeze[1]
        #kh, kw = 2 * sh, 2 * sw
        kh, kw = 3, 3
        tensor = tf.pad(tensor, ((0, 0), ((kh - 1) // 2, kh // 2), ((kw - 1) // 2, kw // 2), (0, 0)),
                constant_values=pad_values)
        if exists(mask): mask = tf.pad(mask, ((0, 0), ((kh - 1) // 2, kh // 2), ((kw - 1) // 2, kw // 2), (0, 0)))

        tensor, mask = convolution(tensor, 2, features, (kh, kw), strides=(sh, sw), padding='VALID',
                mask=mask, weight_function=partial(spectral_normalize, training=trainable), trainable=trainable,
                scope='conv2d')

        fmap = tensor

        tensor = activation(tensor)
        if exists(mask): tensor *= mask

        return tensor, mask, fmap


def single_network(
        tensor,
        features = 16,
        groups = 8,
        stddev_features = 32,
        squeezes = [(2, 2), (2, 2), (2, 2)],
        #pad_values = 0.0,
        mask = None,
        train = True,
        scope = 'disc_net',
        reuse = tf.AUTO_REUSE
        ):
    with tf.variable_scope(scope, reuse=reuse):

        fmaps, fmasks = [], []

        if len(shape(tensor)) < 4:
            tensor = tensor[Ellipsis, None]
            if exists(mask): mask = mask[Ellipsis, None]

        # patch
        tensor, mask = linear(tensor, out_features=features, mask=mask,
                weight_function=partial(spectral_normalize, training=train), trainable=train, scope='embedding')

        for n, sqz in enumerate(squeezes):
            tensor, mask, fmap = blk(tensor, shape(tensor)[-1] * 2, squeeze=sqz, activation=mish, pad_values=0.0,
                    mask=mask, trainable=train, scope=f'blk_{n}')

            if n != len(squeezes) - 1:
                fmap = normalization(fmap, group_size=1, scale=False, shift=False, mask=mask, trainable=True, scope='fmap_norm')
            fmaps.append(mish(fmap))
            fmasks.append(mask)

        tensor = tf.reshape(tensor, [shape(tensor)[0], shape(tensor)[-1]])

        stddev = tf.reshape(tensor, [groups, -1, stddev_features, shape(tensor)[-1]//stddev_features])
        stddev = tf.reduce_mean(tf.square(stddev - tf.reduce_mean(stddev, axis=0, keepdims=True)), axis=0)
        stddev = tf.reduce_mean(tf.sqrt(stddev + 1e-8), axis=2)
        stddev = tf.tile(stddev, [groups, 1])

        tensor = tf.concat([tensor, stddev], axis=1)
        tensor = linear(tensor, out_features=1, trainable=train, scope='linear')[0]

        return tensor, (fmaps, fmasks)


def multi_network(
        tensor,
        networks = 3,
        stddev_features=32,
        squeezes = [(2, 2), (2, 2), (2, 2)],
        mask = None,
        scope = 'multi_disc_net',
        **kwargs
        ):
    with tf.variable_scope(scope):

        if len(shape(tensor)) < 4:
            tensor = tensor[Ellipsis, None]
            if exists(mask): mask = mask[Ellipsis, None]

        input_sets = [(tensor, mask)]
        for n in range(networks - 1):
            pools = list(map(lambda rs: np.prod(rs), zip(*squeezes[:(n+1)])))     # pool_H, pool_W
            #kH, kW = 2 * pools[0], 2 * pools[1]
            #pHL, phR, pWL, pWR = (kH - 1) // 2, kH // 2, (kW - 1) // 2, kW // 2
            #pvalHL, pvalHR = tf.reduce_mean(tensor[:, :kH - pHL], axis=1, keepdims=True), tf.reduce_mean(tensor[:, pHR - kH:], axis=1, keepdims=True)     # each
            # TODO pooling kernel = 2 * pooling size
            #tensor = tf.nn.avg_pool2d(tensor, list(map(lambda v: 2 * v, pools)), pools, padding='VALID')
            #if exists(mask): mask = tf.nn.max_pool2d(mask, list(map(lambda v: 2 * v, pools)), pools, padding='VALID')
            # TODO pooling kernel = pooling size
            pooled_tensor = tf.nn.avg_pool2d(tensor, pools, pools, padding='VALID')
            if exists(mask): pooled_mask = tf.nn.max_pool2d(mask, pools, pools, padding='VALID')

            #with tf.control_dependencies([
            #    tf_print(f'pool_{n}', pooled_tensor),
            #    tf_print(f'mask_{n}', pooled_mask),
            #    ]): pooled_tensor = tf.identity(pooled_tensor)

            input_sets.append((pooled_tensor, pooled_mask))

        outs, fmaps, masks = [], [], []
        for n, (t, m) in enumerate(input_sets):
            score, (fs, ms) = single_network(t, mask=m, scope=f'single_disc_{n}', stddev_features=stddev_features // (2 ** n),
                    squeezes=squeezes[n:], **kwargs)

            #p_list = [[tf_print(f'fs_{n}_{m}', fs[m], color='cyan'), tf_print(f'ms_{n}_{m}', ms[m], color='yellow')] for m in range(len(fs))]
            #prt_list = []
            #for a in p_list: prt_list += a
            #with tf.control_dependencies([
            #    tf_print(f'score_{n}', score, color='green'),
            #    ] + prt_list
            #    ): score = tf.identity(score)

            outs.append(score)
            fmaps += fs
            masks += ms

        return tf.concat(outs, axis=-1), (fmaps, masks)


