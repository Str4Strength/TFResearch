import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

import re
import math
import random

from functools import partial
from termcolor import cprint

from .Neural_Networks import *



R_Q = 0.1
R_D = 0.1

def res_scale(value, dtype, trainable=True, scope='res_scale'):
    scale = tf.get_variable(scope, [], dtype=dtype, initializer=tf.constant_initializer(value=value),
            trainable=trainable)
    scale = tf.minimum(tf.math.abs(scale), 1)
    return scale, tf.sqrt(1 - (scale ** 2))


def shortcut(
        tensor,
        features,
        mask=None,
        blocks=8,
        trainable=True,
        scope='shortcut'
        ):
    with tf.variable_scope('skip'):

        if features == shape(tensor)[-1]:
            sc_out, sc_mask = tensor, mask
        else:
            sc_out, sc_mask = linear(tensor, out_features=int(features), mask=mask, quantization=R_Q,
                    quantization_blocks=blocks, trainable=trainable, scope='proj')

        #sc_out = tf.where(tf.equal(sc_mask, 0), tf.zeros_like(sc_out), tf.nn.softmax(sc_out, axis=-1))

        return sc_out, sc_mask


def res_blk(
        tensor,
        features,
        activation=mish,
        mask=None,
        trainable=True,
        scope='res_blk',
        ):
    with tf.variable_scope(scope):
        res_out, sc_out, res_mask, sc_mask = (* (tensor,) * 2, * (mask,) * 2)

        with tf.variable_scope('residual'):
            res_out = normalization(res_out, group_size=16, mask=res_mask, trainable=trainable, scope='group_norm')
            #TODO 7 reduce, 4 * features 를 줄이기
            res_out, res_mask = convolution(res_out, 1, 2 * features, 5, mask=res_mask, quantization=R_Q, quantization_blocks=5, trainable=trainable, scope='conv1d_0')
            res_out *= (5 ** -0.5)
            res_out = activation(res_out) * res_mask if exists(mask) else activation(res_out)
            if trainable: res_out = tf.nn.dropout(res_out, rate=R_D, name='drop')
            res_out, res_mask = convolution(res_out, 1, features, 3, mask=res_mask, trainable=trainable, scope='conv1d_1')
            res_out *= (3 ** -0.5)

        sc_out, sc_mask = shortcut(sc_out, features, mask=sc_mask, blocks=shape(sc_out)[-1], trainable=trainable, scope='skip')

        #tensor = (res_out + sc_out) * (2 ** -0.5)
        rt, rs = res_scale(2 ** -0.5, tensor.dtype, trainable=trainable, scope='res_scalar')
        tensor = (rt * res_out) + (rs * sc_out)
        if exists(mask): tensor *= sc_mask

    return tensor, sc_mask


def attn_blk(
        tensor,
        features,
        mask=None,
        trainable=True,
        scope='attn_blk'
        ):
    with tf.variable_scope(scope):
        res_out, sc_out, res_mask, sc_mask = (* (tensor,) * 2, * (mask,) * 2)

        with tf.variable_scope('residual'):
            res_out = normalization(res_out, group_size=16, mask=res_mask, trainable=trainable, scope='group_norm')
            #cprint(f'{scope}_{features}_{shape(res_out)[-1]}', color='blue')
            res_out, res_mask, _, _ = attention(res_out, res_out, res_out, out_features=features, heads=features // 64,
                    mask_query=res_mask, mask_key=res_mask, mask_value=res_mask,
                    quantization=R_Q, quantization_blocks=shape(res_out)[-1] // 4,
                    trainable=trainable, scope='attn')

            #res_out, res_mask, _ = fast_attention(res_out, res_out, res_out, out_features=features, heads=features // 64,
            #        mask_query=res_mask, mask_key=res_mask, mask_value=res_mask,
            #        quantization=R_Q, quantization_blocks=features // 4,
            #        trainable=trainable, scope='fast_attn')

            #res_out, res_mask, _ = proximal_attention(tensor, res_out, res_out, window_size=7, out_features=features, heads=features // 64,
            #        mask_query=res_mask, mask_key=res_mask, mask_value=res_mask, quantization=R_Q, quantization_blocks=16, trainable=trainable,
            #        scope='prox_attn')

            #res_out, res_mask, _ = hybrid_fast_attention(tensor, res_out, res_out, window_size=7, out_features=features,
            #        full_heads=features // 128, prox_heads=features // 128, mask_query=res_mask, mask_key=res_mask, mask_value=res_mask,
            #        quantization=R_Q, quantization_blocks=16, trainable=trainable, scope='hybrid_attn')

        sc_out, sc_mask = shortcut(sc_out, features, mask=sc_mask, blocks=shape(sc_out)[-1], trainable=trainable, scope='skip')

        re_scale, sc_scale = res_scale(0, tensor.dtype, trainable=trainable, scope='res_scale')
        tensor = re_scale * res_out + sc_out #sc_scale * sc_out
        if exists(mask): tensor *= sc_mask

        return tensor, sc_mask, res_out


def encoder_block(
        tensor,
        features,
        mask=None,
        trainable=True,
        scope='enc_blk'
        ):
    with tf.variable_scope(scope):

        tensor, mask, _ = attn_blk(tensor, features, mask=mask, trainable=trainable, scope='attn')
        tensor, mask = res_blk(tensor, features, activation=mish, mask=mask, trainable=trainable, scope='res')
        #with tf.control_dependencies([tf_print(f'{scope}_res', tensor, color='green')]): tensor = tf.identity(tensor)
        #tensor, mask = res_blk(tensor, features, group_size=16, activation=mish, mask=mask, trainable=trainable, scope='res_1')

        return tensor, mask


def network(
        tensor,
        features,
        encodings,
        mask=None,
        train=True,
        scope='encoder',
        reuse=tf.AUTO_REUSE
        ):
    with tf.variable_scope(scope, reuse=reuse):

        bs, length, fs = shape(tensor)

        tensor, mask = linear(tensor, out_features=features, mask=mask, trainable=train, scope='embedding')

        x = normalization(tensor, 32, mask=mask, trainable=train, scope='group_norm')
        x = mish(x)
        if exists(mask): x *= mask

        for n in range(encodings): x, mask = encoder_block(x, features, mask=mask, trainable=train, scope=f'enc_blk_{n}')

        return x, mask



