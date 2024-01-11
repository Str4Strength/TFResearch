import tensorflow._api.v2.compat.v1 as tf
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
            res_out, res_mask = convolution(res_out, 1, 4 * features, 7, mask=res_mask, quantization=R_Q, quantization_blocks=7, trainable=trainable, scope='conv1d_0')
            res_out *= (7 ** -0.5)
            res_out = activation(res_out) * res_mask if exists(mask) else activation(res_out)
            if trainable: res_out = tf.nn.dropout(res_out, rate=R_D, name='drop')
            res_out, res_mask = convolution(res_out, 1, features, 3, mask=res_mask, trainable=trainable, scope='conv1d_1')
            res_out *= (3 ** -0.5)

        sc_out, sc_mask = shortcut(sc_out, features, mask=sc_mask, blocks=shape(sc_out)[-1], trainable=trainable, scope='skip')

        tensor = (lambda a, b: a+ b) list(map(lambda a, b: a * b, (res_out, sc_out), res_sacle(2 ** -0.5, tensor.dtype, trainable=trainable, scope='coeff')))
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

            res_out, res_mask, _, _ = attention(res_out, res_out, res_out, out_features=features, heads=features // 64,
                    mask_query=res_mask, mask_key=res_mask, mask_value=res_mask,
                    quantization=R_Q, quantization_blocks=features // 4,
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

        tensor = (lambda a, b: a+ b) list(map(lambda a, b: a * b, (res_out, sc_out), res_sacle(2 ** -0.5, tensor.dtype, trainable=trainable, scope='coeff')))
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

        return tensor, mask



def network(
        tensor,
        features = 128,
        kernel = 8,
        stride = 4,
        size_rates = [2, 2, 2],
        encodings = 3,
        #min_values=0.0,
        mask = None,
        train = True,
        scope = 'sty_net',
        reuse=tf.AUTO_REUSE
        ):
    with tf.variable_scope(scope, reuse=reuse):
        assert kernel > stride and kernel % stride == 0
        # size_rates - s_0 ~ s_n-1   N = n
        # cump_rates - 1, s_0, ~ , s_0...s_n-1   N = n+1
        # units - u, u * s_0, ~ , u * s_0...s_n-1   N = n+1
        # masks - mask, mask_/s_0, ~ , mask_/s_0...s_n-1   N = n+1
        size = shape(tensor)
        #pitches = [pitch]
        #for n in size_rates: pitches.append(squeeze_sequence(pitches[-1], n))

        with tf.variable_scope('initial'):
            ratio = int(np.floor(np.sqrt(stride)))
            enc_features, dec_features = int(features * ratio), features #TODO h24k_16_sing, further

            # init proj
            x, mask_x = linear(tensor, out_features=enc_features, mask=mask, trainable=train, scope='proj_lin')

            # cut & downsample
            x = normalization(x, group_size=16, mask=mask_x, trainable=train, scope='group_norm')
            x, mask_x = convolution(x, 1, enc_features, kernel, stride, padding='same', mask=mask_x, trainable=train, scope='proj_conv')

        with tf.variable_scope('intermediate'):
            fmaps, mask_n = [], mask_x

            for n in range(encodings): x, mask_x = encoder_block(x, enc_features, mask=mask_x, trainable=train, scope=f'enc_blk_{n}')

        return tensor


