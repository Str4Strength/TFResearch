import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

import re
import math
import random

from functools import partial
from termcolor import cprint

from .functions import *
from .layers import *
from .fast_attention import *



def network(
        tensor,
        features,
        hidden_features,
        tokens=5,
        time_visions=3,
        frequency_splits=3,
        #pad_values=0.0,
        mask=None,
        train=True,
        scope='encoder',
        reuse=tf.AUTO_REUSE
        ):
    with tf.variable_scope(scope, reuse=reuse):

        bs, length, fs = shape(tensor)

        # frequency_splitted patch
        t_visions = time_visions if isinstance(time_visions, int) else 3
        f_splits = frequency_splits if fs % (frequency_splits + 1) == 0 else 3

        tensor, mask = linear(tensor[Ellipsis, None], out_features=hidden_features//2,
                mask=mask[Ellipsis, None] if exists(mask) else None, trainable=train, scope='embedding')

        # f - splitted, t - amount viewed feature map
        #padded = tf.pad(tensor, ((0, 0), ((t_visions - 1) // 2, t_visions // 2), (0, 0)))
        #if exists(mask): m_pad = tf.pad(mask, ((0, 0), ((t_visions - 1) // 2, t_visions // 2), (0, 0)))

        tensor = mish(tensor)
        if exists(mask): tensor *= mask

        fmap, m_fmap = convolution(tensor, 2, filters=hidden_features,
                kernels=(t_visions, 2 * fs // (f_splits + 1)), strides=(1, fs // (f_splits + 1)),
                padding='valid', mask=mask, trainable=train, scope='conv_split')     # b * e, t, n_splits, hf

        fmap = normalization(fmap, 32, mask=m_fmap, trainable=train, scope='group_norm')
        fmap = mish(fmap)
        if exists(mask): fmap *= m_fmap

        # get prediction
        token_splits = weight_([f_splits, hidden_features], tensor.dtype, init=tf.ones_initializer, trainable=train,
                scope='class_tokens')
        token_q = tf.tile(token_splits[None], [bs, 1, 1]) # b * e, n_splits, hf

        styles = []
        if exists(mask): m_fsp = tf.split(m_fmap, f_splits, axis=-2)
        for n, tsp, fsp in zip(range(f_splits), *map(lambda x: tf.split(x, f_splits, axis=-2), (token_q, fmap))):
            msp = m_fsp[n][Ellipsis, 0, :] if exists(mask) else None     # tsp -> b * e, 1, hf     fsp -> b * e, t, hf
            style, _, _, _ = attention(tsp, fsp[Ellipsis, 0, :], fsp[Ellipsis, 0, :], out_features=features,
                    hidden_features=hidden_features, heads=hidden_features // 64, mask_key=msp, mask_value=msp, dropout=0.1,
                    quantization=0.1, quantization_blocks=hidden_features, trainable=train, scope=f"attn_{n}")
            styles.append(style)
            # n_splits * b * e, 1, f

        styles = tf.concat(styles, axis=-2)     # b * e, n_splits, f

        return styles



