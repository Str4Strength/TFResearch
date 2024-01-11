import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow_probability as tfp
import numpy as np

import re
import math
import random

from functools import partial
from termcolor import cprint

from .Neural_Networks import *




R_D = 0.1
R_Q = 0.1



def network(
        latent,
        dims=[1024, 1024, 1024, 1024,],
        lrmul=1,
        update_emas=True,
        momentum=0.99,
        mask=None,
        train=True,
        scope='map_net',
        reuse=tf.AUTO_REUSE
        ):
    with tf.variable_scope(scope, reuse=reuse):

        latent = tf.to_float(latent)
        dtype = latent.dtype

        vector = p_normalize(latent, p=2, axis=-1, epsilon=1e-8)
        prev_dims = [shape(vector)[-1]] + dims[:-1]

        for n, D in enumerate(dims):
            with tf.variable_scope(f'map_layer_{n}'):

                #vector, _ = feed_forward(vector, dims[n], hidden_features=dims[n] + prev_dims[n] // 2, activation=mish,
                #        dropout_rate=R_D, lrmul=lrmul, quantization=R_Q, trainable=train, scope='ff')
                vector, mask = linear(vector, out_features=dims[n] + prev_dims[n] // 2, mask=mask,
                        lrmul=lrmul, quantization=R_Q, quantization_blocks=shape(vector)[-1],
                        trainable=train, scope='feed_0')

                vector = mish(vector) * mask if exists(mask) else mish(vector)

                vector, mask = linear(vector, out_features=dims[n], mask=mask,
                        lrmul=lrmul, quantization=R_Q, quantization_blocks=shape(vector)[-1],
                        trainable=train, scope='feed_1')

                if update_emas:
                    ema = tf.get_variable('ema', shape=[shape(vector)[-1]], dtype=dtype, trainable=False,
                            initializer=tf.zeros_initializer)
                    if trainable:
                        ema_new = ((1. - momentum) * tf.reduce_mean(vector, axis=0)) + (momentum * ema) # dims
                        with tf.control_dependencies([tf.assign(ema, ema_new)]):
                            ema_norm = ema_new
                    else:
                        ema_norm = ema
                    vector *= tf.math.rsqrt(tf.maximum(tf.reduce_mean(tf.math.square(ema_norm)), 1e-8))
                else:
                    vector = p_normalize(vector, p=2, axis=-1, epsilon=1e-8)

            if exists(mask): vector *= mask

        return vector, mask
