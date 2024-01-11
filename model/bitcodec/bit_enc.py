import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np

import re
import math
import random

from functools import partial
from termcolor import cprint

from ..Neural_Networks import *
from ..block_setting import *



R_Q = 0.1
R_D = 0.1



def snake(
        tensor,
        mask = None,
        trainable = True,
        scope = 'snake'
        ):
    with tf.variable_scope(scope):
        alpha = bias(
                shape = shape(tensor)[-1],
                dtype = tensor.dtype,
                trainable = trainable,
                scope = 'alpha'
                )
        alpha = tf.reshape(
                alpha,
                [1] * (len(shape(tensor)) - 1) + shape(tensor)[-1:]
                )
        tensor += tf.square(tf.sin(alpha * tensor)) / tf.maximum(alpha, 1e-8)
        if exists(mask): tensor *= mask

    return tensor



def res_blk(
        tensor,
        features,
        kernels_list = [3, 1],
        dilations_list = [1, 1],
        compress = 2,
        weight_func = None,
        bias_func = None,
        mask = None,
        trainable = True,
        scope = 'res_blk',
        ):
    with tf.variable_scope(scope):
        assert len(kernels_list) == len(dilations_list)
        res_out, sc_out, res_mask, sc_mask = (* (tensor,) * 2, * (mask,) * 2)

        with tf.variable_scope('residual'):
            for n, (kernels, dilations) in enumerate(zip(kernels_list, dilations_list)):
                res_out = snake(
                        tensor = res_out,
                        mask = res_mask,
                        trainable = trainable,
                        scope = f'snake_{n:02d}'
                        )
                res_out, res_mask = convolution(
                    tensor = res_out,
                    rank = 1,
                    filters = features // (1 if n == len(kernels_list) - 1 else compress),
                    kernels = kernels,
                    dilations = dilations,
                    mask = res_mask,
                    quantization = R_Q,
                    quantization_blocks = kernels,
                    weight_function = weight_func,
                    bias_function = bias_func,
                    trainable = trainable,
                    scope = f'conv1d_{n:02d}'
                    )

        sc_out, sc_mask = shortcut(
                tensor = sc_out,
                features = features,
                weight_function = weight_func,
                bias_function = bias_func,
                mask = sc_mask,
                blocks = shape(sc_out)[-1],
                trainable = trainable,
                scope = 'skip'
                )

        tensor = (lambda a, b: a+ b)(
                *map(lambda a, b: a * b,
                    (res_out, sc_out),
                    res_scale(0.9, tensor.dtype, trainable = trainable, scope = 'coeff'))
                )

        if exists(mask): tensor *= sc_mask

    return tensor, sc_mask



def network(
        tensor,
        features,
        filters = 32,
        num_layers_list = [1, 1, 1, 1],
        ratios_list = [8, 5, 4, 2],
        kernels_list = [7, 3, 7],
        dilation_base = 2,
        compress = 2,
        weight_func = None,
        bias_func = None,
        mask = None,
        training = True,
        scope = 'bit_enc',
        reuse = tf.AUTO_REUSE,
        ):
    with tf.variable_scope(scope, reuse):
        assert len(num_layers_list) == len(ratios_list)

        pad_length = -shape(tensor)[1] % np.prod(ratios_list)
        padded = tf.pad(tensor, ((0, 0), (0, pad_length)))[Ellipsis, None]
        mask_pad = tf.pad(mask, ((0, 0), (0, pad_length)))[Ellipsis, None] if exists(mask) else None
        multiplier = 1

        encoded, mask_enc = convolution(
                tensor = padded,
                rank = 1,
                filters = multiplier * filters,
                kernels = kernels_list[0],
                mask = mask_pad
                trainable = training,
                scope = 'conv1d_in'
                )

        for idx, (ratio, num_layers) in enuemrate(zip(reversed(ratios_list), num_layers_list)):
            with tf.variable_scope(f'{idx:02d}_multi_{multiplier:03d}'):
                for n_layer in range(num_layers):
                    encoded, mask_enc = res_blk(
                            tensor = encoded,
                            features = multiplier * filters,
                            kernels_list = [kernels_list[1], 1],
                            dilations_list = [dilation_base ** n_layer, 1],
                            compress = compress,
                            weight_func = weight_func,
                            bias_func = bias_func,
                            mask = mask_enc,
                            trainable = training,
                            scope = f'blk_{n_layer:02d}',
                            )

                encoded = snake(
                        tensor = encoded,
                        mask = mask_enc,
                        trainable = training,
                        scope = 'snake'
                        )
                encoded, mask_enc = convolution(
                        tensor = encoded,
                        rank = 1,
                        filters = multiplier * filters * 2,
                        kernels = ratio * 2,
                        strides = ratio,
                        mask = mask_enc,
                        weight_function = weight_func,
                        bias_function = bias_func,
                        trainable = training,
                        scope = 'conv1d_down'
                        )
                multiplier *= 2

        encoded = snake(
                tensor = encoded,
                mask = mask_enc,
                trainable = training,
                scope = 'snake'
                )
        encoded, mask_enc = convolution(
                tensor = encoded,
                kernels = kernels_list[2],
                weight_function = weight_func,
                bias_function = bias_func,
                trainable = traning,
                scope = 'conv1d_enc')
        encoded = tf.tanh(encoded)
        if exists(mask): encoded *= mask_enc

    return encoded, mask_enc



